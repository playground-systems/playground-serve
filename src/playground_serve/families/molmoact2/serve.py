"""MolmoAct2 (Ai2) worker: one checkpoint behind Playground Serve's websocket protocol (pgs_server.py).

Stock path: the repo's own servers at molmoact2_variants.COMMIT (examples/droid/host_server_droid.py and
examples/yam/host_server_yam.py, `run_droid.sh` / `run_yam.sh`): AutoProcessor (extra_special_tokens={}) and
AutoModelForImageTextToText with trust_remote_code from the checkpoint's local snapshot (predict_action reads
norm_stats.json from it), bfloat16 weights, the servers' cast of float inputs to the weights' dtype, and
`model.predict_action(...)` in continuous mode, depth reasoning off, 10 flow steps, normalize_language=True and the
servers' default enable_cuda_graph=False, under torch.inference_mode. The servers' two bf16 source patches are upstream
at every pinned revision (one identical modeling_molmoact2.py), so nothing is patched. The repo has no SO-101 or LIBERO
server; those entries make the same call with the model card's norm tag and cameras.

Observation (msgpack-numpy dict), keys as in the repo's servers:
    "<camera>_cam": HxWx3 uint8 RGB per camera (VARIANTS[...]["cameras"]; any size: the processor resizes each frame
                    to 378 x 378), "state": the robot's raw state (state_dim,), "instruction": str,
    optional "num_steps": flow steps (default 10, as the servers), optional "rs_seed": int, which seeds torch's
    generators right before the call (the same noise for the same seed; a measurement hook).
Reply: {"actions": float32 (horizon, action_dim), "policy_timing": {"preprocess_ms", "model_ms", "postprocess_ms"}}, the
actions in robot units: absolute joint targets on SO-101, YAM and DROID, LIBERO's delta end-effector pose and gripper on
LIBERO (norm_stats.json "control_mode"). model_ms is predict_action until the actions are on the CPU (the processor
included); preprocess_ms the request's checks and images; pgs_server's server_timing.decode_ms is the JPEG decode.

Metadata: the model facts, image_mode "jpeg" and image_size "native": the client sends each camera frame at its own size
as JPEG (q90), and the processor resizes it as it would the raw frame. mode="raw" sends the exact pixels.

Endpoint env: RS_MODEL (a VARIANTS key), RS_CHECKPOINT (hf://repo@revision), RS_CKPT_DIR (boot.sh: the snapshot's
directory), RS_FAST (auto: the fast path where it was measured, after its self-test; off: stock).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import pathlib
import re
import time
import traceback

import numpy as np
from PIL import Image

import pgs_server
from molmoact2_variants import DEFAULT_NUM_STEPS, VARIANTS

DTYPE = "bfloat16"  # the repo's run scripts (run_droid.sh, run_yam.sh: --dtype bfloat16)
HERE = pathlib.Path(__file__).resolve().parent
# Fast path (RS_FAST=auto): the same model, weights and call with the action expert's own CUDA graph
# (enable_cuda_graph=True, the model card's default) and torch.compile where the call spends the most kernel launches:
# one compiled block function shared by the expert's 36 blocks (and its final layer), run 10 times inside that graph,
# and the per-layer context projection. Inductor rounds its intermediates as eager does (emulate_precision_casts).
# Shapes stay dynamic, so a new prompt length compiles nothing (it captures the action graph again: one call of about
# 0.5 s). Compile: 25 s on a first start, 8 s from the cache on the volume (RTX 4090). Only on GPUs where it was
# measured served (MODELS.md), and only after the self-test below.
FAST_GPUS = ("NVIDIA GeForce RTX 4090",)
# Self-test bounds against the stored stock actions (selftest_<entry>.json, seed 7), as fractions of what a new noise
# draw changes there (its noise_actions, seed 8), since the action scale differs per robot: the actions come out of
# bfloat16, and one bfloat16 step is 0.0156 for a DROID joint near 2 rad and 1.0 for an SO-101 joint near 185 degrees.
# Measured on the RTX 4090 with the model cards' samples, fast against stock: 1 to 4 % of a new noise draw's mean
# difference, 0.6 to 7 % of its max (MODELS.md).
SELFTEST_MEAN_OF_NOISE, SELFTEST_MAX_OF_NOISE = 0.1, 0.5


def to_pil(arr) -> Image.Image:
    """As _to_pil in the repo's servers: HxWx3, clipped to uint8."""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


class Policy:
    """The stock call of the repo's servers. `predict(**kwargs)` is model.predict_action; `seed(n)` seeds torch."""

    def __init__(self, name: str, predict, *, checkpoint: str = "", seed=None, inference_mode=contextlib.nullcontext,
                 path: str = "stock", path_reason: str = "", cuda_graph: bool = False):
        self.name, self.variant = name, VARIANTS[name]
        self._predict, self._seed, self._inference_mode = predict, seed, inference_mode
        self.cuda_graph = cuda_graph  # the fast path: the action expert's own CUDA graph
        v = self.variant
        self.metadata = {"model": name, "family": "molmoact2", "checkpoint": checkpoint, "norm_tag": v["norm_tag"],
                         "action_shape": [v["horizon"], v["action_dim"]], "action_hz": v["hz"],
                         "cameras": [f"{c}_cam" for c in v["cameras"]], "state_dim": v["state_dim"],
                         "prompt_key": "instruction", "dtype": DTYPE, "num_steps": DEFAULT_NUM_STEPS,
                         "image_mode": "jpeg", "image_size": "native",
                         "playground_serve": {"path": path, "path_reason": path_reason or "stock MolmoAct2 code"}}

    def inputs(self, obs: dict) -> dict:
        """predict_action's arguments for an observation, as the repo's servers build them."""
        v = self.variant
        missing = [k for k in [f"{c}_cam" for c in v["cameras"]] + ["state", "instruction"] if k not in obs]
        if missing:
            raise KeyError(f"observation lacks {missing}; {self.name} takes {self.metadata['cameras']}, state, "
                           "instruction")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (v["state_dim"],):
            raise ValueError(f"state must be shape ({v['state_dim']},), got {state.shape}")
        return dict(images=[to_pil(obs[f"{cam}_cam"]) for cam in v["cameras"]], task=str(obs["instruction"]),
                    state=state, norm_tag=v["norm_tag"], inference_action_mode="continuous",
                    enable_depth_reasoning=False, num_steps=int(obs.get("num_steps", DEFAULT_NUM_STEPS)),
                    normalize_language=True, enable_cuda_graph=self.cuda_graph)

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        kwargs = self.inputs(obs)
        t1 = time.perf_counter()
        with self._inference_mode():
            if obs.get("rs_seed") is not None and self._seed is not None:
                self._seed(int(obs["rs_seed"]))
            raw = self._predict(**kwargs).actions
            if hasattr(raw, "detach"):  # a torch tensor: on the CPU, which waits for the GPU
                raw = raw.detach().float().cpu().numpy()
        t2 = time.perf_counter()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if not np.isfinite(actions).all():
            raise FloatingPointError("non-finite actions")
        timing = {"preprocess_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3,
                  "postprocess_ms": (time.perf_counter() - t2) * 1e3}
        return {"actions": actions, "policy_timing": timing}


def load_model(local_dir: str, dtype_name: str = DTYPE):
    """(model, processor) as the repo's servers load them."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    # tokenizer_config.json lists extra_special_tokens; transformers 4.57 wants a dict (the servers' workaround).
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, extra_special_tokens={})
    model = AutoModelForImageTextToText.from_pretrained(local_dir, trust_remote_code=True, torch_dtype=dtype)
    model = model.to("cuda:0").eval()
    target = next(model.parameters()).dtype

    def move_and_cast(inputs, dev, _target=target):  # the servers' fix: cast float inputs to the weights' dtype
        out = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(dev)
                if value.is_floating_point() and value.dtype != _target:
                    value = value.to(_target)
            out[key] = value
        return out

    model._move_inputs_to_device = move_and_cast
    return model, processor


def seed_all(seed: int) -> None:
    import torch

    torch.manual_seed(seed)  # seeds the CPU and every CUDA device's default generator


# ----- fast path
def reference(name: str) -> pathlib.Path:
    return HERE / f"selftest_{name}.json"


def selftest_observation(ref: dict) -> dict:
    """The stored observation as the server's policy sees a jpeg-mode request: JPEG bytes decoded to arrays."""
    from server_decode import decode_images

    jpeg = {cam: {b"__jpeg__": True, b"data": base64.b64decode(data)} for cam, data in ref["images_jpeg"].items()}
    return {**decode_images(jpeg), "state": np.asarray(ref["state"], np.float32), "instruction": ref["instruction"],
            "rs_seed": int(ref["seed"])}


def check(actions, ref: dict) -> tuple[bool, dict]:
    """(passed, detail) for `actions` against the stored stock actions: finite, same shape, within the bounds."""
    a, want = np.asarray(actions, np.float64), np.asarray(ref["actions"], np.float64)
    if a.shape != want.shape:
        return False, {"error": f"actions shape {list(a.shape)}, stock {list(want.shape)}"}
    if not np.isfinite(a).all():
        return False, {"error": "non-finite actions"}
    d, noise = np.abs(a - want), np.abs(np.asarray(ref["noise_actions"], np.float64) - want)
    detail = {"mean_abs_diff": round(float(d.mean()), 6), "max_abs_diff": round(float(d.max()), 5),
              "mean_bound": round(float(noise.mean() * SELFTEST_MEAN_OF_NOISE), 6),
              "max_bound": round(float(noise.max() * SELFTEST_MAX_OF_NOISE), 5)}
    return bool(d.mean() <= detail["mean_bound"] and d.max() <= detail["max_bound"]), detail


def decide(mode: str, name: str, gpu: str, have_reference: bool) -> tuple[str, str]:
    """("fast" or "stock", why). mode: the endpoint's RS_FAST, "auto" (default), "off", or "measure" (any GPU: for
    measuring the fast path on a GPU not in FAST_GPUS yet; the self-test still decides)."""
    if mode not in ("auto", "measure"):
        return "stock", f"fast path turned off (RS_FAST={mode}, `pgs up --stock-path`)"
    if gpu not in FAST_GPUS and mode != "measure":
        return "stock", f"fast path not measured on {gpu or 'this GPU'}, only on {', '.join(FAST_GPUS)}"
    if not have_reference:
        return "stock", f"no self-test reference for {name}"
    return "fast", "action CUDA graph and compiled action expert, if its boot self-test passes"


class GpuImages:
    """The checkpoint's image processor in its "resize" crop mode, with the resize and normalization on the GPU: the
    same torchvision bilinear resize (antialias off) of the uint8 frame to 378 x 378, the same float32 arithmetic
    (x / 255 * 2 - 1) and the same patch layout, as one batch for the frames of one size. The outputs that do not
    depend on the frames (pooling indices, grids, crop counts) come from the official processor, once per number of
    images. A frame can differ from the CPU's resize by one level where the GPU's float rounding falls on the other side
    of .5; check() counts that for the self-test."""

    def __init__(self, official):
        self.official = official
        self.size = (int(official.size["height"]), int(official.size["width"]))
        self.patch = int(official.patch_size)
        self._fixed: dict = {}

    def pixels(self, images):
        import torch
        import torchvision.transforms.functional as TF

        arrays = [np.asarray(im.convert("RGB") if isinstance(im, Image.Image) else im) for im in images]
        out = [None] * len(arrays)
        for shape in {a.shape for a in arrays}:
            idx = [i for i, a in enumerate(arrays) if a.shape == shape]
            batch = torch.from_numpy(np.stack([arrays[i] for i in idx])).to("cuda", non_blocking=True)
            resized = TF.resize(batch.permute(0, 3, 1, 2), list(self.size),
                                interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
            x = (resized.to(torch.float32) - 0.0) / 255.0 * 2.0 - 1.0  # resize_image, then normalize_image
            k, p = len(idx), self.patch
            x = x.permute(0, 2, 3, 1).reshape(k, self.size[0] // p, p, self.size[1] // p, p, 3)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(k, (self.size[0] // p) * (self.size[1] // p), p * p * 3)
            for j, i in enumerate(idx):
                out[i] = x[j]
        return torch.stack(out)

    def __call__(self, images, **kwargs):
        if self.official.crop_mode != "resize":
            return self.official(images, **kwargs)
        n = len(images)
        if n not in self._fixed:
            blank = [Image.new("RGB", (self.size[1], self.size[0]))] * n
            self._fixed[n] = self.official(blank, **kwargs)
        out = type(self._fixed[n])({**self._fixed[n]})
        out["pixel_values"] = self.pixels(images)
        return out

    def check(self, images) -> dict:
        """The GPU's pixel values against the official processor's, in units of one uint8 level."""
        official = np.asarray(self.official(images, return_tensors="np")["pixel_values"], np.float32)
        gpu = self.pixels(images).float().cpu().numpy()
        levels = np.abs(gpu - official) * 127.5
        return {"pixels": int(levels.size), "differ": int((levels > 0.01).sum()), "max_levels": round(float(levels.max()), 3)}


class ShapeGraphs:
    """`fn` captured as one CUDA graph per signature of its tensor arguments (shapes, dtypes; other arguments must
    repeat): a call copies its tensors into that graph's inputs and replays it. The result is the graph's own output,
    overwritten by the next call with the same signature; the policy runs one call at a time and uses it before then.
    At most `keep` graphs (the oldest is dropped)."""

    def __init__(self, fn, keep: int):
        import collections

        self.fn, self.keep, self.graphs = fn, keep, collections.OrderedDict()

    def __call__(self, *args, **kwargs):
        import torch

        named = list(enumerate(args)) + sorted(kwargs.items())
        tensors = [(k, v) for k, v in named if torch.is_tensor(v)]
        key = (tuple((k, tuple(v.shape), v.dtype) for k, v in tensors),
               tuple((k, repr(v)) for k, v in named if not torch.is_tensor(v)))
        entry = self.graphs.get(key)
        if entry is None:
            static = {k: v.clone() for k, v in tensors}

            def run():
                return self.fn(*[static.get(i, a) for i, a in enumerate(args)],
                               **{k: static.get(k, v) for k, v in kwargs.items()})

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run()  # warm-up outside the graph (lazy buffers, cuBLAS workspaces)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            entry = self.graphs[key] = (graph, static, out)
            while len(self.graphs) > self.keep:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        graph, static, out = entry
        for k, v in tensors:
            static[k].copy_(v)
        graph.replay()
        return out


PIXEL_DIFFER_MAX = 0.005  # share of pixel values the GPU resize may put one uint8 level away from the CPU's


def build_fast(model, processor, policy: Policy, ref: dict) -> dict:
    """Turn `policy` (the stock call on `model`) into the fast path and run the self-test; raises if it fails."""
    import functools

    import torch

    torch._inductor.config.emulate_precision_casts = True
    expert = model.model._require_action_expert()
    for modules in (list(expert.blocks), [expert.final_layer]):  # one compiled function per module class
        compiled = torch.compile(type(modules[0]).forward, dynamic=True)
        for module in modules:
            module.forward = functools.partial(compiled, module)
    expert._prepare_kv_context = torch.compile(expert._prepare_kv_context, dynamic=True)
    # Graphs of the vision encoder (one input shape per entry) and of the language model's prefill (one per prompt
    # length; a new length captures once, about 0.5 s).
    vision = model.model.vision_backbone
    vision.encode_image = ShapeGraphs(vision.encode_image, keep=2)
    model.model.transformer.forward = ShapeGraphs(model.model.transformer.forward, keep=4)
    images = GpuImages(processor.image_processor)
    processor.image_processor = images
    policy.cuda_graph = True
    obs = selftest_observation(ref)
    pixels = images.check(policy.inputs(obs)["images"])
    pgs_server.log("fast_pixels", **pixels)
    if pixels["max_levels"] > 1.01 or pixels["differ"] > PIXEL_DIFFER_MAX * pixels["pixels"]:
        raise RuntimeError(f"GPU image preprocessing differs from the official processor: {pixels}")
    t = time.time()
    example = make_example()
    for task in ("warmup", "compile once more with another prompt length so later lengths need nothing"):
        policy.infer({**example, "instruction": task})
    pgs_server.log("fast_built", compile_s=round(time.time() - t, 1))
    for attempt in (1, 2):  # each attempt after another request went through the same graph and buffers
        ok, detail = check(policy.infer(obs)["actions"], ref)
        pgs_server.log("fast_selftest", attempt=attempt, ok=ok, **detail)
        if not ok:
            raise RuntimeError(f"self-test against the stored stock actions failed: {detail}")
        policy.infer(example)
    return {**detail, "pixels_differ": pixels["differ"]}


def undo_fast(model, processor, policy: Policy) -> None:
    import torch

    expert = model.model._require_action_expert()
    for module in [*expert.blocks, expert.final_layer]:  # the compiled wrappers are instance attributes
        module.__dict__.pop("forward", None)
    expert.__dict__.pop("_prepare_kv_context", None)
    model.model.vision_backbone.__dict__.pop("encode_image", None)
    model.model.transformer.__dict__.pop("forward", None)
    if isinstance(processor.image_processor, GpuImages):
        processor.image_processor = processor.image_processor.official
    policy.cuda_graph = False
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def load() -> Policy:
    import functools

    gpu = pgs_server.gpu_info("name").splitlines()[0].strip() if pgs_server.gpu_info("name") else ""
    # Inductor's and Triton's caches per GPU type on the volume: a later start compiles from them in seconds.
    slug = re.sub(r"\W+", "_", gpu) or "gpu"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/runpod-volume/pgs/molmoact2/inductor/{slug}")
    import torch

    name = os.environ["RS_MODEL"]
    t = time.time()
    model, processor = load_model(os.environ["RS_CKPT_DIR"])
    pgs_server.log("model_loaded", s=round(time.time() - t, 1), dtype=str(next(model.parameters()).dtype),
                   gpu_memory=pgs_server.gpu_info("memory.used,memory.total"))
    predict = functools.partial(model.predict_action, processor=processor)
    path, reason = decide(os.environ.get("RS_FAST", "auto"), name, gpu, reference(name).is_file())
    pgs_server.log("model_path", path=path, reason=reason)
    policy = Policy(name, predict, checkpoint=os.environ.get("RS_CHECKPOINT", ""), seed=seed_all,
                    inference_mode=torch.inference_mode, path_reason=os.environ.get("RS_PATH_REASON") or reason)
    if path == "fast":
        try:
            detail = build_fast(model, processor, policy, json.loads(reference(name).read_text()))
            policy.metadata["playground_serve"].update(path="fast", path_reason=(
                f"GPU preprocessing, vision and prefill graphs, action CUDA graph and compiled action expert on {gpu}, "
                f"same bf16 weights and 10 flow steps; boot self-test passed (max {detail['max_abs_diff']}, mean "
                f"{detail['mean_abs_diff']} against the stock path's stored actions)"))
            return policy
        except Exception as e:  # noqa: BLE001  (never serve an unchecked fast path: stock in this process)
            reason = f"fast path failed: {e}"[:500]
            pgs_server.log("fast_failed", error=traceback.format_exc()[-1500:])
            pathlib.Path(pgs_server.PATH_REASON).write_text(reason)
            undo_fast(model, processor, policy)
            policy.metadata["playground_serve"]["path_reason"] = reason
            try:
                policy.infer(make_example())
            except Exception:  # noqa: BLE001  (the process may be broken: boot.sh starts it again with RS_FAST=off)
                pgs_server.log("fallback_failed", error=traceback.format_exc()[-1500:])
                raise SystemExit(3)
    return policy


def make_example() -> dict:
    """Warm-up input: zero frames and state, as the repo's servers warm up (180 x 320 frames, "warmup")."""
    v = VARIANTS[os.environ["RS_MODEL"]]
    obs = {f"{cam}_cam": np.zeros((180, 320, 3), np.uint8) for cam in v["cameras"]}
    return {**obs, "state": np.zeros(v["state_dim"], np.float32), "instruction": "warmup"}


if __name__ == "__main__":
    pgs_server.run_worker(load, make_example)
