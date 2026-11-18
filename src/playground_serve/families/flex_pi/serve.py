"""Flex-pi (UW + Ai2) worker: the LIBERO checkpoint behind Playground Serve's websocket protocol.

There is no LIBERO server upstream, so this follows the evaluator at COMMIT, experiments/libero/eval_libero_single.py
(EVAL below): start-up as eval_single_process, a call as _predict_action_chunk on the T-shape float path, with
sim_libero.yaml's values and scripts/eval_flexpi_libero_single.sh's regime flags. EVAL imports the simulator at module
level, so its tensor helpers are copied here, cited; the rest is the installed package. verify.py runs EVAL's own
functions on the same model and inputs (simulator stubbed) and compares.

Observation (msgpack-numpy dict), keys as openpi's LIBERO client (Playground Serve's pi05_libero uses the same):
    "observation/image" (agentview), "observation/wrist_image" (eye-in-hand): HxWx3 uint8 RGB, rotated 180 degrees from
        LIBERO's render as get_libero_image does (upright, as openpi's LIBERO example sends them); the evaluator renders
        256 x 256 (the metadata's image_size);
    "observation/state": (8,) eef position (3), eef axis-angle (3), gripper qpos (2);
    "prompt": the task's language;
    optional "observation/image_depth", "observation/wrist_image_depth": (H, W) uint16 millimetres, rotated like the RGB
        (get_libero_per_cam_depth_uint16_mm). With both, the pointmap stream is present, as in the published eval;
        without them the call passes present_pointmap=False.
Reply: {"actions": float32 (32, 7)} for env.step: delta eef pose (6) and gripper (-1 open, +1 close). The evaluator runs
the first 10 and replans (sim_libero.yaml replan_steps). Every call uses seed 42 (train.yaml seed, which EVAL passes to
infer_action on each call), so the same request gives the same actions.

Model paths (metadata "path", "path_reason"): "stock" is the evaluator's setting (eager, torch_compile false). "fast" is
the same model with the repo's own inference options that keep the math (prepare_for_inference: torch.compile with CUDA
graphs, the default of its RoboTwin launchers); it serves only after its boot self-test reproduced the stock actions
stored for this GPU (selftest_<model>.json), else stock serves.

Regime (RS_FLEXPI_JOINT): "false" (the default here) is the action-only regime, which the checkpoint's card publishes
for LIBERO (98.4 % success against 98.5 % for full joint) and gives as "the same command plus three flags"
(INFER_JOINT_VIDEO/DINO/POINTMAP=false); "true" is the evaluator's default full joint regime, which also generates
the video, DINO and pointmap futures (about 3.4 to 6 times the compute). Same weights, steps and inputs either way.

Endpoint env: RS_MODEL, HF_HOME, DIFFSYNTH_MODEL_BASE_PATH (VAE, T5 and tokenizer from ModelScope; both on the volume),
RS_FLEXPI_JOINT ("false" or "true", above), RS_FAST "auto" or "off".
"""

from __future__ import annotations

import contextlib
import json
import os
import time

import numpy as np

COMMIT = "b27019ec454aaf5e09f9655368f02a521f6dffa5"  # github.com/geyan21/flex-pi, 2026-10-28
IMAGE, WRIST, STATE = "observation/image", "observation/wrist_image", "observation/state"
DEPTHS = ("observation/image_depth", "observation/wrist_image_depth")
# datasets/lerobot/robot_video_dataset.py DEFAULT_PROMPT (that module imports the video decoders, so it is copied)
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
# sim_libero.yaml EVALUATION values and train.yaml's seed, as EVAL's _predict_action_chunk passes them.
INFER_DEFAULTS = dict(negative_prompt="", text_cfg_scale=1.0, num_inference_steps=4, sigma_shift=None, seed=42,
                      rand_device="cpu", tiled=False, dynamic_step_skip=False,
                      step_skip_thresholds=[(0.95, 4), (0.93, 2)], step_skip_sim_aggregation="mean")
# Model fields EVAL keeps from sim_libero.yaml over the saved config (_maybe_swap_eval_cfg_with_saved).
EVAL_MODEL_FIELDS = {"load_text_encoder": True, "skip_dit_load_from_pretrain": True, "action_dit_pretrained_path": None}
LICENSE = "MIT (code and weights); DINOv3 License for the vision encoder it loads"
# Hugging Face revisions read 2026-11-13. K is one per camera for every suite (paper App. E); the 4 suites' files match.
VARIANTS = {
    "flex_pi_libero": dict(repo="flex-pi/flexpi-libero", revision="f853cb49331aa0ab8124cbd1e1fb3a56e07a2523",
                           weights="checkpoints/weights/step_010860.pt", layout="tshape_libero_2cam_448x512",
                           horizon=32, action_dim=7, state_dim=8, hz=20,
                           intrinsics=("flex-pi/libero_mujoco3.3.2_depth", "0dedf7fcce85b4de2b82687c4f99c148b97b3a90",
                                       "libero_spatial_no_noops_lerobot/meta/camera_intrinsics.json")),
}
# The repo's prepare_for_inference options of each path. "stock" = sim_libero.yaml (torch_compile false). "fast" =
# options its docstrings call the same math (compile with CUDA graphs per denoise step, as the RoboTwin launchers run).
PATH_OPTIONS = {"stock": dict(torch_compile=False, torch_compile_mode="reduce-overhead", quantization=None),
                "fast": dict(torch_compile=True, torch_compile_mode="reduce-overhead", quantization=None)}


def per_cam_float_tensor(img, target_hw, device, dtype, intermediate_hw=(256, 320)):
    """_build_per_cam_float_tensor (EVAL): uint8 -> float [0, 1] -> 256x320 -> target (bilinear, antialias)
    -> [-1, 1], as [1, 3, H, W]."""
    import torch
    import torchvision.transforms.functional as transforms_F

    def resize(t, hw):
        return transforms_F.resize(t, size=[int(hw[0]), int(hw[1])],
                                   interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)

    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
    assert t.dtype == torch.uint8, f"expected uint8 input, got {t.dtype}"
    t = t.to(torch.float32) / 255.0
    if (t.shape[-2], t.shape[-1]) != tuple(intermediate_hw):
        t = resize(t, intermediate_hw)
    if tuple(intermediate_hw) != tuple(target_hw):
        t = resize(t, target_hw)
    return (t * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)


def depth_to_grid(arr, target_hw):
    """_depth_to_grid (EVAL): int32 mm, linspace-endpoint nearest regrid."""
    import torch

    t = torch.from_numpy(arr.astype(np.int32))
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if tuple(t.shape) != (h_t, w_t):
        ys = torch.linspace(0, t.shape[0] - 1, h_t).round().long()
        xs = torch.linspace(0, t.shape[1] - 1, w_t).round().long()
        t = t[ys[:, None], xs[None, :]]
    return t


def build_K(entry: dict, target_hw):
    """_build_K (EVAL): K rescaled onto target_hw with the (target - 1) / (raw - 1) convention."""
    import torch

    h_t, w_t = target_hw
    raw_w, raw_h = float(entry["width"]), float(entry["height"])
    sx = (float(w_t) - 1.0) / (raw_w - 1.0) if raw_w > 1 else 1.0
    sy = (float(h_t) - 1.0) / (raw_h - 1.0) if raw_h > 1 else 1.0
    return torch.tensor([[entry["fx"] * sx, 0.0, entry["cx"] * sx], [0.0, entry["fy"] * sy, entry["cy"] * sy],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)


class TorchOps:
    """The tensor steps of one evaluator call, bound to the loaded model, processor and composite layout."""

    def __init__(self, model, processor, layout, slot_key_map, depth_hw, video_hw):
        self.model, self.processor, self.layout, self.skm = model, processor, layout, slot_key_map
        self.depth_hw, self.video_hw = depth_hw, tuple(video_hw)

    def inputs(self, image, wrist):
        """EVAL T-shape stretch path: per-cam float tensors at each slot's size, synthetic right wrist at -1,
        composite."""
        import torch
        from flexpi.per_cam_compose import compose_from_per_cam

        dev, dtype, phys = self.model.device, self.model.torch_dtype, {"cam_high": image, "cam_left_wrist": wrist}
        per_cam = {cam: per_cam_float_tensor(phys[cam], hw, dev, dtype) if cam in phys
                   else torch.full((1, 3, *hw), fill_value=-1.0, device=dev, dtype=dtype)
                   for cam, hw in self.layout.resolved_per_cam_hw(self.skm).items()}
        x = compose_from_per_cam({k: v.unsqueeze(2) for k, v in per_cam.items()}, self.layout,
                                 slot_key_map=self.skm).squeeze(2)
        if tuple(x.shape[-2:]) != self.video_hw:
            raise ValueError(f"T-shape composite size mismatch: got {tuple(x.shape[-2:])}, expected {self.video_hw}")
        return x, per_cam

    def proprio(self, state):
        """_normalize_proprio (EVAL)."""
        import torch

        p = self.processor
        batch = {"state": {p.shape_meta["state"][0]["key"]: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        return p.action_state_merger.forward(p.normalizer.forward(p.action_state_transform(batch)))["state"]

    def depth(self, image_depth, wrist_depth):
        """_extract_per_cam_depth_for_model (EVAL), from depth already in uint16 mm."""
        import torch

        def grid(arr, hw):
            return depth_to_grid(arr, hw).to(self.model.device).reshape(1, 1, *hw)

        left = grid(wrist_depth, self.depth_hw["cam_left_wrist"])
        return {"cam_high": grid(image_depth, self.depth_hw["cam_high"]), "cam_left_wrist": left,
                "cam_right_wrist": torch.zeros_like(left)}

    def denormalize(self, action) -> np.ndarray:
        """_denormalize_action (EVAL): merger, normalizer, then transforms backward; [B, T, raw_dim]."""
        import torch

        p, merger = self.processor, self.processor.action_state_merger
        action = (action.unsqueeze(0) if action.ndim == 2 else action).to(dtype=torch.float32, device="cpu")
        batch = {"action": action}
        state_dim = getattr(merger, "state_target_dim", None) or getattr(merger, "total_dim", None)
        if state_dim is not None:
            batch["state"] = torch.zeros(action.shape[0], action.shape[1], int(state_dim), dtype=action.dtype)
        batch = p.normalizer.backward(merger.backward(batch))
        for trans in reversed(p.action_state_transforms or []):
            batch = trans.backward(batch)
        return batch["action"][p.shape_meta["action"][0]["key"]].numpy()


def finish_actions(action: np.ndarray) -> np.ndarray:
    """EVAL's last steps: gripper back to -1 open / +1 close, invert_gripper_action, binarize_gripper (true)."""
    action = np.array(action, dtype=np.float32)
    action[..., -1] = action[..., -1] * 2 - 1
    action[..., -1] = action[..., -1] * -1.0
    action[..., -1] = np.sign(action[..., -1])
    return action


class Policy:
    def __init__(self, name: str, model, ops, *, joint: bool = False, num_video_frames: int = 9,
                 no_grad=contextlib.nullcontext, path: str = "stock", path_reason: str = "",
                 text_cache: bool = False):
        self.name, self.variant, self.model, self.ops = name, VARIANTS[name], model, ops
        self.joint, self.num_video_frames, self._no_grad = joint, num_video_frames, no_grad
        # Fast path only: the text context of each prompt, encoded once by the model's own encode_prompt and passed
        # as infer_action's context/context_mask (its documented alternative to `prompt`), as the repo's latency
        # benchmark does with "precomputed text context". Same tensors the call would compute.
        self.text_cache: dict | None = {} if text_cache else None
        v = self.variant
        self.metadata = {"model": name, "family": "flex_pi", "checkpoint": f"{v['repo']}@{v['revision']}",
                         "action_shape": [v["horizon"], v["action_dim"]], "action_hz": v["hz"],
                         "cameras": [IMAGE, WRIST], "keys": [IMAGE, WRIST, STATE, "prompt", *DEPTHS],
                         "state_dim": v["state_dim"], "prompt_key": "prompt", "image_size": [256, 256],
                         "history": 0, "replan_steps": 10, "regime": "full_joint" if joint else "action_only",
                         "license": LICENSE, "path": path, "path_reason": path_reason}

    def kwargs(self, obs: dict) -> dict:
        """infer_action's keywords for one request, as EVAL's _predict_action_chunk builds them."""
        v = self.variant
        images = [np.asarray(obs[key]) for key in (IMAGE, WRIST)]
        if any(im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8 for im in images):
            raise ValueError(f"images must be HxWx3 uint8, got {[(im.dtype, im.shape) for im in images]}")
        state = np.asarray(obs[STATE], dtype=np.float32).reshape(-1)
        if state.shape != (v["state_dim"],):
            raise ValueError(f"{STATE} must be shape ({v['state_dim']},), got {state.shape}")
        if sum(key in obs for key in DEPTHS) == 1:
            raise ValueError(f"send both {DEPTHS} or neither")
        x, per_cam = self.ops.inputs(*images)
        kwargs = dict(INFER_DEFAULTS, prompt=DEFAULT_PROMPT.format(task=str(obs["prompt"])), input_image=x,
                      action_horizon=v["horizon"], proprio=self.ops.proprio(state), per_cam=per_cam,
                      num_video_frames=self.num_video_frames, present_video=True, present_dino=True,
                      present_pointmap=DEPTHS[0] in obs, joint_video=self.joint, joint_dino=self.joint,
                      joint_pointmap=self.joint)
        if DEPTHS[0] in obs:
            kwargs["per_cam_depth"] = self.ops.depth(*(np.asarray(obs[key], dtype=np.uint16) for key in DEPTHS))
        return kwargs

    def infer(self, obs: dict) -> dict:
        kwargs = self.kwargs(obs)
        with self._no_grad():
            if self.text_cache is not None:
                prompt = kwargs.pop("prompt")
                if prompt not in self.text_cache:
                    if len(self.text_cache) >= 32:
                        self.text_cache.pop(next(iter(self.text_cache)))
                    self.text_cache[prompt] = self.model.encode_prompt(prompt)
                context, mask = self.text_cache[prompt]
                kwargs.update(context=context.clone(), context_mask=mask.clone())
            pred = self.model.infer_action(**kwargs)
        return {"actions": finish_actions(self.ops.denormalize(pred["action"])[0])}


def load_parts(name: str):
    """Model, processor, ops and EVAL's settings as eval_single_process builds them (no prepare_for_inference)."""
    import torch
    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from flexpi.composite_layouts import canonical_layout_name, get_layout
    from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from flexpi.models.helpers.action_pad import resolve_action_dim_mask
    from flexpi.utils.libero_setup import patch_torch_load
    from flexpi.utils.pytorch_utils import set_global_seed

    patch_torch_load()  # EVAL runs it at import (libero_utils -> prepare_libero): torch.load(weights_only=False)
    v = VARIANTS[name]
    cfg = OmegaConf.load(hf_hub_download(v["repo"], "config.yaml", revision=v["revision"]))
    OmegaConf.set_struct(cfg, False)
    for key, value in EVAL_MODEL_FIELDS.items():
        cfg.model[key] = value
    if canonical_layout_name(str(cfg.model.composite_layout).strip().lower()) != v["layout"]:
        raise ValueError(f"composite_layout {cfg.model.composite_layout!r}: this adapter has the T-shape path only")
    set_global_seed(INFER_DEFAULTS["seed"], get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")  # train.yaml mixed_precision bf16
    model.load_checkpoint(hf_hub_download(v["repo"], v["weights"], revision=v["revision"]))
    model = model.to("cuda").eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(
        hf_hub_download(v["repo"], "dataset_stats.json", revision=v["revision"])))
    layout = get_layout(canonical_layout_name(str(cfg.model.composite_layout).strip().lower()))
    skm = cfg.model.get("composite_layout_slot_key_map")
    skm = None if skm is None else {str(k): str(x) for k, x in OmegaConf.to_container(skm, resolve=True).items()}
    depth_hw = {"cam_high": (256, 320), "cam_left_wrist": (224, 224)}  # EVAL _per_cam_depth_target_hw defaults
    depth_hw.update({m["key"]: (int(m["shape"][1]), int(m["shape"][2])) for m in cfg.data.train.shape_meta.depth})
    with open(hf_hub_download(v["intrinsics"][0], v["intrinsics"][2], revision=v["intrinsics"][1],
                              repo_type="dataset")) as f:
        raw = json.load(f)
    k_by_cam = {cam: build_K(raw[cam], depth_hw[cam]) for cam in ("cam_high", "cam_left_wrist")}
    k_by_cam["cam_right_wrist"] = k_by_cam["cam_left_wrist"].clone()  # EVAL: K stacked in slot order
    kmap = layout.resolve_slot_key_map(skm)
    frames, ratio = int(cfg.data.train.num_frames), int(cfg.data.train.action_video_freq_ratio)
    if v["horizon"] != frames - 1:  # EVALUATION.action_horizon: null -> num_frames - 1
        raise ValueError(f"checkpoint config gives horizon {frames - 1}, VARIANTS says {v['horizon']}")
    ops = TorchOps(model, processor, layout, skm, depth_hw, cfg.data.train.video_size)

    def finish(options: dict):
        """prepare_for_inference, then the action mask and K, in EVAL's order."""
        model.prepare_for_inference(**options)
        pad, _ = resolve_action_dim_mask(processor, model, cfg, None)  # EVALUATION.action_dim_mask: null
        model.set_action_dim_mask(pad)
        model.set_camera_intrinsics(torch.stack([k_by_cam[kmap[s.key]] for s in layout.cam_slots()]).to("cuda"))

    return model, processor, ops, cfg, finish, (frames - 1) // ratio + 1


DEFAULT_THREADS = 16


def set_threads(torch, log) -> None:
    """torch sizes its CPU thread pool by the host's cores (120 on one RTX 5090 host, 112 threads on a 224-core H100
    host) unless OMP_NUM_THREADS is set, which boot.sh sets to the container's CPUs (RUNPOD_CPU_COUNT); with 120 threads
    the CPU input step took 124 to 590 ms instead of 3 to 6. Set once, before anything compiles (dynamo recompiles when
    it changes): RS_TORCH_THREADS, default 16, at most torch's own default; "0" keeps torch's default."""
    want = int(os.environ.get("RS_TORCH_THREADS") or DEFAULT_THREADS)
    if want > 0:
        torch.set_num_threads(min(want, torch.get_num_threads()))
    log("cpu", host_cores=os.cpu_count(), container_cpus=os.environ.get("RUNPOD_CPU_COUNT"),
        torch_threads=torch.get_num_threads())


def load() -> Policy:
    import torch

    import pgs_server

    set_threads(torch, pgs_server.log)
    name = os.environ["RS_MODEL"]
    joint = os.environ.get("RS_FLEXPI_JOINT", "false").lower() == "true"
    import prefetch  # /rs/prefetch.py: ModelScope files in parallel ranges, checked, before the loader looks

    prefetch.run(log=pgs_server.log)
    t = time.time()
    model, processor, ops, cfg, finish, num_video_frames = load_parts(name)
    pgs_server.log("model_loaded", s=round(time.time() - t, 1))

    def make(path: str, reason: str) -> Policy:
        return Policy(name, model, ops, joint=joint, num_video_frames=num_video_frames, no_grad=torch.no_grad,
                      path=path, path_reason=reason, text_cache=path == "fast")

    if os.environ.get("RS_VERIFY") == "1":  # bring-up check against EVAL's own functions (verify.py)
        import verify

        verify.ready_now(pgs_server)

        import fast

        finish(PATH_OPTIONS["stock"])
        stock, best = verify.run(name, make("stock", "verify"), model, processor, cfg, num_video_frames,
                                 log=pgs_server.log, finish=finish, path_options=PATH_OPTIONS)
        ref = json.loads(fast.reference_path(name).read_text())
        warm = [fast.decode_observation(ref["observation"], depth=d) for d in (True, False)]
        switch = verify.Switch(make("stock", "verify"), make("fast", "verify"), finish,
                               {"stock": PATH_OPTIONS["stock"], "fast": best or fast.FAST_OPTIONS}, warm)
        verify.selftests(switch, ref, stock, (pgs_server.gpu_info("name").splitlines() or [""])[0].strip(),
                         pgs_server.log)
        return switch
    import fast  # /rs/fast.py, next to this file

    gpu = (pgs_server.gpu_info("name").splitlines() or [""])[0].strip()
    return fast.choose(name, make, finish, PATH_OPTIONS, gpu=gpu, mode=os.environ.get("RS_FAST", "auto"),
                       log=pgs_server.log, joint=joint)


def make_example() -> dict:
    """Warm-up input: zero frames and state, no depth."""
    v = VARIANTS[os.environ["RS_MODEL"]]
    return {IMAGE: np.zeros((256, 256, 3), np.uint8), WRIST: np.zeros((256, 256, 3), np.uint8),
            STATE: np.zeros(v["state_dim"], np.float32), "prompt": "warmup"}


def main() -> None:
    import pgs_server

    # load() downloads through huggingface_hub and ModelScope into the volume itself: the spec's RS_SELF_FETCH=1 tells
    # boot.sh to skip stage.py fetch, so run_worker waits for no checkpoint copy.
    pgs_server.run_worker(load, make_example)


if __name__ == "__main__":
    main()
