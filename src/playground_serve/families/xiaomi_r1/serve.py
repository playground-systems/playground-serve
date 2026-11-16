"""Xiaomi-Robotics-1 (XR-1) worker: the RoboCasa365 checkpoint behind Playground Serve's websocket protocol.

Official inference at COMMIT is split in two: deploy/server.py loads the model (line 20) and runs `model(**data)`
(lines 57-63); the RoboCasa365 evaluator builds `data` with the processor (eval_robocasa365/entry.py:161-185) and
deploy/client.py:61 decodes the reply. Here the stock path is EvalClient.infer with the socket replaced by the server's
lines. entry.py is a script (it imports the socket client and imageio at module level), so its helpers are copied
and cited.

Two paths; the metadata's "path" and "path_reason" and the log line {"rs": "path"} say which one serves and why.
stock  as above, unchanged.
fast   xr1_fast.py: the same request, built with its per-instruction parts cached and its pixels on the GPU, and the
       whole forward (vision tower, prefill, the 5 flow steps of the DiT head) run as one CUDA graph per prompt, from
       the same noise draw. Only on GPUs in FAST, and only after a boot self-test against the stock path
       (selftest_<model>.json); else stock serves. See xr1_fast.py and NOTES.md.

Observation (msgpack-numpy dict), keys as entry.py's CAMERA_KEYS (lines 33-37):
    "video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand": each the frames of
        ticks t-6, t-4, t-2, t, oldest first (history 4, interval 2; `sample_history` builds them from the robot's last
        7 frames and repeats the first frame at episode start), 256 x 256 x 3 uint8 RGB: a (4, 256, 256, 3) array (raw
        mode), or a list or a map {"0": .., "1": .., "2": .., "3": ..} of four 256 x 256 x 3 frames (a map's frames go
        as JPEG in the client's jpeg mode);
    "state": (4, 14) float32 at the same ticks: eef position rel (3), eef rotation rel as axis-angle (3), gripper
        qpos (2), base position (3), base rotation as axis-angle (3) (entry.py:58-71, observation_to_state);
    "instruction": str (RoboCasa's annotation.human.task_description);
    optional "rs_seed": int, the flow noise from torch.manual_seed(rs_seed) for this call only (checks).
Reply: {"actions": float32 (16, 12), "policy_timing": {"preprocess_ms", "model_ms", "postprocess_ms"}}, the actions in
robocasa's convert_action layout (robocasa/utils/env_utils.py:142-154): eef
position (3), eef rotation (3), gripper close (1), base motion (4), control mode (1).

Flow noise: the model draws it with torch.randn_like from torch's default stream, seeded once when the model code is
imported (modeling_mibot.py:29-41: 7 + the CUDA_VISIBLE_DEVICES index), as the official server does; a call with
"rs_seed" uses its own seed and leaves that stream where it was.

Weights: hf://<repo>@<revision>, every file of the repo at that revision, into RS_CKPT_DIR (boot.sh: the volume's
complete copy or a container-disk directory); until stage.py fetch takes hf:// URIs, prepare() lists the repo through
the Hugging Face API and runs stage.fetch with that listing; stage.py fill saves it to the volume after ready.
Endpoint env: RS_MODEL, RS_CHECKPOINT, RS_CKPT_DIR (boot.sh), RS_FAST ("auto" default, "off"), RS_FAST_COMPILE ("1"
default: the DiT step compiled with inductor inside the graph; "0": the graph alone), RS_COMPILE_CACHE (inductor's
cache, per GPU type, on the volume), HF_HOME (the module cache of trust_remote_code), TOKENIZERS_PARALLELISM=false
(as scripts/deploy.sh).
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

import pgs_server

COMMIT = "0dd7aef8dc87296246aae812a1f59ccb708e5546"  # github.com/XiaomiRobotics/Xiaomi-Robotics-1, 2026-09-09
STATE_DIM = 60  # the processor's padded state width (entry.py:31; config.json state_dim)
CAMERA_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
LICENSE = "Apache-2.0 (code and weights)"
# Hugging Face revision read 2026-11-13 and 2026-11-16. Eval settings: entry.py defaults (lines 30-32, 324-330).
VARIANTS = {
    "xiaomi_r1_robocasa365": dict(
        repo="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365", revision="3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4",
        robot_type="robocasa365", cameras=CAMERA_KEYS, image_size=(256, 256), state_dim=14, action_dim=12,
        horizon=16, hz=20, history=4, interval=2, crop_ratio=0.95),
}
# Model -> GPU names (nvidia-smi) where the fast path was measured served against stock (NOTES.md).
FAST = {"xiaomi_r1_robocasa365": ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA H100 80GB HBM3",
                                   "NVIDIA L40S")}
FAST_FAILED = "/tmp/rs_fast_failed"
HERE = pathlib.Path(__file__).resolve().parent
HF = "https://huggingface.co"
log = pgs_server.log


def sample_history(history, length: int = 4, interval: int = 2) -> np.ndarray:
    """entry.py:81-89, unchanged: `length` items `interval` apart ending at the newest, oldest first; at an episode's
    start the oldest item stands in for the ticks that do not exist yet. The robot side keeps the last
    (length - 1) * interval + 1 frames and states (entry.py:227-234) and sends this for each."""
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [max(0, len(items) - 1 - (length - 1 - index) * interval) for index in range(length)]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    """entry.py:92-104, unchanged: crop the centre `crop_ratio` and resize back to the original size (bilinear)."""
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)


def build_messages(image_history: dict, instruction: str, crop_ratio: float) -> list[dict]:
    """EvalClient._build_messages (entry.py:134-159), unchanged."""
    videos = {key: [center_crop(frame, crop_ratio) for frame in image_history[key]] for key in CAMERA_KEYS}
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": videos[CAMERA_KEYS[0]]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": videos[CAMERA_KEYS[1]]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": videos[CAMERA_KEYS[2]]},
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def frame_stack(value, frames: tuple) -> np.ndarray:
    """A camera's history as a (T, H, W, 3) uint8 array: from such an array, a list of T frames, or a map of T frames
    keyed "0".."T-1" (oldest first)."""
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: int(k.decode() if isinstance(k, bytes) else k))
        value = [value[k] for k in keys]
    if isinstance(value, (list, tuple)):
        value = np.stack([np.asarray(v) for v in value])
    stack = np.asarray(value)
    if stack.shape != frames or stack.dtype != np.uint8:
        raise ValueError(f"each camera must be {frames[0]} uint8 frames of {frames[1:]}, got {stack.dtype} "
                         f"{stack.shape}")
    return stack


class Policy:
    def __init__(self, name: str, model, processor, run=None, path: str = "stock", reason: str = ""):
        """`model`, `processor`: as deploy/server.py and deploy/client.py load them. `run(data) -> raw actions`: the
        model call, default the server's `model(**data).actions`; the fast path passes its own."""
        self.name, self.variant = name, VARIANTS[name]
        self.model, self.processor = model, processor
        self.run = run or (lambda data: self.model(**data).actions)
        v = self.variant
        self.metadata = {"model": name, "family": "xiaomi_r1", "checkpoint": f"hf://{v['repo']}@{v['revision']}",
                         "commit": COMMIT, "action_shape": [v["horizon"], v["action_dim"]], "action_hz": v["hz"],
                         "cameras": list(v["cameras"]), "keys": [*v["cameras"], "state", "instruction"],
                         "state_dim": v["state_dim"], "prompt_key": "instruction", "image_size": list(v["image_size"]),
                         "history": v["history"], "history_interval": v["interval"], "license": LICENSE,
                         "path": path, "path_reason": reason}

    def _to_model(self, value):
        """deploy/server.py:59: floating tensors to the model's device and dtype, other tensors to its device."""
        if not hasattr(value, "is_floating_point"):  # not a torch tensor (task_id)
            return value
        if value.is_floating_point():
            return value.to(device=self.model.device, dtype=self.model.dtype)
        return value.to(device=self.model.device)

    def request(self, obs: dict) -> dict:
        """EvalClient.infer (entry.py:167-182) up to the request it sends."""
        v = self.variant
        frames = (v["history"], *v["image_size"], 3)
        images = {key: frame_stack(obs[key], frames) for key in v["cameras"]}
        states = np.asarray(obs["state"], dtype=np.float32)
        if states.shape != (v["history"], v["state_dim"]):
            raise ValueError(f"state must be shape ({v['history']}, {v['state_dim']}), got {states.shape}")
        state = np.zeros((1, states.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, : states.shape[-1]] = states
        inputs = self.processor.apply_chat_template(
            build_messages(images, str(obs["instruction"]), v["crop_ratio"]), tokenize=True, return_dict=True,
            return_tensors="pt", do_resize=False, state=state, robot_type=v["robot_type"])
        request = dict(inputs)
        request["task_id"] = v["robot_type"]
        return request

    def infer(self, obs: dict) -> dict:
        import torch

        v = self.variant
        t0 = time.perf_counter()
        request = self.request(obs)
        data = {key: self._to_model(value) for key, value in request.items()}  # deploy/server.py:58-59
        t1 = time.perf_counter()
        seed = obs.get("rs_seed")
        with torch.random.fork_rng(devices=[self.model.device], enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
            raw = self.run(data).cpu()  # deploy/server.py:61-63
        t2 = time.perf_counter()
        actions = self.processor.decode_action(raw, robot_type=request["task_id"])[0, :, : v["action_dim"]]
        actions = np.asarray(actions.detach().float().cpu().numpy(), dtype=np.float32)  # entry.py:183-185
        if not np.isfinite(actions).all():
            raise ValueError("the policy returned non-finite actions")
        # where the time went inside the policy call (the server's infer_ms adds the JPEG decode before it)
        timing = {"preprocess_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3,
                  "postprocess_ms": (time.perf_counter() - t2) * 1e3}
        return {"actions": actions, "policy_timing": timing}


# ----- weights
def weights_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("RS_CKPT_DIR") or "/runpod-volume/pgs/weights/xiaomi_r1")


def hf_listing(repo: str, revision: str) -> list[tuple[str, int, str]]:
    """(path, size, download URL) of every file of a Hugging Face repo at a revision (the Hub's tree API)."""
    url = f"{HF}/api/models/{repo}/tree/{revision}?recursive=1"
    headers = {"User-Agent": "playground-serve-worker"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        entries = json.load(r)
    return [(e["path"], int(e["size"]), f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(e['path'])}")
            for e in entries if e.get("type") == "file"]


def prepare() -> pathlib.Path:
    """The checkpoint directory, downloaded into RS_CKPT_DIR unless that holds a complete copy (stage.py's marker)."""
    import stage  # /rs/stage.py

    v = VARIANTS[os.environ["RS_MODEL"]]
    dest = weights_dir()
    if os.environ.get("RS_CKPT_DIR"):  # boot.sh's stage.py fetch may be writing it: let it end first
        pgs_server.wait_for_fetch(dest)
    if not stage.complete(dest):
        (dest / ".rs_failed").unlink(missing_ok=True)  # boot.sh's stage.py fetch takes gs:// URIs only
        t = time.time()
        uri = f"hf://{v['repo']}@{v['revision']}"
        log("ckpt_fetch_start", uri=uri, dest=str(dest))
        stage.listing = lambda _uri: hf_listing(v["repo"], v["revision"])
        files = stage.fetch(uri, dest)
        n = sum(files.values())
        log("ckpt_fetch_done", bytes=n, files=len(files), s=round(time.time() - t, 1),
            mb_s=round(n / (time.time() - t) / 1e6))
    return dest


# ----- policy
def load_stock(path: pathlib.Path):
    """deploy/server.py:20 and deploy/client.py:22, from the local copy of the repo at the pinned revision."""
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(str(path), trust_remote_code=True, attn_implementation="flash_attention_2",
                                      dtype=torch.bfloat16).cuda().to(torch.bfloat16)
    return model, processor


def decide(name: str, gpu: str) -> tuple[str, str]:
    mode = os.environ.get("RS_FAST", "auto")
    if mode != "auto":
        return "stock", f"fast path turned off (RS_FAST={mode})"
    if name not in FAST:
        return "stock", f"no fast path for {name}"
    if gpu not in FAST[name]:
        return "stock", f"fast path not measured on {gpu or 'this GPU'}, only on {', '.join(FAST[name])}"
    if not selftest_file(name).is_file():
        return "stock", f"no self-test reference for {name}"
    if os.path.exists(FAST_FAILED):
        return "stock", ("an earlier start in this container crashed while it built the fast path: "
                         f"{pathlib.Path(FAST_FAILED).read_text()[:200]}")
    return "fast", f"CUDA graph path on {gpu}, if its boot self-test passes"  # load() updates it after the test


def selftest_file(name: str) -> pathlib.Path:
    return HERE / f"selftest_{name}.json"


def selftest_cases(name: str) -> tuple[list[dict], dict]:
    """The stored observation, an episode's first tick (one frame per camera as JPEG, decoded as server_decode does,
    and one state, each repeated over the history by sample_history), with the stock actions per seed."""
    from server_decode import decode_jpeg

    ref = json.loads(selftest_file(name).read_text())
    v = VARIANTS[name]
    obs = {k: sample_history([decode_jpeg(base64.b64decode(b))], v["history"], v["interval"])
           for k, b in ref["frames_jpeg"].items()}
    obs.update(state=sample_history([np.asarray(ref["state"], np.float32)], v["history"], v["interval"]),
               instruction=ref["instruction"])
    out = [{"obs": dict(obs, rs_seed=case["seed"]), "actions": np.asarray(case["actions"], np.float64),
            "name": case["name"]} for case in ref["cases"]]
    return out, ref["bounds"]


def compare(a, b) -> dict:
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"mean_abs": round(float(d.mean()), 6), "max_abs": round(float(d.max()), 5),
            "finite": bool(np.isfinite(a).all())}


def selftest(name: str, stock: Policy, fast: Policy) -> tuple[bool, dict]:
    """Each stored case on the stock path (against the stored actions) and twice on the fast path (against stock)."""
    cases, bounds = selftest_cases(name)
    detail, ok = {}, True
    for case in cases:
        s = stock.infer(dict(case["obs"]))["actions"]
        vs_stored = compare(s, case["actions"])
        runs = [compare(fast.infer(dict(case["obs"]))["actions"], s) for _ in range(2)]
        worst = max(runs, key=lambda r: r["max_abs"])
        passed = worst["finite"] and worst["max_abs"] <= bounds["fast_max"] and worst["mean_abs"] <= bounds["fast_mean"]
        detail[case["name"]] = {"stock_vs_stored": vs_stored, "fast_vs_stock": worst, "passed": passed}
        if vs_stored["max_abs"] > bounds["stored_max"]:
            detail[case["name"]]["warning"] = "the stock path differs from the stored stock actions"
        ok = ok and passed
    return ok, detail


def build_fast(name, stock, model, processor, gpu, reason, label, compiled):
    """(fast Policy, "") when the boot self-test passes, else (None, why); the stock path is untouched either way."""
    import torch

    t = time.time()
    pathlib.Path(FAST_FAILED).write_text(f"fast path build started ({label})")
    fast = None
    try:
        import xr1_fast

        fast = Policy(name, model, processor, xr1_fast.FastForward(model, compile_dit=compiled), "fast", reason)
        fast.request = xr1_fast.FastRequest(stock, sys.modules[__name__])  # the cached, GPU request builder
        ok, detail = selftest(name, stock, fast)
        log("fast_selftest", variant=label, passed=ok, s=round(time.time() - t, 1), **detail)
        if ok:
            worst = max(d["fast_vs_stock"]["max_abs"] for d in detail.values())
            fast.metadata["path_reason"] = why = (f"{label} on {gpu}; boot self-test passed (max difference from "
                                                  f"stock {worst:g}, built in {time.time() - t:.0f} s)")
            log("path", path="fast", variant=label, reason=why)
            return fast, ""
        why = f"{label}: self-test failed: {json.dumps(detail)[:300]}"
    except Exception as e:  # noqa: BLE001  (any failure: try the next variant, then stock)
        why = f"{label} failed: {e!r}"[:300]
    finally:
        pathlib.Path(FAST_FAILED).unlink(missing_ok=True)
    del fast
    torch.cuda.empty_cache()
    return None, why


def load() -> Policy:
    name = os.environ["RS_MODEL"]
    v = VARIANTS[name]
    t = time.time()
    model, processor = load_stock(weights_dir())
    if v["robot_type"] not in processor.list_robot_types():  # entry.py:125-131
        raise ValueError(f"robot type {v['robot_type']!r} missing; checkpoint has {processor.list_robot_types()}")
    log("stock_loaded", s=round(time.time() - t, 1))
    stock = Policy(name, model, processor)
    gpu = (pgs_server.gpu_info("name").splitlines() or [""])[0].strip()
    path, reason = decide(name, gpu)
    if path == "fast":
        compile_dit = os.environ.get("RS_FAST_COMPILE", "1") == "1"
        if compile_dit:  # inductor's compiled kernels on the volume: only a GPU type's first start compiles
            cache = pathlib.Path(os.environ.get("RS_COMPILE_CACHE", "/runpod-volume/pgs/xiaomi_r1/inductor"))
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / gpu.replace(" ", "_")))
        failures = []
        for label, compiled in ((("compiled DiT step in the CUDA graph", True),) if compile_dit else ()) + (
                ("CUDA graph", False),):
            fast, why = build_fast(name, stock, model, processor, gpu, reason, label, compiled)
            if fast is not None:
                return fast
            failures.append(why)
        reason = "; ".join(failures)[:600]
        log("fallback", reason=reason)
    stock.metadata.update(path="stock", path_reason=reason)
    log("path", path="stock", reason=reason)
    return stock


def make_example() -> dict:
    """Warm-up input: an episode's first tick (one grey frame and a zero state, padded by sample_history)."""
    v = VARIANTS[os.environ["RS_MODEL"]]
    frame = np.full((*v["image_size"], 3), 128, np.uint8)
    obs = {key: sample_history([frame], v["history"], v["interval"]) for key in v["cameras"]}
    state = sample_history([np.zeros(v["state_dim"], np.float32)], v["history"], v["interval"])
    return {**obs, "state": state, "instruction": "Close the blender lid."}


def main() -> None:
    prepare()  # before run_worker, which waits for RS_CKPT_DIR's marker
    pgs_server.run_worker(load, make_example)


if __name__ == "__main__":
    main()
