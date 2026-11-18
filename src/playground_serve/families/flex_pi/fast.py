"""Flex-pi's fast path and its boot self-test (numpy and json at import; /rs/fast.py on the worker, next to serve.py).

The fast path is the same model, checkpoint and bf16 weights, called the same way (serve.py Policy), with options the
repo's own Model.prepare_for_inference offers for inference (FAST_OPTIONS; models/backbone.py documents each as the
same math, and the repo's RoboTwin launchers run torch.compile by default): torch.compile with CUDA graphs per denoise
step. Same 4 denoise steps, same seed 42 per call, same 256 x 256 input path, same pre and post processing.

It serves only on a GPU where it was measured (FAST_GPUS) and only after its self-test: the stored observation must give
the stock actions stored for that GPU (selftest_<model>.json) within MAX_ABS / MEAN_ABS, twice, with and without depth.
Otherwise serve.py switches the model back to the evaluator's eager setting and says why.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# torch.compile with CUDA graphs per denoise step (the RoboTwin launchers' default), attn_backend "auto" (the action
# path's all-True masks dropped so SDPA can pick flash), glue_cache (memoized per-call constants, documented as
# bit-identical) and compile_encoders (the VAE, DINO and pointmap first-frame encoders compiled). Measured on an A40,
# action-only (NOTES.md): 133 ms without depth against 301 for stock. Scope "loop" was 7 % faster still but took
# 3 + 6 minutes to compile at start, too long for a serverless start (Runpod restarted a worker not healthy within
# about 9.5 minutes).
FAST_OPTIONS = dict(torch_compile=True, torch_compile_mode="reduce-overhead", quantization=None, attn_backend="auto",
                    glue_cache=True, compile_encoders=True)
# Model -> GPU names (nvidia-smi) where the fast path was measured served and checked against stock (NOTES.md).
FAST_GPUS: dict[str, tuple[str, ...]] = {
    "flex_pi_libero": ("NVIDIA A40", "NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 5090")}
# Bounds of the self-test (NOTES.md): the fast path moved actions by 0.0006 to 0.0008 mean and 0.005 to 0.0075 max from
# stock; stock is bit-identical across worker starts; a new noise seed moves them by 0.007 mean and 0.06 max.
MAX_ABS, MEAN_ABS = 0.03, 0.003


def reference_path(model: str) -> pathlib.Path:
    return HERE / f"selftest_{model}.json"


def regime(joint: bool) -> str:
    return "full_joint" if joint else "action_only"


def decide(mode: str, model: str, gpu: str, joint: bool = False) -> tuple[bool, str]:
    if mode == "off":
        return False, "fast path turned off (RS_FAST=off)"
    if gpu not in FAST_GPUS.get(model, ()):
        return False, f"fast path not measured on {gpu or 'this GPU'} for {model}"
    ref = reference_path(model)
    refs = json.loads(ref.read_text()).get("references", {}) if ref.is_file() else {}
    if regime(joint) not in refs.get(gpu, {}):
        return False, f"no self-test reference for {model} on {gpu} in the {regime(joint)} regime"
    return True, f"torch.compile with CUDA graphs on {gpu}, if its boot self-test passes"


def decode_observation(o: dict, depth: bool = True) -> dict:
    """A request from a stored observation (libero_obs.json, selftest_<model>.json): JPEG and PNG in base32, lists."""
    import base64
    import io

    from PIL import Image

    def array(b32: str, rgb: bool) -> np.ndarray:
        with Image.open(io.BytesIO(base64.b32decode(b32))) as im:
            return np.array(im.convert("RGB")) if rgb else np.array(im).astype(np.uint16)

    obs = {"observation/image": array(o["image"], True), "observation/wrist_image": array(o["wrist_image"], True),
           "observation/state": np.asarray(o["state"], np.float32), "prompt": o["prompt"]}
    if depth:
        obs["observation/image_depth"] = array(o["image_depth"], False)
        obs["observation/wrist_image_depth"] = array(o["wrist_image_depth"], False)
    return obs


def check(actions, want) -> tuple[bool, dict]:
    a, w = np.asarray(actions, np.float64), np.asarray(want, np.float64)
    if a.shape != w.shape:
        return False, {"error": f"actions shape {list(a.shape)}, stock {list(w.shape)}"}
    if not np.isfinite(a).all():
        return False, {"error": "non-finite actions"}
    d = np.abs(a - w)
    detail = {"mean_abs_diff": round(float(d.mean()), 5), "max_abs_diff": round(float(d.max()), 4)}
    return bool(d.max() <= MAX_ABS and d.mean() <= MEAN_ABS), detail


def selftest(policy, ref: dict, gpu: str, joint: bool = False, attempts: int = 2) -> tuple[bool, dict]:
    """The stored observation with and without depth, `attempts` times each, against the stock actions stored for
    this GPU and regime."""
    want = ref["references"][gpu][regime(joint)]
    detail: dict = {}
    for depth, key in ((True, "actions"), (False, "actions_no_depth")):
        obs = decode_observation(ref["observation"], depth=depth)
        for _ in range(attempts):
            ok, detail = check(policy.infer(dict(obs))["actions"], want[key])
            if not ok:
                return False, {"depth": depth, **detail}
    return True, detail


def choose(name: str, make, finish, path_options: dict, *, gpu: str, mode: str, log, joint: bool = False):
    """serve.py's path decision: prepare the fast options and self-test them, else the evaluator's eager setting."""
    import time

    use, reason = decide(mode, name, gpu, joint)
    log("path", path="fast" if use else "stock", reason=reason)
    if use:
        t = time.time()
        try:
            finish(FAST_OPTIONS)
            policy = make("fast", reason)
            ok, detail = selftest(policy, json.loads(reference_path(name).read_text()), gpu, joint)
            log("fast_selftest", passed=ok, s=round(time.time() - t, 1), **detail)
            if ok:
                return policy
            reason = f"fast path failed its self-test against stock actions: {detail}"
        except Exception as e:  # noqa: BLE001  (any failure: serve stock)
            reason = f"fast path failed: {e!r}"[:400]
        log("fallback", reason=reason)
    finish(path_options["stock"])
    return make("stock", reason)
