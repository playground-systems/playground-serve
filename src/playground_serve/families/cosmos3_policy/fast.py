"""The Cosmos 3 policy's fast path and its boot self-test (standard library and numpy at import; /rs/fast.py on the
worker, next to serve.py).

The fast path is the stock service (same checkpoint, bf16 weights, compile settings, 4 UniPC steps, guidance 3.0 and
interval, shift 5.0, seed handling) with a faster pre-processing step and nothing else. RobolabPolicyService builds a
33-frame clip [3, 33, H, W] whose frames after the first are zeros (only the first is an observation; the model encodes
only that frame), and its transform pipeline resizes and pads all 33 frames on the CPU. ZeroTailFunctional runs those
two per-frame operations on the first frame only and fills the other frames with zeros, which is what resize (bicubic,
antialias) and pad (reflect or edge) give for zero frames; it falls back to torchvision for any clip whose later
frames are not all zero. So the sample is the same tensor, bit for bit, which the self-test checks first.

Measured and dropped (NOTES.md): OmniSetupOverrides use_cuda_graphs=True (fails at capture: a CPU to GPU copy inside
the block), compile_dynamic=False (no gain, and actions moved 0.011 mean / 0.085 max from stock), and
generate_samples_from_batch(use_batched_cfg=True) (slower on the A40: 1107 to 1150 ms against 837 to 860).

It serves only on a GPU where it was measured (FAST_GPUS) and only after its self-test: the fast sample of the stored
observation must equal the stock sample exactly, and its actions at the stored seed must match the stock actions
stored for that GPU (selftest_<model>.json) within MAX_ABS / MEAN_ABS (the cross-process noise floor), twice.
Otherwise serve.py serves the stock path and says why (metadata path_reason, log line "fallback").
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TRANSFORMS_MODULE = "cosmos_framework.data.generator.action.utils.transforms"
# Model -> GPU names (nvidia-smi) where the fast path was measured served and checked against stock (NOTES.md).
# H100: 235 ms median against 274 for stock (JPEG, one worker, requests alternating, 16 torch threads); RTX 5090: 439.5
# against 514.0 (same method); A40: 839 against 853 (another worker, before the thread cap) and 755 against 830 in the
# worker with 360 x 640 frames. The sample is equal bit for bit (checked at every start) and the actions were identical
# (0.0) in the same process.
FAST_GPUS: dict[str, tuple[str, ...]] = {
    "cosmos3_edge_droid": ("NVIDIA A40", "NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 5090")}
# Bounds of the self-test, from the measured noise floor (NOTES.md): the same seed on another A40 worker (another host
# and process) moved the stock actions by 0.004 to 0.0055 mean and 0.026 to 0.039 max; a new seed by 0.089 to 0.10 mean
# and 0.26 to 0.49 max.
MAX_ABS, MEAN_ABS = 0.1, 0.02


def reference_path(model: str) -> pathlib.Path:
    return HERE / f"selftest_{model}.json"


def decide(mode: str, model: str, gpu: str) -> tuple[bool, str]:
    if mode == "off":
        return False, "fast path turned off (RS_FAST=off)"
    if gpu not in FAST_GPUS.get(model, ()):
        return False, f"fast path not measured on {gpu or 'this GPU'} for {model}"
    if not reference_path(model).is_file() or gpu not in load_reference(model)["references"]:
        return False, f"no self-test reference for {model} on {gpu}"
    return True, f"first-frame-only clip resize and pad on {gpu}, if its boot self-test passes"


def load_reference(model: str) -> dict:
    return json.loads(reference_path(model).read_text())


def decode_observation(o: dict) -> dict:
    """A request from a stored observation (droid_obs.json, selftest_<model>.json): JPEG frames in base32, lists."""
    import base64
    import io

    from PIL import Image

    def frame(b32: str) -> np.ndarray:
        with Image.open(io.BytesIO(base64.b32decode(b32))) as im:
            return np.array(im.convert("RGB"))

    return {**{f"observation/{k}": frame(o[k]) for k in ("wrist_image_left", "exterior_image_1_left",
                                                           "exterior_image_2_left")},
            "observation/joint_position": np.asarray(o["joint_position"], np.float32),
            "observation/gripper_position": np.asarray(o["gripper_position"], np.float32), "prompt": o["prompt"]}


def check(actions, want) -> tuple[bool, dict]:
    a, w = np.asarray(actions, np.float64), np.asarray(want, np.float64)
    if a.shape != w.shape:
        return False, {"error": f"actions shape {list(a.shape)}, stock {list(w.shape)}"}
    if not np.isfinite(a).all():
        return False, {"error": "non-finite actions"}
    d = np.abs(a - w)
    detail = {"mean_abs_diff": round(float(d.mean()), 5), "max_abs_diff": round(float(d.max()), 4)}
    return bool(d.max() <= MAX_ABS and d.mean() <= MEAN_ABS), detail


def selftest(policy, ref: dict, gpu: str, attempts: int = 2) -> tuple[bool, dict]:
    """Run the stored observation at its seed `attempts` times; every run must match the stock actions stored for
    this GPU."""
    want = ref["references"][gpu]["actions"]
    obs = {**decode_observation(ref["observation"]), "rs_seed": ref["seed"]}
    detail: dict = {}
    for _ in range(attempts):
        ok, detail = check(policy.infer(dict(obs))["actions"], want)
        if not ok:
            return False, detail
    return True, detail


class ZeroTailFunctional:
    """torchvision.transforms.functional for the transforms module: resize and pad of a [C, T, H, W] clip whose frames
    after the first are all zero run on the first frame, and the result's other frames are zeros; everything else is
    torchvision's own function."""

    def __init__(self, functional):
        self._f = functional

    def __getattr__(self, name):
        return getattr(self._f, name)

    def _first_frame(self, fn, tensor, *args, **kwargs):
        import torch

        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[1] > 1 and not tensor[:, 1:].any():
            head = fn(tensor[:, :1], *args, **kwargs)
            out = torch.zeros((tensor.shape[0], tensor.shape[1], *head.shape[2:]), dtype=head.dtype)
            out[:, :1] = head
            return out
        return fn(tensor, *args, **kwargs)

    def resize(self, tensor, *args, **kwargs):
        return self._first_frame(self._f.resize, tensor, *args, **kwargs)

    def pad(self, tensor, *args, **kwargs):
        return self._first_frame(self._f.pad, tensor, *args, **kwargs)


class zero_tail:
    """Context: the transforms module uses ZeroTailFunctional (requests run one at a time on the main thread)."""

    def __enter__(self):
        import importlib

        self._mod = importlib.import_module(TRANSFORMS_MODULE)
        self._f = self._mod.transforms_F
        self._mod.transforms_F = ZeroTailFunctional(self._f)

    def __exit__(self, *exc):
        self._mod.transforms_F = self._f


def same_sample(a, b) -> bool:
    """Two _build_sample results are equal: tensors exactly (shape, dtype, values), containers item by item, other
    objects by their fields or ==."""
    import dataclasses

    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
                and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(
            same_sample(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_sample(x, y) for x, y in zip(a, b))
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(same_sample(getattr(a, f.name), getattr(b, f.name))
                                          for f in dataclasses.fields(a))
    if hasattr(a, "__dict__") and type(a) is type(b) and type(a).__eq__ is object.__eq__:
        return same_sample(vars(a), vars(b))
    return bool(a == b)


def sample_check(service, obs: dict) -> tuple[bool, dict]:
    """The stock and the fast _build_sample of one observation, and their CPU times."""
    import time

    t = time.perf_counter()
    stock = service._build_sample(dict(obs))
    t_stock = time.perf_counter() - t
    t = time.perf_counter()
    with zero_tail():
        quick = service._build_sample(dict(obs))
    t_fast = time.perf_counter() - t
    return same_sample(stock, quick), {"stock_ms": round(t_stock * 1000, 1), "fast_ms": round(t_fast * 1000, 1)}
