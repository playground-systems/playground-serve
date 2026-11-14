"""Boot self-test of the fast path. selftest_<model>.json holds one real observation (pi05_libero: LIBERO bench frame
0, both camera images as the client's JPEG q90 bytes, the state, the prompt), a noise seed, and the actions stock
openpi gave for it (openpi's unchanged JAX policy, rs_seed = seed, through the gateway on an RTX 4090). The fast path must reproduce them up to rounding before the worker reports ready.
"""

from __future__ import annotations

import base64
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# Rounding-level bounds. Measured on checked processes: max 0.011 to 0.018, mean 0.0014 to 0.0022; a new noise draw
# moves the actions by 0.036 mean and 2.0 max (docs/BENCHMARKS.md).
MAX_ABS, MEAN_ABS = 0.1, 0.02


def load(model: str) -> dict:
    return json.loads((HERE / f"selftest_{model}.json").read_text())


def observation(ref: dict) -> dict:
    """The stored observation as the client's jpeg mode sends it (server_decode turns it into arrays), with its seed."""
    jpeg = lambda k: {b"__jpeg__": True, b"data": base64.b64decode(ref[k])}  # noqa: E731
    return {"observation/image": jpeg("image_jpeg"), "observation/wrist_image": jpeg("wrist_jpeg"),
            "observation/state": np.asarray(ref["state"]), "prompt": ref["prompt"], "rs_seed": ref["seed"]}


def check(actions, ref: dict) -> tuple[bool, dict]:
    """(passed, detail) for `actions` against the stored stock actions: finite, same shape, within the bounds."""
    a, want = np.asarray(actions, np.float64), np.asarray(ref["actions"], np.float64)
    if a.shape != want.shape:
        return False, {"error": f"actions shape {list(a.shape)}, stock {list(want.shape)}"}
    if not np.isfinite(a).all():
        return False, {"error": "non-finite actions"}
    d = np.abs(a - want)
    detail = {"mean_abs_diff": round(float(d.mean()), 5), "max_abs_diff": round(float(d.max()), 4)}
    return bool(d.max() <= MAX_ABS and d.mean() <= MEAN_ABS), detail
