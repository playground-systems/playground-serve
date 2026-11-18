"""Server side of the client's `jpeg` image mode. Standalone (numpy and Pillow only): copy this file into the server.

In `jpeg` mode the client replaces each HxWx3 uint8 image in the observation with the map
{b"__jpeg__": True, b"data": <JPEG bytes>}. openpi's msgpack_numpy.unpackb leaves such a map as a plain dict, so the
stock server passes it to the policy, which fails on it. Wrap the policy so that it decodes the images first.

Where the server calls it: in openpi's scripts/serve_policy.py, `main()`, right after `policy = create_policy(args)`:

    from server_decode import DecodeImages
    policy = DecodeImages(policy)
    # unchanged: policy_metadata = policy.metadata; WebsocketPolicyServer(policy=policy, ...)

The decode then runs inside the server's timed `policy.infer` call, so `server_timing.infer_ms` includes it. raw and
resize observations hold no such maps and pass through unchanged, so one server serves all three modes. Two or more
images are decoded at the same time on a few threads (Pillow's decoder releases the GIL; DECODE_THREADS, at most the
container's CPU count): the same libjpeg calls, so the same pixels as one after another.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

JPEG_KEY = b"__jpeg__"
JPEG_STACK_KEY = b"__jpeg_stack__"  # a (T, H, W, 3) stack, one JPEG per frame (servers advertising "jpeg_stack")
DECODE_THREADS = max(1, min(8, int(os.environ.get("RUNPOD_CPU_COUNT") or os.cpu_count() or 1)))
_pool = None


def decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected a JPEG image, got {image.format}")
        return np.array(image.convert("RGB"))


def _jpegs(obs, out: list) -> None:
    """The JPEG payloads in `obs`, in the order decode_images rebuilds them."""
    if isinstance(obs, dict):
        if obs.get(JPEG_KEY) is True:
            out.append(obs[b"data"])
        elif obs.get(JPEG_STACK_KEY) is True:
            out.extend(obs[b"frames"])
        else:
            for v in obs.values():
                _jpegs(v, out)
    elif isinstance(obs, list):
        for v in obs:
            _jpegs(v, out)


def _rebuild(obs, frames):
    if isinstance(obs, dict):
        if obs.get(JPEG_KEY) is True:
            return next(frames)
        if obs.get(JPEG_STACK_KEY) is True:
            return np.stack([next(frames) for _ in obs[b"frames"]])
        return {k: _rebuild(v, frames) for k, v in obs.items()}
    if isinstance(obs, list):
        return [_rebuild(v, frames) for v in obs]
    return obs


def decode_images(obs, threads: int | None = None):
    """Return `obs` with every {b"__jpeg__": True, b"data": ...} map replaced by its HxWx3 uint8 array, every
    {b"__jpeg_stack__": True, b"frames": [...]} map by its (T, H, W, 3) stack, inside maps and lists. Two or more
    images decode on up to `threads` (default DECODE_THREADS) threads at once."""
    global _pool
    payloads: list = []
    _jpegs(obs, payloads)
    if not payloads:
        return obs
    threads = DECODE_THREADS if threads is None else threads
    if len(payloads) < 2 or threads < 2:
        arrays = [decode_jpeg(p) for p in payloads]
    else:
        if _pool is None or _pool._max_workers < threads:
            _pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="jpeg")
        arrays = list(_pool.map(decode_jpeg, payloads))
    return _rebuild(obs, iter(arrays))


class DecodeImages:
    """Wraps an openpi policy (anything with `infer(obs) -> dict`) so that JPEG images are decoded before inference."""

    def __init__(self, policy) -> None:
        self._policy = policy
        Image.preinit()  # load Pillow's JPEG plugin now (~30 ms) instead of in the first request

    def infer(self, obs: dict) -> dict:
        return self._policy.infer(decode_images(obs))

    def __getattr__(self, name):  # metadata, reset and anything else go to the wrapped policy
        return getattr(self._policy, name)
