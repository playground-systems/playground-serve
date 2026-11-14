"""Server side of the client's `jpeg` image mode. Standalone (numpy and Pillow only): copy this file into the server.

In `jpeg` mode the client replaces each HxWx3 uint8 image in the observation with the map
{b"__jpeg__": True, b"data": <JPEG bytes>}. openpi's msgpack_numpy.unpackb leaves such a map as a plain dict, so the
stock server passes it to the policy, which fails on it. Wrap the policy so that it decodes the images first.

Where the server calls it: in openpi's scripts/serve_policy.py, `main()`, right after `policy = create_policy(args)`:

    from server_decode import DecodeImages
    policy = DecodeImages(policy)
    # unchanged: policy_metadata = policy.metadata; WebsocketPolicyServer(policy=policy, ...)

The decode then runs inside the server's timed `policy.infer` call, so `server_timing.infer_ms` includes it. raw and
resize observations hold no such maps and pass through unchanged, so one server serves all three modes.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

JPEG_KEY = b"__jpeg__"


def decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected a JPEG image, got {image.format}")
        return np.array(image.convert("RGB"))


def decode_images(obs):
    """Return `obs` with every {b"__jpeg__": True, b"data": ...} map replaced by its HxWx3 uint8 array."""
    if isinstance(obs, dict):
        if obs.get(JPEG_KEY) is True:
            return decode_jpeg(obs[b"data"])
        return {k: decode_images(v) for k, v in obs.items()}
    return obs


class DecodeImages:
    """Wraps an openpi policy (anything with `infer(obs) -> dict`) so that JPEG images are decoded before inference."""

    def __init__(self, policy) -> None:
        self._policy = policy
        Image.preinit()  # load Pillow's JPEG plugin now (~30 ms) instead of in the first request

    def infer(self, obs: dict) -> dict:
        return self._policy.infer(decode_images(obs))

    def __getattr__(self, name):  # metadata, reset and anything else go to the wrapped policy
        return getattr(self._policy, name)
