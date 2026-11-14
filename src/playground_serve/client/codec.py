"""The wire format (openpi's msgpack with raw numpy arrays) and the three image transport modes.

`pack_array`, `unpack_array` and `resize_with_pad` are adapted from openpi-client
(packages/openpi-client/src/openpi_client/msgpack_numpy.py and image_tools.py at openpi commit
215abfb217dbac7d5f1273282331b9b1866c0479), Copyright Physical Intelligence, Apache License 2.0. They are copied so that
this package does not inherit openpi-client's numpy<2 pin and its dm-tree dependency.

Modes, applied to every HxWx3 uint8 array in the observation (nested dicts included):
  raw     the observation as given; the bytes equal openpi-client's.
  resize  resize with padding to size x size (or H x W for size (H, W)), the same PIL bilinear code as
          openpi_client.image_tools.
  jpeg    resize, then JPEG at `quality`; each image becomes {b"__jpeg__": True, b"data": <bytes>}, which the server
          turns back into an array with server_decode.decode_images.
"""

from __future__ import annotations

import io

import msgpack
import numpy as np
from PIL import Image

MODES = ("raw", "resize", "jpeg")
JPEG_KEY = b"__jpeg__"


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=pack_array)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=unpack_array)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """One HxWx3 uint8 image, resized without distortion and padded with zeros (tf.image.resize_with_pad via PIL)."""
    if image.shape[:2] == (height, width):
        return image
    pil = Image.fromarray(image)
    cur_width, cur_height = pil.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height, resized_width = int(cur_height / ratio), int(cur_width / ratio)
    resized = pil.resize((resized_width, resized_height), resample=Image.BILINEAR)
    out = Image.new(resized.mode, (width, height), 0)
    out.paste(resized, (max(0, int((width - resized_width) / 2)), max(0, int((height - resized_height) / 2))))
    return np.asarray(out)


def is_image(value) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_images(obs, mode: str, size=224, quality: int = 90):
    """Return a copy of `obs` with its images transformed for `mode`; other values are unchanged. `size` is an int or
    (H, W)."""
    if mode == "raw":
        return obs
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if isinstance(obs, dict):
        return {k: encode_images(v, mode, size, quality) for k, v in obs.items()}
    if not is_image(obs):
        return obs
    height, width = (size, size) if isinstance(size, int) else size
    image = resize_with_pad(obs, height, width)
    return {JPEG_KEY: True, b"data": encode_jpeg(image, quality)} if mode == "jpeg" else image
