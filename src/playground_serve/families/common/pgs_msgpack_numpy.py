"""openpi's wire format: msgpack with numpy arrays as {__ndarray__, data, dtype, shape} maps.

`pack_array` and `unpack_array` are adapted from openpi-client
(packages/openpi-client/src/openpi_client/msgpack_numpy.py at openpi commit 215abfb217dbac7d5f1273282331b9b1866c0479),
Copyright Physical Intelligence, Apache License 2.0, by way of the Playground Serve client's codec.py. Copied so that
model environments need only msgpack and numpy.
"""

from __future__ import annotations

import msgpack
import numpy as np


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
