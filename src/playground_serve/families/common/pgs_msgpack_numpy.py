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
    if b"__ndarray__" not in obj and b"__npgeneric__" not in obj:
        return obj
    dtype = np.dtype(obj[b"dtype"])
    # Object arrays interpret the wire bytes as Python pointers. Reject them on input,
    # including structured/subarray dtypes containing objects, before constructing anything.
    if dtype.hasobject or dtype.kind not in "biufSU" or not 0 < dtype.itemsize <= 32 * 1024 * 1024:
        raise ValueError("unsupported wire dtype")
    data = obj[b"data"]
    if b"__ndarray__" in obj:
        shape = obj[b"shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) > 32:
            raise ValueError("invalid wire shape")
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 0 or dim > 32 * 1024 * 1024:
                raise ValueError("invalid wire dimension")
            count *= dim
            if count * dtype.itemsize > 32 * 1024 * 1024:
                raise ValueError("wire array too large")
        if not isinstance(data, bytes) or len(data) != count * dtype.itemsize:
            raise ValueError("wire array byte length does not match shape")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)
    if not isinstance(data, (bool, int, float, str, bytes)):
        raise ValueError("invalid wire scalar")
    return dtype.type(data)


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=pack_array)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=unpack_array)


class Packer:
    """The interface used by openpi's server, with the same safe wire codec."""

    def pack(self, obj):
        return packb(obj)
