"""server_decode decodes several JPEG payloads at once on a few threads: the same pixels, in the same places, as one
after another (a request with three cameras of four frames each, a single image, a list and plain values)."""

import numpy as np

from playground_serve.client import codec, server_decode


def test_threaded_decode_equals_serial():
    rng = np.random.default_rng(3)
    frames = {f"cam{i}": rng.integers(0, 256, (4, 64, 96, 3), dtype=np.uint8) for i in range(3)}
    obs = {**frames, "wrist": rng.integers(0, 256, (48, 48, 3), dtype=np.uint8),
           "list": [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)] * 2, "state": np.arange(5.0), "task": "x"}
    wire = codec.unpackb(codec.packb(codec.encode_images(obs, "jpeg", None, stacks=True)))
    serial = server_decode.decode_images(wire, threads=1)
    threaded = server_decode.decode_images(wire, threads=8)
    for key in frames:
        assert threaded[key].shape == (4, 64, 96, 3) and np.array_equal(threaded[key], serial[key])
    assert np.array_equal(threaded["wrist"], serial["wrist"])
    assert all(np.array_equal(a, b) for a, b in zip(threaded["list"], serial["list"]))
    assert np.array_equal(threaded["state"], obs["state"]) and threaded["task"] == "x"
    assert server_decode.DECODE_THREADS >= 1
