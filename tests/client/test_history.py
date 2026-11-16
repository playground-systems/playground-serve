"""History (client/history.py) and frame stacks in jpeg mode: the stacks Xiaomi-Robotics-1's own sample_history builds,
sent as one JPEG per frame to a server that advertises "jpeg_stack" and rebuilt there."""

import pathlib
import sys

import numpy as np
import pytest

from playground_serve import History
from playground_serve.client import codec, connect, server_decode


def sample_history(history, length=4, interval=2):  # the model's own (Xiaomi-Robotics-1 eval_robocasa365/entry.py)
    items = list(history)
    indices = [max(0, len(items) - 1 - (length - 1 - index) * interval) for index in range(length)]
    return np.stack([items[index] for index in indices], axis=0)


def test_history_matches_the_models_sampling_and_resets():
    h = History(["cam", "state"], length=4, interval=2)
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(11)]
    states = [np.full(14, i, np.float32) for i in range(11)]
    for t in range(11):
        out = h({"cam": frames[t], "state": states[t], "instruction": "x"})
        assert np.array_equal(out["cam"], sample_history(frames[:t + 1]))
        assert np.array_equal(out["state"], sample_history(states[:t + 1])) and out["instruction"] == "x"
    assert out["cam"].shape == (4, 8, 8, 3) and list(out["state"][:, 0]) == [4, 6, 8, 10]
    h.reset()
    assert list(h({"cam": frames[5], "state": states[5]})["state"][:, 0]) == [5, 5, 5, 5]
    meta = {"history": 4, "history_interval": 2, "cameras": ["video.a", "video.b"]}
    assert History.from_metadata(meta).keys == ["video.a", "video.b", "state"]
    with pytest.raises(ValueError):
        History.from_metadata({"cameras": ["c"]})


def test_stacks_go_as_jpeg_frames_only_where_advertised():
    stack = np.stack([np.full((16, 16, 3), 40 * i, np.uint8) for i in range(4)])
    obs = {"cam": stack, "frames": [stack[0], stack[1]], "state": np.zeros(3)}
    plain = codec.encode_images(obs, "jpeg", None)
    assert plain["cam"] is stack and plain["frames"] is obs["frames"]  # a server without "jpeg_stack": unchanged
    enc = codec.encode_images(obs, "jpeg", None, stacks=True)
    assert enc["cam"][codec.JPEG_STACK_KEY] is True and len(enc["cam"][b"frames"]) == 4
    back = server_decode.decode_images(codec.unpackb(codec.packb(enc)))
    assert back["cam"].shape == (4, 16, 16, 3) and np.abs(back["cam"].astype(int) - stack).max() <= 3
    assert back["frames"][1].shape == (16, 16, 3) and np.array_equal(back["state"], obs["state"])


def test_a_family_server_rebuilds_stacks_end_to_end():
    pkg = pathlib.Path(codec.__file__).resolve().parents[1]
    for d in (pkg / "families" / "common", pkg / "worker"):
        if str(d) not in sys.path:
            sys.path.append(str(d))
    import pgs_server

    class Echo:
        metadata = {"history": 4, "history_interval": 2, "cameras": ["cam"]}
        seen = []

        def infer(self, obs):
            self.seen.append(obs)
            return {"actions": np.zeros((2, 2), np.float32)}

    server = pgs_server.Server(Echo(), host="127.0.0.1", port=0).start()
    try:
        with connect(f"ws://127.0.0.1:{server.port}/ws", wait=10, keepalive=None, size=16) as client:
            h = History.from_metadata(client.metadata)
            for t in range(3):
                client.act(h({"cam": np.full((16, 16, 3), 50 * t, np.uint8), "state": np.full(2, t, np.float32)}))
            assert client.features["jpeg_stack"] is True
    finally:
        server.stop()
    got = Echo.seen[-1]
    assert got["cam"].shape == (4, 16, 16, 3) and [int(f[0, 0, 0]) for f in got["cam"]] == [0, 0, 0, 100]
    assert got["state"].shape == (4, 2) and list(got["state"][:, 0]) == [0, 0, 0, 2]  # ticks 0, 0, 0, 2
