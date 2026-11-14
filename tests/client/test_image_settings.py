"""The client takes the image size and mode from the server's metadata ("image_size", "image_mode"), so a user does not
set them by hand; explicit arguments still win, and a server without them gets 224 x 224 as before."""

import threading

import numpy as np
from conftest import IMAGE_KEYS, RecordingPolicy, free_port
from openpi_websocket_policy_server import WebsocketPolicyServer

from playground_serve.client import connect
from playground_serve.client.server_decode import decode_images

FEATURES = {"playground_serve": {"jpeg": True, "noop": True}}


class DecodingPolicy(RecordingPolicy):
    """What a Playground Serve worker hands its policy: jpeg-mode images decoded back to arrays."""

    def infer(self, obs: dict) -> dict:
        return super().infer(decode_images(obs))


def serve(metadata: dict) -> tuple:
    policy, port = DecodingPolicy(), free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata=metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return policy, f"ws://127.0.0.1:{port}"


def shapes(policy) -> list:
    return [policy.seen[-1][key].shape for key in IMAGE_KEYS]


def test_size_from_metadata(observations):
    policy, url = serve({**FEATURES, "image_size": [378, 378]})
    with connect(url, wait=10, keepalive=None) as client:
        assert (client.mode, client.size) == ("jpeg", (378, 378))
        client.infer(observations[0])
        assert shapes(policy) == [(378, 378, 3)] * 2
    with connect(url, wait=10, keepalive=None, size=224) as client:  # an explicit size wins
        client.infer(observations[0])
        assert client.size == (224, 224) and shapes(policy) == [(224, 224, 3)] * 2


def test_non_square_size_and_int_size(observations):
    policy, url = serve({"image_size": [180, 320]})  # a stock server: resize mode, letterboxed to 180 x 320
    with connect(url, wait=10, keepalive=None) as client:
        client.infer(observations[0])
        assert client.mode == "resize" and shapes(policy) == [(180, 320, 3)] * 2
        frame = policy.seen[-1][IMAGE_KEYS[0]]
        assert frame[:, :20].max() == 0 and frame[:, -20:].max() == 0  # the 256 x 256 render, padded left and right
    policy, url = serve({**FEATURES, "image_size": 256})
    with connect(url, wait=10, keepalive=None) as client:
        assert client.size == (256, 256)


def test_mode_from_metadata(observations):
    policy, url = serve({**FEATURES, "image_mode": "raw", "image_size": [378, 378]})
    with connect(url, wait=10, keepalive=None) as client:
        assert client.mode == "raw"
        client.infer(observations[0])
        assert all(np.array_equal(policy.seen[-1][k], observations[0][k]) for k in IMAGE_KEYS)
    with connect(url, wait=10, keepalive=None, mode="jpeg") as client:  # an explicit mode wins
        assert client.mode == "jpeg"
    policy, url = serve({"image_mode": "jpeg"})  # jpeg asked for, but the server cannot decode it: resize
    with connect(url, wait=10, keepalive=None) as client:
        assert client.mode == "resize"


def test_defaults_without_image_settings(observations):
    policy, url = serve({**FEATURES, "image_size": "large", "image_mode": "png"})  # unusable values are ignored
    with connect(url, wait=10, keepalive=None) as client:
        assert (client.mode, client.size) == ("jpeg", (224, 224))
        client.infer(observations[0])
        assert shapes(policy) == [(224, 224, 3)] * 2
