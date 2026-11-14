"""raw mode is byte-for-byte openpi-client's format, and the client and the mock interoperate with real openpi code."""

import socket
import threading

import numpy as np
from conftest import IMAGE_KEYS, RecordingPolicy, free_port
from openpi_client import image_tools, msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_websocket_policy_server import WebsocketPolicyServer

from playground_serve.client import connect
from playground_serve.client.codec import encode_images, packb
from playground_serve.client.mock_server import Config, MockServer


def test_raw_bytes_equal_openpi_client(observations):
    for obs in observations:
        wire = packb(encode_images(obs, "raw"))
        assert wire == msgpack_numpy.Packer().pack(obs)
        back = msgpack_numpy.unpackb(wire)
        assert back["prompt"] == obs["prompt"]
        for key in (*IMAGE_KEYS, "observation/state"):
            assert back[key].dtype == obs[key].dtype and np.array_equal(back[key], obs[key])


def test_resize_equals_openpi_image_tools(observations):
    for obs in observations:
        resized = encode_images(obs, "resize")
        for key in IMAGE_KEYS:
            assert np.array_equal(resized[key], image_tools.resize_with_pad(obs[key], 224, 224))


def test_client_against_openpi_server(observations):
    policy = RecordingPolicy()
    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={"config": "pi05_libero"})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with connect(f"ws://127.0.0.1:{port}", mode="raw", wait=10) as client:  # retries until the server is listening
        assert client.metadata == {"config": "pi05_libero"}
        assert client._ws.socket.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
        out = client.infer(observations[0])
        assert out["actions"].shape == (10, 7) and out["server_timing"]["infer_ms"] >= 0
        assert client.timing["infer_ms"] == out["server_timing"]["infer_ms"]
        seen = policy.seen[0]
        assert set(seen) == set(observations[0]) and seen["prompt"] == observations[0]["prompt"]
        for key in (*IMAGE_KEYS, "observation/state"):
            assert np.array_equal(seen[key], observations[0][key])
    with connect(f"ws://127.0.0.1:{port}", mode="resize") as client:
        client.infer(observations[0])
        assert all(policy.seen[1][key].shape == (224, 224, 3) for key in IMAGE_KEYS)


def test_openpi_client_against_mock(observations):
    with MockServer(Config(infer_ms=1)) as mock:
        client = WebsocketClientPolicy(host=mock.url)
        assert client.get_server_metadata()["mock"] is True
        out = client.infer(observations[0])
        assert out["actions"].shape == (10, 7) and out["server_timing"]["infer_ms"] >= 1
