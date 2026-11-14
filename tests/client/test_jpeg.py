"""jpeg mode: what the stock server's unpack yields, decoded by server_decode, is the resized image up to JPEG loss."""

import io

import numpy as np
import pytest
from conftest import IMAGE_KEYS, RecordingPolicy
from openpi_client import msgpack_numpy
from PIL import Image

from playground_serve.client import connect
from playground_serve.client.codec import encode_images, packb
from playground_serve.client.mock_server import MockServer
from playground_serve.client.server_decode import DecodeImages, decode_images


def test_jpeg_round_trip(observations):
    for obs in observations:
        wire = packb(encode_images(obs, "jpeg", quality=90))
        received = msgpack_numpy.unpackb(wire)  # what openpi's server hands to the policy
        assert all(isinstance(received[key], dict) for key in IMAGE_KEYS)
        decoded = decode_images(received)
        resized = encode_images(obs, "resize")
        for key in IMAGE_KEYS:
            assert decoded[key].shape == (224, 224, 3) and decoded[key].dtype == np.uint8
            error = np.abs(decoded[key].astype(int) - resized[key].astype(int)).mean()
            assert error < 3.0, f"{key}: mean absolute error {error:.2f}"
        assert np.array_equal(decoded["observation/state"], obs["observation/state"])
        assert decoded["prompt"] == obs["prompt"]
        assert len(wire) < 0.1 * len(packb(resized))


def test_decode_images_wrapper(observations):
    inner = RecordingPolicy()
    policy = DecodeImages(inner)
    assert policy.metadata == inner.metadata
    policy.infer(msgpack_numpy.unpackb(packb(encode_images(observations[0], "jpeg"))))
    assert all(isinstance(inner.seen[0][key], np.ndarray) for key in IMAGE_KEYS)


def test_decode_rejects_other_formats():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    with pytest.raises(ValueError, match="JPEG"):
        decode_images({"img": {b"__jpeg__": True, b"data": buf.getvalue()}})


def test_jpeg_through_mock(observations):
    with MockServer() as mock, connect(mock.url, mode="jpeg") as client:
        out = client.infer(observations[0])  # the mock raises if an image reaches inference undecoded
        assert out["actions"].shape == (10, 7) and client.timing["payload_bytes"] < 40_000
