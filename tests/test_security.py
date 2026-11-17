"""Regression checks for wire input, credential scope, checkpoint paths and episode ownership."""

import importlib.util
import pathlib
import sys
import time

import msgpack
import numpy as np
import pytest

from playground_serve import bundle, models, resolve, state
from playground_serve.client import codec
from playground_serve.worker import stage

ROOT = pathlib.Path(bundle.__file__).parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER_CODEC = load_module("safe_worker_codec", ROOT / "families/common/pgs_msgpack_numpy.py")


@pytest.mark.parametrize("decoder", [codec, WORKER_CODEC])
@pytest.mark.parametrize("dtype", ["O", [("nested", "O")], "V8", "c16"])
def test_wire_rejects_unsafe_dtypes_before_constructing_arrays(decoder, dtype):
    # No dereference: the old decoder would create attacker-controlled Python pointers.
    payload = {b"__ndarray__": True, b"dtype": dtype, b"shape": [1], b"data": bytes(16)}
    with pytest.raises((ValueError, TypeError)):
        decoder.unpackb(msgpack.packb(payload))
    with pytest.raises((ValueError, TypeError)):
        decoder.unpackb(msgpack.packb({b"__npgeneric__": True, b"dtype": dtype, b"data": 1}))


@pytest.mark.parametrize("decoder", [codec, WORKER_CODEC])
@pytest.mark.parametrize("shape,data", [([1], b""), ([1], bytes(16)), ([-1], b""), ([True], bytes(8)),
                                         ([2**40], b""), ([1] * 33, bytes(8))])
def test_wire_rejects_invalid_shapes_and_lengths(decoder, shape, data):
    with pytest.raises(ValueError):
        decoder.unpackb(msgpack.packb({b"__ndarray__": True, b"dtype": "f8", b"shape": shape, b"data": data}))


@pytest.mark.parametrize("decoder", [codec, WORKER_CODEC])
def test_valid_wire_arrays_and_scalars_remain_compatible(decoder):
    for a in (np.arange(12, dtype=np.float32).reshape(3, 4), np.zeros((0, 3)), np.array(2), np.float32(1.5)):
        np.testing.assert_array_equal(decoder.unpackb(codec.packb(a)), a)
    shipped = bundle.unpack(bundle.endpoint_env(models.get("pi05_libero"), "security-test")["RS_FILES"])
    assert shipped["pgs_msgpack_numpy.py"] == (ROOT / "families/common/pgs_msgpack_numpy.py").read_bytes()


@pytest.mark.parametrize("url,trusted", [
    ("wss://endpoint.api.runpod.ai/ws", True), ("wss://endpoint.api.runpod.ai:443/ws", True),
    ("ws://endpoint.api.runpod.ai/ws", False), ("wss://endpoint.api.runpod.ai.example/ws", False),
    ("wss://endpoint.api.runpod.ai@elsewhere.example/ws", False),
    ("wss://endpoint.api.runpod.ai:8443/ws", False), ("ws://127.0.0.1:8000/ws", False),
])
@pytest.mark.parametrize("source", ["argument", "env", "state"])
def test_ambient_runpod_key_is_scoped_to_tls_gateway(monkeypatch, tmp_path, url, trusted, source):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "dummy-runpod-key")
    monkeypatch.delenv(resolve.URL_ENV, raising=False)
    monkeypatch.setattr("playground_serve.client.connect", lambda url, **kw: kw["api_key"])
    target = url
    if source == "env":
        monkeypatch.setenv(resolve.URL_ENV, url)
        target = "pi05_libero"
    elif source == "state":
        state.save(state.default_path(), {"models": {"pi05_libero": {"endpoint": {"url": url}}}})
        target = "pi05_libero"
    assert resolve.connect(target) == ("dummy-runpod-key" if trusted else None)
    assert resolve.connect(target, api_key="custom-service-key") == "custom-service-key"


@pytest.mark.parametrize("bad", ["../outside", "nested/../../outside", "/outside", "nested/../file",
                                "nested\\outside", "C:/outside", ".rs_complete", "nested/.rs_small"])
def test_checkpoint_listing_rejected_before_any_writes(monkeypatch, tmp_path, bad):
    dest = tmp_path / "checkpoint"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    marker = dest / stage.MARKER
    marker.write_text('{"files": {}}')
    monkeypatch.setattr(stage, "listing", lambda uri: [("valid", 0, "unused"), (bad, 0, "unused")])
    with pytest.raises(ValueError):
        stage.fetch("gs://example/model", dest)
    assert outside.read_bytes() == b"keep"
    assert marker.read_text() == '{"files": {}}'
    assert not (dest / "valid").exists()


def test_checkpoint_symlink_and_marker_paths_are_rejected(monkeypatch, tmp_path):
    dest = tmp_path / "checkpoint"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    (dest / "weights").symlink_to(outside)
    monkeypatch.setattr(stage, "listing", lambda uri: [("weights", 0, "unused")])
    with pytest.raises(ValueError):
        stage.fetch("gs://example/model", dest)
    (dest / stage.MARKER).write_text('{"files": {"../outside": 4}}')
    assert not stage.complete(dest)
    with pytest.raises(ValueError):
        stage.copy_dir(dest, tmp_path / "copy")
    assert outside.read_bytes() == b"keep"


def test_episode_id_alone_cannot_resume_or_replace_session(monkeypatch):
    for folder in ("families/llm_agent", "families/common", "worker"):
        monkeypatch.syspath_prepend(str(ROOT / folder))
    module = load_module("agent_security_server", ROOT / "families/llm_agent/serve.py")

    class Session:
        def __init__(self, key, card, *args):
            self.card, self.touched, self.steps = card, time.monotonic(), 0

        def step(self, request):
            self.steps += 1
            return {"agent": {"steps": self.steps}}

    monkeypatch.setattr(module.llm_session, "Session", Session)
    policy = module.AgentPolicy(module.llm_session.Settings())
    req = {"episode": "shared-name", "embodiment": "libero_franka"}
    first = policy.infer(req, "owner")
    secret = first["agent"]["episode_token"]
    assert len(secret) >= 32
    for overrides in ({}, {"episode_token": "x" * 43}, {"embodiment": "yam"}):
        # A different valid card must not replace the protected session either.
        if overrides.get("embodiment"):
            overrides = {"embodiment": {**policy.sessions["e:shared-name"].card.raw, "name": "changed"}}
        with pytest.raises(PermissionError):
            policy.infer({**req, **overrides}, "stranger")
    policy.close_connection("owner")
    resumed = policy.infer({**req, "episode_token": secret}, "reconnected")
    assert resumed["agent"]["steps"] == 2
    policy.sessions["e:shared-name"].touched -= module.IDLE_S + 1
    policy._expire()
    assert not policy.sessions and not policy.access
