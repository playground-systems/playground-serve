"""FLUX 3 Action family without a GPU (never run: gated weights): a stand-in for the official RoboLabPolicy checks the
protocol, the request keys, the reply rename, the load_serving_policy keywords (src/flux_action/serving/robolab.py and
cli.py at the pinned commit), the license marking and the missing-token refusal."""

from __future__ import annotations

import contextlib
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[2] / "src" / "playground_serve"
for _p in (PKG / "families" / "common", PKG / "worker"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.flux3_action import serve, spec  # noqa: E402

NAME = "flux3_action_droid"


class FakeRoboLabPolicy:
    """RoboLabPolicy.infer(obs): composes the views (wrist on top, the exteriors at half size below, 540 x 640) and
    replies {"action": float32 (32, 8)}."""

    def __init__(self):
        self.calls = []
        self.reply = np.linspace(-1, 1, 32 * 8, dtype=np.float32).reshape(32, 8)

    def infer(self, obs):
        wrist = np.asarray(obs["observation/wrist_image_left"])
        assert (wrist.shape[0] + wrist.shape[0] // 2, wrist.shape[1]) == (540, 640), "composite size"
        self.calls.append(obs)
        return {"action": self.reply.copy()}


@contextlib.contextmanager
def serving(policy):
    import pgs_server

    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


def test_entry_is_marked_non_commercial_and_gated():
    s = spec.SPECS[NAME]
    assert spec.NON_COMMERCIAL and "non-commercial" in serve.LICENSE and s.secrets == ("HF_TOKEN",)
    assert s.checkpoint == "hf://black-forest-labs/flux-3-action-droid@d84c80adc0721780245aabb75e45d4164ead7b97"
    assert s.action_shape == (32, 8) and s.gpu_validated == "2026-11-16" and s.env["RS_COMPILE_DIT"] == "0"


def test_over_the_wire_with_the_default_mode():
    adapter = FakeRoboLabPolicy()
    obs = spec.example_observation(NAME)
    with serving(serve.Policy(NAME, adapter)) as url:
        client = CLIENT.connect(url, wait=10, keepalive=None)
        try:
            assert client.metadata["image_size"] == [360, 640] and client.mode == "jpeg"
            assert "non-commercial" in client.metadata["license"]
            actions = np.asarray(client.act(obs))
        finally:
            client.close()
    assert np.array_equal(actions, adapter.reply)
    assert adapter.calls[0]["observation/exterior_image_1_left"].shape == (360, 640, 3)


def test_serving_kwargs_are_the_cli_defaults():
    assert serve.serving_kwargs(NAME, env={}) == dict(
        revision="d84c80adc0721780245aabb75e45d4164ead7b97", subfolder=None, device="cuda", settings={},
        dtype="bfloat16", compile_dit=True, offload_text_encoder=False, warmup=1)
    kw = serve.serving_kwargs(NAME, env={"RS_SUBFOLDER": "variants/gd", "RS_COMPILE_DIT": "0",
                                          "RS_OFFLOAD_TEXT_ENCODER": "1"})
    assert kw["subfolder"] == "variants/gd" and kw["compile_dit"] is False and kw["offload_text_encoder"] is True


def test_worker_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        serve.main()


def test_recorded_observations_are_served_at_360x640():
    obs = spec.test_observations(NAME)
    assert len(obs) == 2 and all(o[k].shape == (360, 640, 3) and o[k].dtype == np.uint8 for o in obs
                                 for k in serve.IMAGE_KEYS)
    assert serve.verify_observations()[0]["prompt"] == obs[0]["prompt"]  # verify_obs.json holds observation 0


def test_rs_seed_reaches_robolab_infer():
    seen = []

    class Adapter:
        def infer(self, obs, *, seed=None):
            seen.append((seed, "rs_seed" in obs))
            return {"action": np.zeros((32, 8), np.float32)}

    policy = serve.Policy(NAME, Adapter())
    policy.infer({**spec.example_observation(NAME), "rs_seed": 7})
    policy.infer(spec.example_observation(NAME))
    assert seen == [(7, False), (None, False)]


def test_bringup_never_prints_the_token(tmp_path, capsys):
    from playground_serve.families.cosmos3_policy import bringup

    token_file = tmp_path / "token"
    token_file.write_text("not-a-real-token-value\n")
    args = type("A", (), {"hf_token_file": str(token_file)})()
    env = bringup.endpoint_env(NAME, "test", bringup.secret_env(args))
    assert env["HF_TOKEN"] == "not-a-real-token-value"
    print(bringup.show(f"log line with not-a-real-token-value inside"))
    assert "not-a-real-token-value" not in capsys.readouterr().out
    bringup.SECRETS.clear()


def test_entry_is_experimental():
    from playground_serve import models

    assert spec.EXPERIMENTAL == {NAME: True} and spec.SPECS[NAME].experimental and models.get(NAME).experimental
    meta = serve.Policy(NAME, object()).metadata
    assert meta["experimental"] is True and meta["non_commercial"] is True
