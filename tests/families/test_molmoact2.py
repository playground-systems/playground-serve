"""MolmoAct2 family without a GPU: the registry entries, the bundled files and env map, and serve.py's Policy over the
wire against a stand-in for model.predict_action with the official signature (the repo's servers' call)."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from playground_serve import bundle, models
from playground_serve.client import connect

PKG = pathlib.Path(models.__file__).resolve().parent
FAMILY = PKG / "families" / "molmoact2"
NAMES = ["molmoact2_libero", "molmoact2_droid", "molmoact2_so101", "molmoact2_yam"]


def load_serve():
    """families/molmoact2/serve.py as the worker imports it: its folder, families/common and worker/ on sys.path."""
    for d in (FAMILY, PKG / "families" / "common", PKG / "worker"):
        if str(d) not in sys.path:
            sys.path.append(str(d))
    spec = importlib.util.spec_from_file_location("molmoact2_serve", FAMILY / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load_serve()


def test_entries():
    assert models.names("molmoact2") == NAMES
    for name in NAMES:
        spec = models.get(name)
        v = serve.VARIANTS[name]
        assert spec.family == "molmoact2" and spec.openpi_config == ""
        assert spec.checkpoint == f"hf://{v['repo']}@{v['revision']}" and len(v["revision"]) == 40
        assert spec.action_shape == (v["horizon"], v["action_dim"]) and spec.robot == v["robot"]
        assert spec.gpu_pools == ["ADA_24", "AMPERE_48"] and spec.volume_gb == 50
        assert "Apache-2.0" in spec.license


def test_bundle():
    files = bundle.worker_files("molmoact2")
    assert {"serve.py", "build_env.sh", "molmoact2_variants.py", "boot.sh", "pgs_server.py", "stage.py",
            "health.py", "server_decode.py", "pgs_msgpack_numpy.py", "build_lib.sh"} <= set(files)
    assert not {"spec.py", "NOTES.md"} & set(files)
    env = bundle.endpoint_env(models.get("molmoact2_libero"), "r1")
    assert env["RS_ENV_TAR"] == "/runpod-volume/pgs/molmoact2/env-secure-v2.tar" and env["RS_FAMILY"] == "molmoact2"
    assert env["RS_CHECKPOINT"].startswith("hf://allenai/MolmoAct2-LIBERO@") and env["RS_FAST"] == "auto"
    shipped = bundle.unpack(env["RS_FILES"])
    assert shipped == bundle.worker_files("molmoact2", entry="molmoact2_libero")
    assert {n for n in files if n.startswith("selftest_")} - set(shipped) == {  # the other entries' references
        "selftest_molmoact2_droid.json", "selftest_molmoact2_so101.json", "selftest_molmoact2_yam.json"}
    for script in ("build_env.sh",):
        subprocess.run(["bash", "-n", str(FAMILY / script)], check=True)


def test_build_env_pins_the_variants_commit():
    from playground_serve.families.molmoact2.molmoact2_variants import COMMIT

    assert f"molmoact2.git {COMMIT}" in (FAMILY / "build_env.sh").read_text()


def predict_action(self, *, processor, images, task, state, norm_tag, inference_action_mode=None,
                   enable_depth_reasoning=False, enable_adaptive_depth=True, depth_cache=None, action_tokenizer=None,
                   num_steps=None, n_action_steps=None, generator=None, normalize_language=True,
                   enable_cuda_graph=True, return_dict=True):
    """The signature of MolmoAct2ForConditionalGeneration.predict_action (modeling_molmoact2.py at the pinned
    revisions); the body is a stand-in."""


class FakeModel:
    """Records the calls; actions encode the seed so the test can see it was applied before the call."""

    def __init__(self, name):
        self.calls, self.seed, self.v = [], None, serve.VARIANTS[name]

    def predict_action(self, **kw):
        inspect.signature(predict_action).bind(self, **kw)  # the official keywords only
        self.calls.append(kw)

        class Out:
            actions = np.full((1, self.v["horizon"], self.v["action_dim"]), float(self.seed or 0), np.float32)

        return Out()


@pytest.mark.parametrize("name", NAMES)
def test_over_the_wire(name):
    import pgs_server

    fake = FakeModel(name)
    policy = serve.Policy(name, lambda **kw: fake.predict_action(processor="P", **kw),
                          seed=lambda s: setattr(fake, "seed", s))
    obs = models.example_observation(name)
    cams = [f"{c}_cam" for c in serve.VARIANTS[name]["cameras"]]
    obs[cams[0]] = np.full((180, 320, 3), 200, np.uint8)  # non-square: must arrive at its own size
    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        with connect(f"ws://127.0.0.1:{server.port}/ws", wait=10, keepalive=None) as client:
            assert client.mode == "jpeg" and client.size is None
            meta = client.metadata
            assert meta["cameras"] == cams and meta["action_shape"] == list(models.get(name).action_shape)
            assert meta["playground_serve"]["path"] == "stock" and "release" in meta["playground_serve"]
            reply = client.infer({**obs, "rs_seed": 7})
            actions = np.asarray(reply["actions"])
    finally:
        server.stop()
    assert actions.shape == models.get(name).action_shape and float(actions[0, 0]) == 7.0
    assert set(reply["policy_timing"]) == {"preprocess_ms", "model_ms", "postprocess_ms"}
    assert reply["server_timing"]["decode_ms"] <= reply["server_timing"]["infer_ms"]
    call = fake.calls[0]
    assert [im.size for im in call["images"]] == [(320, 180)] + [(256, 256)] * (len(cams) - 1)
    assert abs(int(np.asarray(call["images"][0])[90, 160, 0]) - 200) <= 3  # JPEG q90
    assert call["task"] == "pick up the object" and call["state"].dtype == np.float32
    assert (call["norm_tag"], call["inference_action_mode"], call["num_steps"]) == (
        serve.VARIANTS[name]["norm_tag"], "continuous", 10)
    assert call["enable_depth_reasoning"] is False and call["enable_cuda_graph"] is False
    assert call["normalize_language"] is True


def test_fast_path_decision():
    gpu = "NVIDIA GeForce RTX 4090"
    assert serve.decide("auto", "molmoact2_libero", gpu, True)[0] == "fast"
    assert serve.decide("off", "molmoact2_libero", gpu, True) == (
        "stock", "fast path turned off (RS_FAST=off, `pgs up --stock-path`)")
    assert serve.decide("auto", "molmoact2_libero", "NVIDIA A40", True)[0] == "stock"
    assert serve.decide("measure", "molmoact2_libero", "NVIDIA A40", True)[0] == "fast"
    assert "no self-test reference" in serve.decide("auto", "molmoact2_yam", gpu, False)[1]


@pytest.mark.parametrize("name", NAMES)
def test_selftest_reference(name):
    """Each stored reference: the entry's cameras as JPEG, its state and chunk shape, made by the stock path."""
    import json

    path = serve.reference(name)
    if not path.is_file():
        pytest.skip(f"no reference for {name} yet")
    ref = json.loads(path.read_text())
    obs = serve.selftest_observation(ref)
    v = serve.VARIANTS[name]
    assert sorted(ref["images_jpeg"]) == sorted(f"{c}_cam" for c in v["cameras"])
    assert all(obs[f"{c}_cam"].dtype == np.uint8 and obs[f"{c}_cam"].ndim == 3 for c in v["cameras"])
    assert np.asarray(ref["actions"]).shape == (v["horizon"], v["action_dim"]) and len(ref["state"]) == v["state_dim"]
    assert ref["checkpoint"] == models.get(name).checkpoint and "stock path" in ref["made_with"]
    ok, detail = serve.check(ref["actions"], ref)
    assert ok and detail["mean_abs_diff"] == detail["max_abs_diff"] == 0.0
    assert 0 < detail["mean_bound"] < detail["max_bound"]
    assert serve.check(np.asarray(ref["noise_actions"]), ref)[0] is False  # a new noise draw fails
    off = np.asarray(ref["actions"]) + detail["max_bound"] * 1.5
    assert serve.check(off, ref)[0] is False
    assert serve.check(np.full_like(off, np.nan), ref)[1] == {"error": "non-finite actions"}


def test_bundle_fits_with_references():
    env = bundle.endpoint_env(models.get("molmoact2_yam"), "r1")
    assert len(env["RS_FILES"]) < bundle.MAX_ENV_STRING_BYTES - 20_000  # room left under Linux's limit


def test_bad_observation():
    policy = serve.Policy("molmoact2_droid", lambda **kw: None)
    with pytest.raises(KeyError, match="wrist_cam"):
        policy.infer({"external_cam": np.zeros((8, 8, 3), np.uint8), "state": np.zeros(8), "instruction": "x"})
    obs = models.example_observation("molmoact2_droid")
    with pytest.raises(ValueError, match="state must be shape"):
        policy.infer({**obs, "state": np.zeros(7)})
