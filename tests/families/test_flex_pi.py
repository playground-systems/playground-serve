"""Flex-pi family without a GPU: a stand-in for the model and for its tensor steps checks the protocol, the request
keys, the exact infer_action call (signature of src/flexpi/models/flexpi.py infer_action at the pinned commit) with the
LIBERO evaluator's values, the gripper post-processing, the fast path's decision and self-test, and the bundle."""

from __future__ import annotations

import contextlib
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[2] / "src" / "playground_serve"
for _p in (PKG / "families" / "common", PKG / "worker"):  # the worker's /rs: pgs_server, msgpack_numpy, server_decode
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.common.proposed_spec import POOL_PRICES  # noqa: E402
from playground_serve.families.cosmos3_policy import bringup  # noqa: E402
from playground_serve.families.flex_pi import fast, serve, spec  # noqa: E402

NAME = "flex_pi_libero"


class FakeFlexPi:
    """FlexPi.infer_action with the real signature; returns {"action": [T, 32]} like the model (normalized space)."""

    def __init__(self):
        self.calls = []

    def infer_action(self, prompt=None, input_image=None, action_horizon=None, num_video_frames=None, proprio=None,
                     context=None, context_mask=None, negative_prompt=None, text_cfg_scale=1.0, num_inference_steps=20,
                     sigma_shift=None, seed=None, rand_device="cpu", tiled=False, camera_intrinsics=None, per_cam=None,
                     per_cam_depth=None, dynamic_step_skip=False, step_skip_thresholds=((0.95, 4), (0.93, 2)),
                     step_skip_sim_sources=None, step_skip_sim_aggregation="mean", return_stream_latents=False,
                     joint_video=None, joint_dino=None, joint_pointmap=None, present_video=None, present_dino=None,
                     present_pointmap=None):
        self.calls.append({k: v for k, v in locals().items() if k != "self"})
        return {"action": np.zeros((action_horizon, 32), np.float32)}


class FakeOps:
    """Records what the tensor steps receive; denormalize returns (1, 32, 7) with gripper values 0.8 / 0.2 / 0.5."""

    def __init__(self):
        self.seen = {}

    def inputs(self, image, wrist):
        self.seen["inputs"] = (image, wrist)
        return "COMPOSITE", {"cam_high": "H", "cam_left_wrist": "L", "cam_right_wrist": "R"}

    def proprio(self, state):
        self.seen["proprio"] = state
        return "PROPRIO"

    def depth(self, image_depth, wrist_depth):
        self.seen["depth"] = (image_depth, wrist_depth)
        return {"cam_high": "DH"}

    def denormalize(self, action):
        out = np.tile(np.arange(7, dtype=np.float32) / 10, (1, 32, 1))
        out[0, :, -1] = np.resize([0.8, 0.2, 0.5], 32)
        return out


@contextlib.contextmanager
def serving(policy):
    import pgs_server

    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


def test_entry():
    s = spec.SPECS[NAME]
    assert s.family == "flex_pi" and s.action_shape == (32, 7) and s.robot == "libero"
    assert s.checkpoint == "hf://flex-pi/flexpi-libero@f853cb49331aa0ab8124cbd1e1fb3a56e07a2523"
    assert s.env["RS_FLEXPI_JOINT"] == "false"  # the card's action-only regime by default
    assert spec.HOSTED_IDS[NAME] == s.hosted_id == "flexpi-libero" and "MIT" in s.license and s.license == serve.LICENSE
    assert s.gpu_pools[:2] == ("AMPERE_48", "ADA_32_PRO") and not s.experimental  # A40 first: the 5090 was in Europe
    assert "ADA_24" not in s.gpu_pools  # 26 GB > 24


def test_over_the_wire_without_depth():
    model, ops = FakeFlexPi(), FakeOps()
    obs = spec.example_observation(NAME)
    obs[serve.IMAGE][:], obs[serve.WRIST][:] = 60, 180  # flat colours, to check the order after JPEG
    with serving(serve.Policy(NAME, model, ops, joint=True)) as url:
        client = CLIENT.connect(url, wait=10, keepalive=None)
        try:
            assert client.metadata["action_shape"] == [32, 7] and client.metadata["image_size"] == [256, 256]
            assert client.mode == "jpeg" and client.metadata["regime"] == "full_joint"
            actions = np.asarray(client.act(obs))
        finally:
            client.close()
    assert actions.shape == (32, 7) and actions.dtype == np.float32
    assert np.allclose(actions[:, :6], np.arange(6) / 10)
    # Dataset gripper 1 = open, 0 = close -> LIBERO's -1 open / +1 close, binarized: 0.8 -> -1, 0.2 -> +1, 0.5 -> 0.
    assert actions[:3, -1].tolist() == [-1.0, 1.0, 0.0]
    image, wrist = ops.seen["inputs"]
    assert image.shape == wrist.shape == (256, 256, 3) and image.dtype == np.uint8
    assert abs(int(image[128, 128, 0]) - 60) <= 3 and abs(int(wrist[128, 128, 0]) - 180) <= 3
    call = model.calls[0]
    assert call["prompt"] == serve.DEFAULT_PROMPT.format(task=obs["prompt"])
    assert (call["input_image"], call["proprio"], call["per_cam"]["cam_high"]) == ("COMPOSITE", "PROPRIO", "H")
    # sim_libero.yaml and train.yaml values, as EVAL's _predict_action_chunk passes them
    assert call["num_inference_steps"] == 4 and call["seed"] == 42 and call["rand_device"] == "cpu"
    assert call["text_cfg_scale"] == 1.0 and call["negative_prompt"] == "" and call["sigma_shift"] is None
    assert call["action_horizon"] == 32 and call["num_video_frames"] == 9 and call["dynamic_step_skip"] is False
    assert call["present_pointmap"] is False and call["per_cam_depth"] is None
    assert call["joint_video"] is call["joint_dino"] is call["joint_pointmap"] is True


def test_depth_turns_the_pointmap_stream_on():
    model, ops = FakeFlexPi(), FakeOps()
    policy = serve.Policy(NAME, model, ops, joint=False)
    obs = spec.example_observation(NAME, depth=True)
    policy.infer(obs)
    call = model.calls[0]
    assert call["present_pointmap"] is True and call["per_cam_depth"] == {"cam_high": "DH"}
    assert call["joint_video"] is call["joint_dino"] is call["joint_pointmap"] is False
    assert ops.seen["depth"][0].dtype == np.uint16 and policy.metadata["regime"] == "action_only"
    one = {k: v for k, v in obs.items() if k != serve.DEPTHS[1]}
    with pytest.raises(ValueError, match="send both"):
        policy.infer(one)


def test_bad_inputs_are_refused():
    policy = serve.Policy(NAME, FakeFlexPi(), FakeOps())
    obs = spec.example_observation(NAME)
    with pytest.raises(ValueError, match="state"):
        policy.infer({**obs, serve.STATE: np.zeros(7, np.float32)})
    with pytest.raises(ValueError, match="uint8"):
        policy.infer({**obs, serve.IMAGE: np.zeros((256, 256, 3), np.float32)})


def test_recorded_observations_decode():
    with_depth, without = spec.test_observations(NAME)
    assert with_depth[serve.IMAGE].shape == (256, 256, 3) and with_depth[serve.STATE].shape == (8,)
    assert with_depth[serve.DEPTHS[0]].dtype == np.uint16 and with_depth[serve.DEPTHS[0]].shape == (128, 128)
    assert 500 < int(with_depth[serve.DEPTHS[0]].mean()) < 3000  # agentview depth in millimetres
    assert not set(serve.DEPTHS) & set(without)


def test_fast_decide_and_selftest():
    assert fast.decide("off", NAME, "NVIDIA A40") == (False, "fast path turned off (RS_FAST=off)")
    assert not fast.decide("auto", NAME, "NVIDIA GeForce RTX 4090")[0]
    obs = spec.test_observations(NAME)[0]
    ref = {"observation": __import__("json").loads(fast.reference_path(NAME).read_text())["observation"],
           "references": {"G": {"action_only": {"actions": np.zeros((32, 7)).tolist(),
                                                "actions_no_depth": np.ones((32, 7)).tolist()}}}}

    class P:
        def infer(self, o):
            return {"actions": np.zeros((32, 7)) if serve.DEPTHS[0] in o else np.ones((32, 7))}

    assert fast.selftest(P(), ref, "G")[0]
    ref["references"]["G"]["action_only"]["actions_no_depth"] = np.zeros((32, 7)).tolist()
    ok, detail = fast.selftest(P(), ref, "G")
    assert not ok and detail["depth"] is False and detail["max_abs_diff"] == 1.0
    assert obs[serve.IMAGE].dtype == np.uint8


def test_choose_falls_back_to_stock(monkeypatch):
    calls, logs = [], []
    monkeypatch.setitem(fast.FAST_GPUS, NAME, ("NVIDIA A40",))
    monkeypatch.setattr(fast, "decide", lambda mode, name, gpu, joint: (True, "test"))
    monkeypatch.setattr(fast, "selftest", lambda policy, ref, gpu, joint: (False, {"max_abs_diff": 1.0}))
    monkeypatch.setattr(fast, "reference_path", lambda name: pathlib.Path(spec.__file__).with_name(
        f"selftest_{NAME}.json"))
    policy = fast.choose(NAME, lambda path, reason: (path, reason), calls.append, serve.PATH_OPTIONS,
                         gpu="NVIDIA A40", mode="auto", log=lambda *a, **k: logs.append(a[0]))
    assert calls == [fast.FAST_OPTIONS, serve.PATH_OPTIONS["stock"]]
    assert policy[0] == "stock" and "self-test" in policy[1] and logs == ["path", "fast_selftest", "fallback"]


def test_finish_actions_matches_the_evaluator():
    a = np.zeros((2, 7), np.float32)
    a[:, -1] = [1.0, 0.0]  # dataset: 1 open, 0 close
    out = serve.finish_actions(a)
    assert out[:, -1].tolist() == [-1.0, 1.0]  # LIBERO: -1 open, +1 close


def test_bringup_bundle_fits_the_env_limit():
    env = bringup.endpoint_env(NAME, "test", {})
    files = bringup.bundle.unpack(env["RS_FILES"])
    assert {"serve.py", "build_env.sh", "fast.py", "prefetch.py", f"selftest_{NAME}.json"} <= set(files)
    assert "verify.py" not in files and "bringup.py" not in files  # verify* files ship only with RS_VERIFY=1
    assert "verify.py" in bringup.bundle.worker_files("flex_pi", verify=True)  # bring-up runs (RS_VERIFY=1) only
    assert env["DIFFSYNTH_MODEL_BASE_PATH"] == "/runpod-volume/diffsynth"
    assert sum(len(k) + len(v) for k, v in env.items()) < bringup.MAX_ENV_TOTAL


def test_selftest_reference_for_every_fast_gpu():
    ref = __import__("json").loads(fast.reference_path(NAME).read_text())
    assert set(ref["references"]) == set(fast.FAST_GPUS[NAME])
    for gpu, by_regime in ref["references"].items():
        for key, r in by_regime.items():
            assert key in ("action_only", "full_joint")
            assert np.asarray(r["actions"]).shape == np.asarray(r["actions_no_depth"]).shape == (32, 7), gpu
    assert fast.decide("auto", NAME, "NVIDIA A40", joint=True)[0]


def test_fast_path_encodes_each_prompt_once():
    class Model(FakeFlexPi):
        encoded = 0

        def encode_prompt(self, prompt):
            Model.encoded += 1
            return Arr(np.full((1, 4), 7.0)), Arr(np.ones((1, 4), bool))

    class Arr(np.ndarray):  # numpy stand-in for a tensor's clone()
        def __new__(cls, a):
            return np.asarray(a).view(cls)

        def clone(self):
            return self.copy()

    model = Model()
    policy = serve.Policy(NAME, model, FakeOps(), text_cache=True)
    obs = spec.example_observation(NAME)
    policy.infer(obs)
    policy.infer(obs)
    assert Model.encoded == 1 and all(c["prompt"] is None for c in model.calls)
    assert np.array_equal(model.calls[1]["context"], np.full((1, 4), 7.0))
    assert model.calls[0]["context"] is not model.calls[1]["context"]  # a copy per call
    stock = FakeFlexPi()
    serve.Policy(NAME, stock, FakeOps()).infer(obs)
    assert stock.calls[0]["prompt"] and stock.calls[0]["context"] is None  # stock: the prompt, as EVAL


def test_torch_threads_default_to_16_within_torch_default(monkeypatch):
    """torch's default is the container's CPUs where boot.sh sets OMP_NUM_THREADS (RUNPOD_CPU_COUNT), else the host's."""
    class FakeTorch:
        def __init__(self, n):
            self.n = n

        def get_num_threads(self):
            return self.n

        def set_num_threads(self, n):
            self.n = n

    for env, start, want in ((None, 120, 16), (None, 4, 4), (None, 8, 8), ("0", 112, 112), ("32", 112, 32)):
        if env is None:
            monkeypatch.delenv("RS_TORCH_THREADS", raising=False)
        else:
            monkeypatch.setenv("RS_TORCH_THREADS", env)
        fake, logged = FakeTorch(start), {}
        serve.set_threads(fake, lambda tag, **kw: logged.update(kw))
        assert fake.n == want and logged["torch_threads"] == want
