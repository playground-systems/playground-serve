"""Cosmos 3 policy family without a GPU: the protocol with a stand-in for the official RobolabPolicyService (its
_next_seed / _build_sample / _generate / _format_outputs, cosmos_framework/scripts/action_policy_server_robolab.py at
the pinned commit), the seed hook, the entries, the fast path's decision and self-test, and the bring-up bundle."""

from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import threading

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[2] / "src" / "playground_serve"
for _p in (PKG / "families" / "common", PKG / "worker"):  # the worker's /rs: pgs_server, msgpack_numpy, server_decode
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.common.proposed_spec import POOL_PRICES  # noqa: E402
from playground_serve.families.cosmos3_policy import bringup, fast, serve, spec  # noqa: E402

EDGE, NANO = "cosmos3_edge_droid", "cosmos3_nano_droid"


class FakeService:
    """The parts of RobolabPolicyService that serve.Policy calls, with the official behaviour that matters here: an
    RNG seeded with 0 that advances per request, a lock, a sample check like _build_sample, a reply like
    _format_outputs ({"action": [32, 8]}, the gripper column inverted)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)
        self.seeds, self.samples = [], []

    def _next_seed(self):
        return int(self._rng.integers(0, 2**31))

    def _build_sample(self, obs):
        if not isinstance(obs.get("prompt"), str):
            raise ValueError("'prompt' must be a string")
        if "observation/image" not in obs and not all(key in obs for key in serve.IMAGE_KEYS):
            raise ValueError("Observation must contain 'observation/image' or RoBoArena wrist/exterior image keys")
        assert "rs_seed" not in obs
        self.samples.append(obs)
        return {"sample": len(self.samples)}

    def _generate(self, sample, seed):
        self.seeds.append(seed)
        return {"action": [np.full((33, 8), seed % 100, np.float32)]}

    def _format_outputs(self, obs, samples, *, start_time):
        action = samples["action"][0][1:].copy()
        action[:, -1] = 1.0 - action[:, -1]
        return {"action": action}


@contextlib.contextmanager
def serving(policy):
    import pgs_server

    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


def test_entries():
    edge, nano = spec.SPECS[EDGE], spec.SPECS[NANO]
    assert edge.family == nano.family == "cosmos3_policy" and edge.action_shape == nano.action_shape == (32, 8)
    assert edge.checkpoint == "hf://nvidia/Cosmos3-Edge-Policy-DROID@a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14"
    assert nano.checkpoint == "hf://nvidia/Cosmos3-Nano-Policy-DROID@5e9dd43c3c99d813c512960594bfd7b40678b180"
    assert all(POOL_PRICES[p] <= 2.0 for p in edge.gpu_pools)  # Edge runs under the 2 USD/hr cap
    assert all(p.endswith("_80") or p.endswith("_80_PRO") for p in nano.gpu_pools)  # Nano: 80 GB pools only
    assert spec.HOSTED_IDS == {EDGE: "cosmos3-edge-policy-droid", NANO: "cosmos3-nano-policy-droid"}
    assert edge.min_cuda == nano.min_cuda == "13.0" and "OpenMDW" in serve.LICENSE and not nano.gpu_validated
    assert edge.hosted_id == "cosmos3-edge-policy-droid" and edge.license == serve.LICENSE


def test_server_args_follow_the_docs_launch_lines():
    assert serve.server_args(EDGE) == {"checkpoint_path": "nvidia/Cosmos3-Edge-Policy-DROID",
                                       "hf_revision": "a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14",
                                       "format_prompt_as_json": True, "guidance_interval": (960.0, 1001.0)}
    assert serve.server_args(NANO) == {"checkpoint_path": "nvidia/Cosmos3-Nano-Policy-DROID",
                                       "hf_revision": "5e9dd43c3c99d813c512960594bfd7b40678b180"}


def test_over_the_wire_raw_and_jpeg():
    service = FakeService()
    obs = spec.example_observation(EDGE)
    with serving(serve.Policy(EDGE, service, "stock", "test")) as url:
        for mode in ("raw", None):  # None: the metadata's default, jpeg at image_size [360, 640]
            client = CLIENT.connect(url, mode=mode, wait=10, keepalive=None)
            try:
                meta = client.metadata
                assert meta["action_shape"] == [32, 8] and meta["image_size"] == [360, 640]
                assert meta["path"] == "stock" and meta["keys"] == list(serve.IMAGE_KEYS + serve.STATE_KEYS)
                assert client.mode == (mode or "jpeg") and tuple(client.size) == (360, 640)
                actions = np.asarray(client.act(obs))
            finally:
                client.close()
            assert actions.shape == (32, 8) and actions.dtype == np.float32
    raw, jpeg = service.samples
    for key in serve.IMAGE_KEYS:
        assert np.array_equal(raw[key], obs[key])  # raw: bytes unchanged
        assert jpeg[key].shape == (360, 640, 3) and jpeg[key].dtype == np.uint8  # jpeg: same size, no letterbox
    want = np.random.default_rng(0)
    assert service.seeds == [int(want.integers(0, 2**31)), int(want.integers(0, 2**31))]  # the service's own RNG


def test_rs_seed_fixes_the_seed_and_leaves_the_rng():
    service = FakeService()
    policy = serve.Policy(EDGE, service)
    a = policy.infer({**spec.example_observation(EDGE), "rs_seed": 7})["actions"]
    b = policy.infer(spec.example_observation(EDGE))["actions"]
    assert service.seeds == [7, int(np.random.default_rng(0).integers(0, 2**31))]
    assert np.all(a[:, :-1] == 7) and np.all(a[:, -1] == -6)  # 1 - 7: the gripper inversion of _format_outputs
    assert b.shape == (32, 8)


def test_service_error_reaches_the_client():
    obs = {k: v for k, v in spec.example_observation(NANO).items() if k != "prompt"}
    with serving(serve.Policy(NANO, FakeService())) as url:
        client = CLIENT.connect(url, mode="raw", wait=10, keepalive=None)
        try:
            with pytest.raises(CLIENT.ServerError, match="'prompt' must be a string"):
                client.act(obs)
        finally:
            client.close()


def test_warm_up_input_is_a_valid_request():
    service = FakeService()
    out = serve.Policy(EDGE, service).infer(serve.make_example())
    assert out["actions"].shape == (32, 8) and service.seeds == [0]


def test_recorded_observations_decode():
    for obs in spec.test_observations(EDGE):
        assert set(obs) == set(serve.IMAGE_KEYS + serve.STATE_KEYS) | {"prompt"}
        assert all(obs[k].shape == (180, 320, 3) and obs[k].dtype == np.uint8 for k in serve.IMAGE_KEYS)
        assert obs["observation/joint_position"].shape == (7,) and obs["prompt"] == "Put the blue block in the green bowl"


def test_selftest_reference_matches_the_recorded_observation():
    ref = fast.load_reference(EDGE)
    recorded = json.loads((spec.DROID_OBS).read_text())
    assert ref["observation"] == recorded["observations"][0] and isinstance(ref["seed"], int)
    assert set(fast.FAST_GPUS[EDGE]) <= set(ref["references"]) == {"NVIDIA A40", "NVIDIA H100 80GB HBM3",
                                                                    "NVIDIA GeForce RTX 5090"}
    for gpu, r in ref["references"].items():
        assert np.asarray(r["actions"]).shape == (32, 8) and np.isfinite(r["actions"]).all(), gpu


def test_fast_decide(monkeypatch):
    assert fast.decide("off", EDGE, "NVIDIA A40") == (False, "fast path turned off (RS_FAST=off)")
    assert all(fast.decide("auto", EDGE, g)[0] for g in ("NVIDIA A40", "NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 5090"))
    monkeypatch.setitem(fast.FAST_GPUS, EDGE, ("NVIDIA A40",))
    assert not fast.decide("auto", EDGE, "NVIDIA H100 80GB HBM3")[0]
    assert not fast.decide("auto", EDGE, "NVIDIA GeForce RTX 4090")[0]
    assert not fast.decide("auto", NANO, "NVIDIA A40")[0]


def test_fast_check_and_selftest():
    ref = {"observation": json.loads((spec.DROID_OBS).read_text())
           ["observations"][0], "seed": 7, "references": {"G": {"actions": np.zeros((32, 8)).tolist()}}}

    class P:
        def __init__(self, delta):
            self.delta, self.seen = delta, []

        def infer(self, obs):
            self.seen.append(obs["rs_seed"])
            return {"actions": np.full((32, 8), self.delta, np.float32)}

    ok, detail = fast.selftest(P(0.001), ref, "G")
    assert ok and detail["max_abs_diff"] == 0.001
    p = P(0.2)
    assert fast.selftest(p, ref, "G") == (False, {"mean_abs_diff": 0.2, "max_abs_diff": 0.2}) and p.seen == [7]
    assert fast.check(np.full((32, 8), np.nan), np.zeros((32, 8)))[1] == {"error": "non-finite actions"}
    assert not fast.check(np.zeros((16, 8)), np.zeros((32, 8)))[0]


def test_bringup_bundle_fits_the_env_limit():
    env = bringup.endpoint_env(EDGE, "test", {})
    files = bringup.bundle.unpack(env["RS_FILES"])
    assert {"boot.sh", "build_lib.sh", "pgs_server.py", "health.py", "server_decode.py",
            "stage.py", "serve.py", "build_env.sh", "fast.py", f"selftest_{EDGE}.json"} <= set(files)
    assert "pgs_msgpack_numpy.py" in files and "bringup.py" not in files and "verify.py" not in files
    assert "verify.py" in bringup.bundle.worker_files("cosmos3_policy", verify=True)  # bring-up runs (RS_VERIFY=1) only
    assert env["RS_ENV_TAR"] == "/runpod-volume/pgs/cosmos3_policy/env-secure-v2.tar" and env["RS_FAST"] == "auto"
    assert sum(len(k) + len(v) for k, v in env.items()) < bringup.MAX_ENV_TOTAL  # Runpod's request limit


def test_zero_tail_resize_and_pad_are_bit_identical():
    """The fast path's first-frame-only resize and pad give torchvision's tensors for clips with zero later frames
    (runs where torch and torchvision are installed)."""
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchvision.transforms.functional")
    zt = fast.ZeroTailFunctional(F)
    rng = np.random.default_rng(0)
    for h, w, size, pad in ((270, 320, [436, 517], [0, 0, 96, 4]), (540, 640, [540, 640], [0, 0, 96, 4]),
                            (180, 320, [90, 160], [0, 0, 200, 100])):
        clip = torch.zeros((3, 33, h, w), dtype=torch.uint8)
        clip[:, 0] = torch.from_numpy(rng.integers(0, 256, (3, h, w), dtype=np.uint8))
        a = F.resize(clip, size=size, interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        b = zt.resize(clip, size=size, interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        assert a.dtype == b.dtype and torch.equal(a, b)
        mode = "edge" if pad[2] >= size[1] or pad[3] >= size[0] else "reflect"
        assert torch.equal(F.pad(a, pad, padding_mode=mode), zt.pad(b, pad, padding_mode=mode))
    clip[:, 5, 0, 0] = 1  # a later frame that is not zero: torchvision's own function
    assert torch.equal(zt.pad(clip, [0, 0, 4, 4], padding_mode="reflect"), F.pad(clip, [0, 0, 4, 4], padding_mode="reflect"))


def test_same_sample_compares_tensors_exactly():
    torch = pytest.importorskip("torch")
    a = {"video": torch.zeros(3, 2), "plan": [1, {"x": torch.ones(2)}], "caption": "c"}
    b = {"video": torch.zeros(3, 2), "plan": [1, {"x": torch.ones(2)}], "caption": "c"}
    assert fast.same_sample(a, b)
    b["plan"][1]["x"][0] = 1.0000001
    assert not fast.same_sample(a, b)


def test_entries_are_experimental():
    from playground_serve import models

    assert spec.EXPERIMENTAL == {EDGE: True, NANO: True} and models.get(EDGE).experimental and models.get(NANO).experimental
    assert serve.Policy(EDGE, FakeService()).metadata["experimental"] is True


def test_torch_threads_default_to_16_and_never_exceed_torch_default(monkeypatch):
    class FakeTorch:
        def __init__(self, n):
            self.n = n

        def get_num_threads(self):
            return self.n

        def set_num_threads(self, n):
            self.n = n

    for env, start, want in ((None, 112, 16), (None, 8, 8), ("32", 112, 32), ("0", 112, 112)):
        fake = FakeTorch(start)
        monkeypatch.setitem(sys.modules, "torch", fake)
        if env is None:
            monkeypatch.delenv("RS_TORCH_THREADS", raising=False)
        else:
            monkeypatch.setenv("RS_TORCH_THREADS", env)
        logged = {}
        serve.set_threads(lambda tag, **kw: logged.update(kw))
        assert fake.n == want and logged["torch_threads"] == want
