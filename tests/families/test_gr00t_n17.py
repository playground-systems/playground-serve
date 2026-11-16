"""gr00t_n17 family without a GPU: spec against serve.py, the wire protocol with a stand-in for Gr00tSimPolicyWrapper,
the checkpoint URI and file filter, the path decision, the boot self-test and its fallback, the bundle."""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "playground_serve"
FAMILY = PKG / "families" / "gr00t_n17"
for p in (PKG / "families" / "common", PKG / "worker"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import pgs_server  # noqa: E402

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.abc_vla import bringup  # noqa: E402
from playground_serve.families.common.proposed_spec import POOL_PRICES  # noqa: E402
from playground_serve.families.gr00t_n17 import spec  # noqa: E402


def load_file(name: str, path: pathlib.Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


serve = load_file("gr00t_n17_serve", FAMILY / "serve.py")


class FakeSimWrapper:
    """Gr00tSimPolicyWrapper.get_action's contract: batched flat keys in, ({"action.<k>": (B, 16, d)}, info) out."""

    def __init__(self):
        self.calls = []

    def get_action(self, batch):
        self.calls.append(batch)
        return {f"action.{k}": np.full((1, 16, d), i, np.float32)
                for i, (k, d) in enumerate(serve.ACTION_DIMS.items())}, {}


@contextlib.contextmanager
def serving(policy):
    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


def test_spec_matches_serve():
    assert set(spec.SPECS) == set(serve.VARIANTS)
    for name, s in spec.SPECS.items():
        assert s.family == "gr00t_n17" and s.robot == "libero" and s.robot in name
        assert s.checkpoint == serve.checkpoint_uri(name)
        assert serve.parse_checkpoint(s.checkpoint) == (serve.REPO, serve.REVISION, serve.VARIANTS[name]["suite"])
        assert s.action_shape == (serve.HORIZON, sum(serve.ACTION_DIMS.values()))
        assert s.secrets == ("HF_TOKEN",) and bool(s.gpu_validated) == (name == "gr00t_n17_libero")
        assert s.env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
        assert set(s.gpu_pools) <= set(POOL_PRICES) and s.volume_gb >= 20
        assert s.endpoint_env()["RS_ENV_TAR"] == "/runpod-volume/pgs/gr00t_n17/env.tar"
    assert (spec.COMMIT, spec.REPO, spec.REVISION, spec.LICENSE) == (serve.COMMIT, serve.REPO, serve.REVISION,
                                                                      serve.LICENSE)
    assert spec.STATE_DIMS == serve.STATE_DIMS and spec.PROMPT_KEY == serve.PROMPT_KEY
    assert spec.SUITES == {k: v["suite"] for k, v in serve.VARIANTS.items()}


def test_build_env_pins_the_commit():
    script = FAMILY / "build_env.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    line = next(line for line in script.read_text().splitlines() if line.startswith("start_build "))
    assert line.split()[1:3] == ["https://github.com/NVIDIA/Isaac-GR00T.git", serve.COMMIT]


def test_over_the_wire():
    fake = FakeSimWrapper()
    policy = serve.Policy("gr00t_n17_libero", fake)
    obs = spec.example_observation("gr00t_n17_libero")
    with serving(policy) as url:
        client = CLIENT.connect(url, wait=10, keepalive=None)
        try:
            assert client.mode == "jpeg" and tuple(client.size) == (256, 256)
            reply = client.infer(obs)
        finally:
            client.close()
    assert reply["actions"].shape == (16, 7) and reply["actions"].dtype == np.float32
    assert np.array_equal(reply["actions"][0], np.arange(7, dtype=np.float32))
    assert reply["action.gripper"].shape == (16, 1)
    batch = fake.calls[0]
    assert batch["video.image"].shape == (1, 1, 256, 256, 3) and batch["state.gripper"].shape == (1, 1, 2)
    assert batch[serve.PROMPT_KEY] == (obs[serve.PROMPT_KEY],)


def test_bad_state_width_is_an_error():
    obs = spec.example_observation("gr00t_n17_libero")
    obs["state.gripper"] = np.zeros(1, np.float32)
    with pytest.raises(ValueError, match="state.gripper"):
        serve.Policy("gr00t_n17_libero", FakeSimWrapper()).infer(obs)


def test_listing_keeps_the_inference_files(monkeypatch):
    names = ["config.json", "embodiment_id.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors",
             "model.safetensors.index.json", "processor_config.json", "statistics.json", "trainer_state.json",
             "rng_state_0.pth", "scheduler.pt", "zero_to_fp32.py"]
    entries = [{"type": "file", "path": f"libero_spatial/{n}", "size": i + 1} for i, n in enumerate(names)]
    entries.append({"type": "directory", "path": "libero_spatial/global_step20000", "size": 0})

    class Reply:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json

            return json.dumps(self.data).encode()

    seen = {}

    def urlopen(req, timeout):
        seen["url"], seen["headers"] = req.full_url, dict(req.header_items())
        return Reply(entries)

    monkeypatch.setattr(serve.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    files = serve.hf_listing(serve.REPO, serve.REVISION, "libero_spatial")
    assert sorted(f[0] for f in files) == sorted(names[:7])
    assert all(f[2].startswith(f"https://huggingface.co/{serve.REPO}/resolve/{serve.REVISION}/libero_spatial/")
               for f in files)
    assert seen["url"].endswith(f"/tree/{serve.REVISION}/libero_spatial")
    assert seen["headers"]["Authorization"] == "Bearer test-token"


GPU = "NVIDIA GeForce RTX 4090"
NAME = "gr00t_n17_libero"


def test_decide(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setenv("RS_FAST", "auto")
    assert serve.decide(NAME, GPU)[0] == "fast"
    assert serve.decide("gr00t_n17_libero_goal", GPU) == ("stock", "no fast path for gr00t_n17_libero_goal")
    assert serve.decide(NAME, "NVIDIA A40")[0] == "stock"
    monkeypatch.setenv("RS_FAST", "off")
    assert serve.decide(NAME, GPU)[0] == "stock"


def test_selftest_reference_file():
    cases, bounds = serve.selftest_cases(NAME)
    assert set(bounds) == {"fast_max", "fast_mean", "stored_max"} and len(cases) == 2
    for case in cases:
        assert case["actions"].shape == (16, 7) and np.isfinite(case["actions"]).all()
        obs = case["obs"]
        assert obs["video.image"].shape == (256, 256, 3) and obs["state.gripper"].shape == (2,)
        assert set(obs) >= set(serve.KEYS) and isinstance(obs["rs_seed"], int)
    ref = json.loads(serve.selftest_file(NAME).read_text())
    assert ref["commit"] == serve.COMMIT and ref["gpu"] == GPU
    assert all(base64.b64decode(b)[:2] == b"\xff\xd8" for b in ref["frames_jpeg"].values())


class StoredWrapper:
    """get_action answering the stored stock actions (plus `offset` while a FakeFast is active) per rs_seed."""

    def __init__(self, offset=0.0):
        self.offset, self.fast, self.seed = offset, None, None
        self.cases = {c["obs"]["rs_seed"]: c["actions"] for c in serve.selftest_cases(NAME)[0]}

    def get_action(self, batch):
        a = self.cases[self.seed] + (self.offset if self.fast is not None and self.fast.enabled else 0.0)
        return {f"action.{k}": a[None, :, i:i + 1].astype(np.float32) for i, k in enumerate(serve.ACTION_DIMS)}, {}


class FakeFast:
    def __init__(self):
        self.enabled = False

    @contextlib.contextmanager
    def active(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False


def test_selftest_and_load_paths(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda s: None
    torch.random = types.SimpleNamespace(fork_rng=lambda devices: contextlib.nullcontext())
    torch.cuda = types.SimpleNamespace(current_device=lambda: 0, empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("RS_MODEL", NAME)
    monkeypatch.setenv("RS_FAST", "auto")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setattr(pgs_server, "gpu_info", lambda fields="": GPU)
    original_infer = serve.Policy.infer

    def infer(self, obs):  # route the seed to the stand-in
        self.policy.seed = obs.get("rs_seed")
        return original_infer(self, obs)

    monkeypatch.setattr(serve.Policy, "infer", infer)
    for offset, path in ((0.0, "fast"), (0.5, "stock")):
        wrapper = StoredWrapper(offset)
        wrapper.policy = types.SimpleNamespace(model=types.SimpleNamespace(action_head=types.SimpleNamespace()))
        monkeypatch.setattr(serve, "load_stock", lambda p, w=wrapper: w)

        def fast_head(head, w=wrapper):
            w.fast = FakeFast()
            return w.fast

        monkeypatch.setitem(sys.modules, "gr00t_fast", types.SimpleNamespace(FastActionHead=fast_head))
        policy = serve.load()
        assert policy.metadata["path"] == path, policy.metadata["path_reason"]
        assert ("self-test passed" if path == "fast" else "self-test failed") in policy.metadata["path_reason"]
        assert not pathlib.Path(serve.FAST_FAILED).exists()


def test_bundle():
    files = bringup.family_files("gr00t_n17")
    assert {"serve.py", "gr00t_fast.py", "selftest_gr00t_n17_libero.json", "build_env.sh", "boot.sh", "pgs_server.py",
            "stage.py"} <= set(files)
    assert not {"spec.py", "NOTES.md"} & set(files)
    env = bringup.endpoint_env("gr00t_n17", "gr00t_n17_libero", False, {})
    assert env["RS_CHECKPOINT"] == spec.SPECS["gr00t_n17_libero"].checkpoint and "HF_TOKEN" not in env
