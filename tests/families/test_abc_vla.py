"""abc_vla family without a GPU: spec against serve.py, the wire protocol with a stand-in for the repo's policy, the
path decision, the boot self-test and its fallback, the checkpoint download, and the bring-up bundle. The worker
files are loaded as the worker has them (flat in /rs): families/common and worker/ on sys.path, serve.py by file."""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import http.server
import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "playground_serve"
FAMILY = PKG / "families" / "abc_vla"
for p in (PKG / "families" / "common", PKG / "worker"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import pgs_server  # noqa: E402

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.abc_vla import bringup, spec  # noqa: E402
from playground_serve.families.common.proposed_spec import POOL_PRICES  # noqa: E402


def load_file(name: str, path: pathlib.Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


serve = load_file("abc_vla_serve", FAMILY / "serve.py")
NAME = "abc_vla_yam"


def infer_observation(policy, obs: dict) -> dict:
    """deploy/websocket_server.py:20-32 at the pinned commit, without its cv2 JPEG-bytes decode."""
    action_prefix = obs.pop("action_prefix", None)
    prefix_length = obs.pop("prefix_length", None)
    noise = obs.pop("noise", None)
    return policy.infer(obs, noise=noise, action_prefix=action_prefix, prefix_length=prefix_length)


class FakeDeployPolicy:
    """DeployPolicy.infer's signature (deploy/policy/base.py:48-70); returns (chunk_len, action_dim) like it."""

    chunk_len, action_dim = 30, 14

    def __init__(self, offset: float = 0.0):
        self.calls, self.offset = [], offset

    def infer(self, obs, *, noise=None, action_prefix=None, prefix_length=None):
        self.calls.append(dict(obs=obs, noise=noise, action_prefix=action_prefix, prefix_length=prefix_length))
        base = float(np.asarray(obs["state"]).sum())
        return {"actions": np.full((self.chunk_len, self.action_dim), base + self.offset, np.float32)}


@contextlib.contextmanager
def serving(policy):
    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


@pytest.fixture
def connect():
    opened = []

    def _connect(url, **kw):
        policy = CLIENT.connect(url, **{"wait": 10, "keepalive": None, **kw})
        opened.append(policy)
        return policy

    yield _connect
    for policy in opened:
        policy.close()


# ----- spec and build
def test_spec_matches_serve():
    assert set(spec.SPECS) == set(serve.VARIANTS) == {NAME}
    for name, s in spec.SPECS.items():
        v = serve.VARIANTS[name]
        assert s.name == name and s.family == "abc_vla" and s.robot == "yam" and s.robot in name
        assert s.checkpoint == v["uri"] == spec.CHECKPOINT
        assert s.action_shape == (v["horizon"], v["state_dim"]) == (spec.HORIZON, spec.STATE_DIM)
        assert tuple(v["cameras"]) == spec.CAMERAS and v["hz"] == spec.HZ
        assert s.gpu_pools and set(s.gpu_pools) <= set(POOL_PRICES) and s.gpu_pools[0] == "ADA_24"
        assert s.volume_gb >= 20 and s.min_container_ram_gb >= 16 and not s.secrets
        env = s.endpoint_env()
        assert env["RS_ENV_TAR"] == "/runpod-volume/pgs/abc_vla/env-secure-v2.tar" and env["RS_MODEL"] == name
    assert serve.LICENSE == spec.LICENSE and serve.COMMIT == spec.COMMIT
    obs = spec.example_observation(NAME)
    assert set(obs["images"]) == set(spec.CAMERAS) and obs["state"].shape == (spec.STATE_DIM,)


def test_build_env_pins_the_commit():
    script = FAMILY / "build_env.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    line = next(line for line in text.splitlines() if line.startswith("start_build "))
    url, commit = line.split()[1:3]
    assert url == "https://github.com/amazon-far/abc.git" and commit == serve.COMMIT and len(commit) == 40
    assert "source /rs/build_lib.sh" in text and "finish_build" in text


# ----- protocol
def test_over_the_wire_jpeg(connect):
    fake = FakeDeployPolicy()
    policy = serve.Policy(NAME, fake, infer_observation, "stock", "test")
    obs = spec.example_observation(NAME)
    for i, cam in enumerate(spec.CAMERAS):  # one flat colour per camera, to check the mapping
        obs["images"][cam] = np.full((480, 640, 3), 50 * (i + 1), np.uint8)
    with serving(policy) as url:
        client = connect(url)
        assert client.mode == "jpeg" and tuple(client.size) == (224, 224)
        meta = client.metadata
        assert meta["action_shape"] == [30, 14] and meta["action_hz"] == 30 and meta["path"] == "stock"
        assert meta["license"] == spec.LICENSE
        actions = client.act(obs)
    assert actions.shape == (30, 14) and actions.dtype == np.float32
    call = fake.calls[0]
    for i, cam in enumerate(spec.CAMERAS):
        image = call["obs"]["images"][cam]
        assert image.shape == (3, 224, 224) and image.dtype == np.uint8
        assert abs(int(image[0, 112, 112]) - 50 * (i + 1)) <= 3
    assert call["obs"]["state"].shape == (14,) and call["obs"]["prompt"] == "throw plastic bottles in bin"
    assert call["noise"] is None and call["action_prefix"] is None


def test_raw_channels_first_rtc_prefix_and_noise_pass_through(connect):
    fake = FakeDeployPolicy()
    policy = serve.Policy(NAME, fake, infer_observation)
    images = {cam: np.zeros((3, 168, 224), np.uint8) for cam in spec.CAMERAS}
    prefix, noise = np.ones((4, 14), np.float32), np.full((30, 14), 0.5, np.float32)
    obs = {"images": images, "state": np.zeros(14, np.float32), "prompt": "p", "action_prefix": prefix,
           "prefix_length": 4, "noise": noise}
    with serving(policy) as url:
        connect(url, mode="raw").act(obs)
    call = fake.calls[0]
    assert call["obs"]["images"]["top"].shape == (3, 168, 224)
    assert np.array_equal(call["action_prefix"], prefix) and call["prefix_length"] == 4
    assert np.array_equal(call["noise"], noise) and "action_prefix" not in call["obs"]


def test_missing_camera_and_non_finite_actions_are_errors(connect):
    policy = serve.Policy(NAME, FakeDeployPolicy(), infer_observation)
    obs = spec.example_observation(NAME)
    del obs["images"]["right"]
    with serving(policy) as url, pytest.raises(Exception, match="missing"):
        connect(url).act(obs)
    nan = serve.Policy(NAME, FakeDeployPolicy(offset=float("nan")), infer_observation)
    with pytest.raises(ValueError, match="non-finite"):
        nan.infer(spec.example_observation(NAME))


def test_channels_first():
    assert serve.channels_first(np.zeros((480, 640, 3))).shape == (3, 480, 640)
    assert serve.channels_first(np.zeros((3, 168, 224))).shape == (3, 168, 224)
    assert serve.channels_first(b"\xff\xd8jpeg") == b"\xff\xd8jpeg"


# ----- path decision, self-test and fallback
GPU = "NVIDIA GeForce RTX 4090"


def test_decide(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setenv("RS_FAST", "auto")
    assert serve.decide(NAME, GPU)[0] == "fast"
    assert serve.decide(NAME, "NVIDIA A40") == ("stock", "fast path not measured on NVIDIA A40, only on "
                                                         "NVIDIA GeForce RTX 4090, NVIDIA H100 80GB HBM3")
    assert serve.decide(NAME, "NVIDIA H100 80GB HBM3")[0] == "fast"
    assert serve.decide("other", GPU)[0] == "stock"
    monkeypatch.setenv("RS_FAST", "off")
    assert serve.decide(NAME, GPU) == ("stock", "fast path turned off (RS_FAST=off)")
    monkeypatch.setenv("RS_FAST", "auto")
    (tmp_path / "failed").write_text("crashed")
    path, reason = serve.decide(NAME, GPU)
    assert path == "stock" and "crashed" in reason


def test_selftest_reference_file():
    cases, bounds = serve.selftest_cases(NAME)
    assert {c["name"] for c in cases} >= {"zero_noise", "rtc_prefix_4"}
    assert set(bounds) == {"fast_max", "fast_mean", "stored_max"}
    for case in cases:
        assert case["actions"].shape == (30, 14) and np.isfinite(case["actions"]).all()
        obs = case["obs"]
        assert {k: v.shape for k, v in obs["images"].items()} == {c: (224, 224, 3) for c in spec.CAMERAS}
        assert obs["state"].shape == (14,) and isinstance(obs["prompt"], str)
    ref = json.loads(serve.selftest_file(NAME).read_text())
    assert ref["gpu"] and ref["commit"] == serve.COMMIT


class Echo:
    """A Policy stand-in whose actions are the stored stock actions plus `offset`."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def infer(self, obs):
        cases, _ = serve.selftest_cases(NAME)
        for case in cases:
            if obs.get("prefix_length") == case["obs"].get("prefix_length"):
                return {"actions": (case["actions"] + self.offset).astype(np.float32)}
        raise AssertionError("unexpected observation")


def test_selftest_passes_on_equal_actions_and_fails_beyond_the_bounds():
    ok, detail = serve.selftest(NAME, Echo(), Echo())
    assert ok and all(d["passed"] and d["fast_vs_stock"]["max_abs"] == 0 for d in detail.values())
    ok, detail = serve.selftest(NAME, Echo(), Echo(offset=0.5))
    assert not ok and not any(d["passed"] for d in detail.values())
    ok, detail = serve.selftest(NAME, Echo(offset=0.5), Echo(offset=0.5))  # stock moved: a warning, fast still checked
    assert ok and all("warning" in d for d in detail.values())


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """load()'s collaborators replaced: the stock policy, the repo's server module, torch and abc_fast."""
    monkeypatch.setenv("RS_MODEL", NAME)
    monkeypatch.setenv("RS_FAST", "auto")
    monkeypatch.setenv("RS_CKPT_DIR", str(tmp_path / "w"))
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setattr(serve, "load_stock", lambda path: FakeDeployPolicy())
    monkeypatch.setattr(pgs_server, "gpu_info", lambda fields="": GPU)
    ws = types.ModuleType("deploy.websocket_server")
    ws.infer_observation = infer_observation
    monkeypatch.setitem(sys.modules, "deploy", types.ModuleType("deploy"))
    monkeypatch.setitem(sys.modules, "deploy.websocket_server", ws)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    stock_actions = {c["name"]: c["actions"] for c in serve.selftest_cases(NAME)[0]}

    def install(fast_cls):
        module = types.ModuleType("abc_fast")
        module.FastDeployPolicy = fast_cls
        monkeypatch.setitem(sys.modules, "abc_fast", module)

    class StoredStock(FakeDeployPolicy):  # the stock stand-in answers the stored actions for the stored cases
        def __init__(self, offset=0.0):
            super().__init__(offset)

        def infer(self, obs, *, noise=None, action_prefix=None, prefix_length=None):
            key = "rtc_prefix_4" if prefix_length else "zero_noise"
            return {"actions": (stock_actions[key] + self.offset).astype(np.float32)}

    monkeypatch.setattr(serve, "load_stock", lambda path: StoredStock())
    return types.SimpleNamespace(install=install, StoredStock=StoredStock, tmp=tmp_path)


def test_load_serves_fast_when_the_selftest_passes(worker_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RS_COMPILE_CACHE", str(tmp_path / "inductor"))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    built = []
    worker_env.install(lambda dp, compile_body: built.append(compile_body) or worker_env.StoredStock())
    policy = serve.load()
    assert policy.metadata["path"] == "fast" and not pathlib.Path(serve.FAST_FAILED).exists()
    assert policy.metadata["path_reason"].startswith("compiled graph body on NVIDIA GeForce RTX 4090; boot self-test "
                                                     "passed (max difference from stock 0")
    assert built == [True]
    assert serve.os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor" / "NVIDIA_GeForce_RTX_4090")


def test_load_falls_back_to_the_exact_graph_then_stock(worker_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RS_COMPILE_CACHE", str(tmp_path / "inductor"))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    worker_env.install(lambda dp, compile_body: worker_env.StoredStock(offset=0.3 if compile_body else 0.0))
    policy = serve.load()
    assert policy.metadata["path"] == "fast" and policy.metadata["path_reason"].startswith("CUDA graph on")

    worker_env.install(lambda dp, compile_body: worker_env.StoredStock(offset=0.3))
    policy = serve.load()
    assert policy.metadata["path"] == "stock" and "self-test failed" in policy.metadata["path_reason"]

    def broken(dp, compile_body):
        raise RuntimeError("graph capture failed")

    worker_env.install(broken)
    policy = serve.load()
    assert policy.metadata["path"] == "stock" and "graph capture failed" in policy.metadata["path_reason"]
    assert not pathlib.Path(serve.FAST_FAILED).exists()

    monkeypatch.setenv("RS_FAST_COMPILE", "0")  # the graph alone: the compiled variant is not tried
    built = []
    worker_env.install(lambda dp, compile_body: built.append(compile_body) or worker_env.StoredStock())
    assert serve.load().metadata["path"] == "fast" and built == [False]


# ----- weights
class Files(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def test_prepare_downloads_checks_sha256_and_marks(monkeypatch, tmp_path):
    import stage

    payload = np.random.default_rng(0).bytes(3 << 20)
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "200000.pt").write_bytes(payload)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                            functools.partial(Files, directory=str(tmp_path / "srv")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/200000.pt"
    try:
        variant = dict(serve.VARIANTS[NAME], uri=url, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        monkeypatch.setitem(serve.VARIANTS, NAME, variant)
        monkeypatch.setenv("RS_MODEL", NAME)
        dest = tmp_path / "w"
        monkeypatch.setenv("RS_CKPT_DIR", str(dest))
        dest.mkdir()
        (dest / ".rs_failed").write_text("cannot download https://...: not gs://")  # what boot.sh's fetch leaves
        path = serve.prepare()
        assert path.read_bytes() == payload and stage.complete(dest) and not (dest / ".rs_failed").exists()
        assert serve.prepare() == path  # complete: no second download
        monkeypatch.setitem(serve.VARIANTS, NAME, dict(variant, sha256="0" * 64))
        (dest / stage.MARKER).unlink()
        with pytest.raises(RuntimeError, match="sha256"):
            serve.prepare()
        assert not stage.complete(dest)
    finally:
        httpd.shutdown()


# ----- bring-up bundle and dev hook
def test_bringup_bundle():
    files = bringup.family_files("abc_vla")
    assert {"serve.py", "abc_fast.py", "build_env.sh", f"selftest_{NAME}.json", "boot.sh", "build_lib.sh",
            "pgs_server.py", "pgs_msgpack_numpy.py", "health.py", "server_decode.py", "stage.py"} <= set(files)
    assert not {"spec.py", "NOTES.md", "bringup.py", "devhook.py"} & set(files)
    dev = bringup.family_files("abc_vla", dev=True)
    assert dev["serve.py"] == (FAMILY / "devhook.py").read_bytes() and dev["family_serve.py"] == files["serve.py"]
    env = bringup.endpoint_env("abc_vla", NAME, False, {"RS_FAST": "off"})
    assert env["RS_FAMILY"] == "abc_vla" and env["RS_FAST"] == "off" and env["RS_CHECKPOINT"] == spec.CHECKPOINT
    assert bringup.bundle.unpack(env["RS_FILES"]) == files
    assert len(env["RS_FILES"]) < 100_000


def test_devhook_runs_code_and_passes_requests_through(monkeypatch):
    monkeypatch.setitem(sys.modules, "family_serve", serve)
    devhook = load_file("abc_vla_devhook", FAMILY / "devhook.py")
    dev = devhook.Dev(serve.Policy(NAME, FakeDeployPolicy(), infer_observation))
    assert dev.metadata["dev"] is True
    out = dev.infer({"rs_dev_exec": "print(policy.name, 1 + 1)"})
    assert out["out"].strip() == f"{NAME} 2"
    assert "ZeroDivisionError" in dev.infer({"rs_dev_exec": "1 / 0"})["out"]
    assert dev.infer(spec.example_observation(NAME))["actions"].shape == (30, 14)


def test_selftest_images_are_client_jpegs():
    ref = json.loads(serve.selftest_file(NAME).read_text())
    for b in ref["images_jpeg"].values():
        assert base64.b64decode(b)[:2] == b"\xff\xd8"
