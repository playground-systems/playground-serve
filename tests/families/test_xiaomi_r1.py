"""xiaomi_r1 family without a GPU: spec against serve.py, the history helpers against the evaluator's, the frame-stack
forms, the request the processor gets, the path decision, the boot self-test and its fallback, and the bundle."""

from __future__ import annotations

import base64
import collections
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
FAMILY = PKG / "families" / "xiaomi_r1"
for p in (PKG / "families" / "common", PKG / "worker"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import pgs_server  # noqa: E402

from playground_serve.client import client as CLIENT  # noqa: E402
from playground_serve.families.abc_vla import bringup  # noqa: E402
from playground_serve.families.common.proposed_spec import POOL_PRICES  # noqa: E402
from playground_serve.families.xiaomi_r1 import spec  # noqa: E402


def load_file(name: str, path: pathlib.Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


serve = load_file("xiaomi_r1_serve", FAMILY / "serve.py")
NAME = "xiaomi_r1_robocasa365"
GPU = "NVIDIA GeForce RTX 4090"


class FakeTensor:
    """The few torch.Tensor methods Policy uses, on a numpy array."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def is_floating_point(self):
        return self.a.dtype.kind == "f"

    def to(self, **kw):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self.a.astype(np.float32))

    def detach(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, idx):
        return FakeTensor(self.a[idx])


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def list_robot_types(self):
        return ["robocasa365"]

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"input_ids": FakeTensor(np.zeros((1, 476), np.int64)), "state": FakeTensor(kw["state"]),
                "action_mask": FakeTensor(np.ones((1, 16, 60), np.float32))}

    def decode_action(self, raw, robot_type):
        return FakeTensor(raw.a * 2.0)


class FakeModel:
    device, dtype = "cuda", "bfloat16"

    def __init__(self, offset=0.0):
        self.offset, self.calls = offset, []

    def __call__(self, **data):
        self.calls.append(data)
        state = data["state"].a
        return types.SimpleNamespace(actions=FakeTensor(np.full((1, 16, 60), state.sum() + self.offset, np.float32)))


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    seeds = []
    torch.manual_seed = seeds.append
    torch.random = types.SimpleNamespace(fork_rng=lambda devices, enabled: contextlib.nullcontext())
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return seeds


@contextlib.contextmanager
def serving(policy):
    server = pgs_server.Server(policy, host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


def test_spec_matches_serve():
    assert set(spec.SPECS) == set(serve.VARIANTS) == {NAME}
    s, v = spec.SPECS[NAME], serve.VARIANTS[NAME]
    assert s.family == "xiaomi_r1" and s.robot == "robocasa365" and s.robot in NAME
    assert s.checkpoint == f"hf://{v['repo']}@{v['revision']}" == f"hf://{spec.REPO}@{spec.REVISION}"
    assert s.action_shape == (v["horizon"], v["action_dim"]) == (spec.HORIZON, spec.ACTION_DIM)
    assert tuple(v["cameras"]) == spec.CAMERAS and tuple(v["image_size"]) == spec.IMAGE_SIZE
    assert (v["state_dim"], v["hz"], v["history"], v["interval"]) == (spec.STATE_DIM, spec.HZ, spec.HISTORY,
                                                                      spec.INTERVAL)
    # RTX 4090 first, then the RTX 5090 (Serverless stock only in Europe: more round trip for a US user than the 23 ms
    # it saves), then the A40; `pgs up --gpu` picks another
    assert set(s.gpu_pools) <= set(POOL_PRICES) and s.gpu_pools == ("ADA_24", "ADA_32_PRO", "AMPERE_48")
    assert not s.secrets
    assert s.endpoint_env()["RS_ENV_TAR"] == "/runpod-volume/pgs/xiaomi_r1/env-secure-v2.tar"
    assert serve.COMMIT == spec.COMMIT and serve.LICENSE == spec.LICENSE


def test_build_env_pins_the_commit():
    script = FAMILY / "build_env.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    line = next(line for line in script.read_text().splitlines() if line.startswith("start_build "))
    assert line.split()[1:3] == ["https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git", serve.COMMIT]


def test_sample_history_matches_the_evaluator():
    """entry.py:81-89 with a deque of maxlen 7, as the evaluator keeps it."""
    q = collections.deque(maxlen=7)
    for t in range(12):
        q.append(np.array([t]))
        got = spec.sample_history(q)[:, 0].tolist()
        want = [max(0, len(q) - 1 - (3 - i) * 2) for i in range(4)]
        assert got == [list(q)[i][0] for i in want]
        assert np.array_equal(spec.sample_history(q), serve.sample_history(q))
    assert spec.sample_history([np.array([5])])[:, 0].tolist() == [5, 5, 5, 5]


def test_frame_stack_forms():
    stack = np.random.default_rng(0).integers(0, 256, (4, 256, 256, 3), dtype=np.uint8)
    frames = (4, 256, 256, 3)
    assert np.array_equal(serve.frame_stack(stack, frames), stack)
    assert np.array_equal(serve.frame_stack(list(stack), frames), stack)
    assert np.array_equal(serve.frame_stack(spec.as_frames_map(stack), frames), stack)
    shuffled = {str(i): stack[i] for i in (3, 1, 0, 2)}
    assert np.array_equal(serve.frame_stack(shuffled, frames), stack)
    with pytest.raises(ValueError, match="uint8 frames"):
        serve.frame_stack(stack[:3], frames)


def test_request_is_the_evaluators(fake_torch):
    proc, model = FakeProcessor(), FakeModel()
    policy = serve.Policy(NAME, model, proc)
    obs = spec.example_observation(NAME)
    out = policy.infer(dict(obs, rs_seed=7))
    assert out["actions"].shape == (16, 12) and out["actions"].dtype == np.float32
    messages, kw = proc.calls[0]
    assert kw["state"].shape == (1, 4, 60) and np.array_equal(kw["state"][0, :, :14], obs["state"])
    assert not kw["state"][0, :, 14:].any() and kw["robot_type"] == "robocasa365" and kw["do_resize"] is False
    content = messages[0]["content"]
    assert [c["type"] for c in content] == ["text", "video", "text", "video", "text", "video", "text"]
    assert content[-1]["text"].endswith("Close the blender lid. /no_cot") and len(content[1]["video"]) == 4
    assert messages[1]["content"][0]["text"] == "<cot></cot>"
    assert model.calls[0]["task_id"] == "robocasa365" and fake_torch == [7]


def test_over_the_wire_jpeg_frames_map(fake_torch):
    proc, model = FakeProcessor(), FakeModel()
    policy = serve.Policy(NAME, model, proc)
    obs = spec.example_observation(NAME)
    sent = {**{k: spec.as_frames_map(obs[k]) for k in spec.CAMERAS}, "state": obs["state"],
            "instruction": obs["instruction"]}
    with serving(policy) as url:
        client = CLIENT.connect(url, wait=10, keepalive=None)
        try:
            assert client.mode == "jpeg" and tuple(client.size) == (256, 256)
            assert client.metadata["history"] == 4 and client.metadata["path"] == "stock"
            actions = client.act(sent)
        finally:
            client.close()
    assert actions.shape == (16, 12)
    video = proc.calls[0][0][0]["content"][1]["video"]
    assert video[0].size == (256, 256)


def test_errors(fake_torch):
    policy = serve.Policy(NAME, FakeModel(), FakeProcessor())
    obs = spec.example_observation(NAME)
    with pytest.raises(ValueError, match="state must be shape"):
        policy.infer(dict(obs, state=obs["state"][:3]))
    nan = serve.Policy(NAME, FakeModel(offset=float("nan")), FakeProcessor())
    with pytest.raises(ValueError, match="non-finite"):
        nan.infer(obs)


def test_decide(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setenv("RS_FAST", "auto")
    assert serve.decide(NAME, GPU)[0] == ("fast" if NAME in serve.FAST else "stock")
    assert serve.decide(NAME, "NVIDIA A40")[0] == "stock"
    monkeypatch.setenv("RS_FAST", "off")
    assert serve.decide(NAME, GPU) == ("stock", "fast path turned off (RS_FAST=off)")


def test_selftest_reference_file():
    cases, bounds = serve.selftest_cases(NAME)
    assert set(bounds) == {"fast_max", "fast_mean", "stored_max"} and len(cases) >= 2
    for case in cases:
        assert case["actions"].shape == (16, 12) and np.isfinite(case["actions"]).all()
        obs = case["obs"]
        assert all(obs[k].shape == (4, 256, 256, 3) for k in spec.CAMERAS)
        assert obs["state"].shape == (4, 14) and isinstance(obs["rs_seed"], int)
    ref = json.loads(serve.selftest_file(NAME).read_text())
    assert ref["commit"] == serve.COMMIT and ref["gpu"]
    assert all(base64.b64decode(b)[:2] == b"\xff\xd8" for b in ref["frames_jpeg"].values())


class Stored:
    """A Policy stand-in that answers each stored case's stock actions plus `offset`."""

    def __init__(self, offset=0.0):
        self.offset = offset
        self.cases = {c["obs"]["rs_seed"]: c["actions"] for c in serve.selftest_cases(NAME)[0]}

    def infer(self, obs):
        return {"actions": (self.cases[obs["rs_seed"]] + self.offset).astype(np.float32)}


def test_selftest_bounds():
    assert serve.selftest(NAME, Stored(), Stored())[0]
    assert not serve.selftest(NAME, Stored(), Stored(offset=0.5))[0]


def test_load_paths(monkeypatch, tmp_path, fake_torch):
    monkeypatch.setenv("RS_MODEL", NAME)
    monkeypatch.setenv("RS_FAST", "auto")
    monkeypatch.setattr(serve, "FAST_FAILED", str(tmp_path / "failed"))
    monkeypatch.setattr(serve, "FAST", {NAME: (GPU,)})
    monkeypatch.setattr(serve, "load_stock", lambda path: (FakeModel(), FakeProcessor()))
    monkeypatch.setattr(pgs_server, "gpu_info", lambda fields="": GPU)
    calls = {"n": 0}

    def make_run(offset):
        module = types.ModuleType("xr1_fast")

        class FastRun:
            def __init__(self, model, compile_dit=False):
                pass

            def __call__(self, data):
                calls["n"] += 1
                return types.SimpleNamespace(cpu=lambda: FakeTensor(np.zeros((1, 16, 60), np.float32) + offset))

        class FastRequest:
            def __init__(self, stock, serve_module):
                assert serve_module is serve and stock.metadata["path"] == "stock"
                self.stock = stock

            def __call__(self, obs):
                return self.stock.request(obs)

        module.FastForward, module.FastRequest = FastRun, FastRequest
        monkeypatch.setitem(sys.modules, "xr1_fast", module)

    stored = Stored()
    monkeypatch.setattr(serve.Policy, "infer", lambda self, obs: (
        {"actions": (stored.cases[obs["rs_seed"]] + (0.3 if self.metadata["path"] == "fast" and calls.get("bad")
                                                   else 0.0)).astype(np.float32)}))
    make_run(0.0)
    policy = serve.load()
    assert policy.metadata["path"] == "fast" and "self-test passed" in policy.metadata["path_reason"]
    calls["bad"] = True
    policy = serve.load()
    assert policy.metadata["path"] == "stock" and "self-test failed" in policy.metadata["path_reason"]
    assert not pathlib.Path(serve.FAST_FAILED).exists()


def test_bundle():
    files = bringup.family_files("xiaomi_r1")
    assert {"serve.py", "xr1_fast.py", "build_env.sh", f"selftest_{NAME}.json", "boot.sh", "stage.py"} <= set(files)
    assert not {"spec.py", "NOTES.md"} & set(files)
    env = bringup.endpoint_env("xiaomi_r1", NAME, False, {})
    assert len(env["RS_FILES"]) < 100_000 and env["TOKENIZERS_PARALLELISM"] == "false"
