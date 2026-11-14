"""Smoke test of the bundled serve.py on CPU: unpack RS_FILES, run serve.py with the endpoint env against a stand-in
for openpi's scripts/serve_policy.py (the real config's robot transforms, no model, no weights) and a fake
nvidia-smi, and read its JSON log up to the localhost self-test. Covers the env handling, the per-model and per-GPU
JAX cache dir, the warm-up input per robot family, the openpi websocket server with the no-op and the metadata.
Needs openpi importable (see test_models_openpi.py); skipped otherwise.
"""

import json
import os
import pathlib
import queue
import socket
import subprocess
import sys
import threading
import time

import pytest

from playground_serve import bundle, models

pytest.importorskip("openpi")

FAKE_SERVE_POLICY = '''"""Stand-in for openpi's scripts/serve_policy.py: the config's real robot transforms, zero actions, no model."""
import dataclasses
import json

import numpy as np

from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config


class _NoFastProcessor:  # FAST's Hugging Face processor runs remote code; inputs do not use it
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None


_tokenizer.AutoProcessor = _NoFastProcessor


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    port: int = 8000
    policy: Checkpoint = None


class _Composite:
    def __init__(self, transforms):
        self.transforms = transforms


class _Policy:
    def __init__(self, train_config, data_config):
        self._input_transform = _Composite(list(data_config.data_transforms.inputs))
        self._outputs = data_config.data_transforms.outputs
        self._shape = (train_config.model.action_horizon, train_config.model.action_dim)
        self.metadata = dict(train_config.policy_metadata or {})
        self._rng = None

    def infer(self, obs):
        for transform in self._input_transform.transforms:  # raises if obs is not this robot's observation
            obs = transform(obs)
        out = {"state": np.asarray(obs["state"]), "actions": np.zeros(self._shape, np.float32)}
        for transform in self._outputs:
            out = transform(out)
        return {"actions": out["actions"]}


def create_policy(args):
    print(json.dumps({"fake_create_policy": args.policy.config, "dir": args.policy.dir}), flush=True)
    train_config = _config.get_config(args.policy.config)
    return _Policy(train_config, train_config.data.create(train_config.assets_dirs, train_config.model))
'''
ACTIONS = {"pi05_libero": [10, 7], "pi05_droid": [15, 8], "pi0_aloha_towel": [50, 14]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("name", ACTIONS)
def test_serve_py_on_cpu(name, tmp_path):
    spec = models.get(name)
    env = bundle.endpoint_env(spec, "smoke")
    rs, checkout, bin_dir = tmp_path / "rs", tmp_path / "openpi", tmp_path / "bin"
    for d in (rs, checkout / "scripts", bin_dir):
        d.mkdir(parents=True)
    for file_name, data in bundle.unpack(env.pop("RS_FILES")).items():
        (rs / file_name).parent.mkdir(parents=True, exist_ok=True)
        (rs / file_name).write_bytes(data)
    (checkout / "scripts" / "serve_policy.py").write_text(FAKE_SERVE_POLICY)
    (bin_dir / "nvidia-smi").write_text('#!/bin/sh\necho "NVIDIA GeForce RTX 4090, GPU-test, 595.91.07, 24564 MiB"\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    port = _free_port()
    env.update(PORT=str(port), RS_JAX_CACHE_ROOT=str(tmp_path / "jax_cache"),
               OPENPI_DATA_HOME=os.environ.get("OPENPI_DATA_HOME") or str(tmp_path / "openpi_data"),
               PATH=f"{bin_dir}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin", HOME=str(tmp_path))
    had_ready = os.path.exists("/tmp/rs_ready")
    proc = subprocess.Popen([sys.executable, "-u", str(rs / "serve.py")], cwd=checkout, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    log, output, deadline = {}, [], time.time() + 180
    try:
        while "selftest_ws" not in log and time.time() < deadline and proc.poll() is None:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            output.append(line)
            if line.startswith("{"):
                record = json.loads(line)
                log[record.get("rs") or next(iter(record))] = record
    finally:
        proc.kill()
        proc.wait()
        if not had_ready and os.path.exists("/tmp/rs_ready"):
            os.remove("/tmp/rs_ready")
    assert "selftest_ws" in log, "".join(output[-40:])

    start = log["process_start"]
    assert (start["model"], start["config"], start["checkpoint"]) == (name, spec.openpi_config, spec.checkpoint)
    assert start["jax_cache"]["dir"] == str(tmp_path / "jax_cache" / name / "NVIDIA_GeForce_RTX_4090")
    assert log["fake_create_policy"] == {"fake_create_policy": spec.openpi_config, "dir": spec.checkpoint}
    assert log["weights_loaded"]["robot"] == spec.robot
    assert len(log["steady_infer"]["ms"]) == 5 and log["ready"]["port"] == port
    selftest = log["selftest_ws"]
    assert selftest["actions_shape"] == ACTIONS[name]
    assert selftest["noop_reply"] == "{'rs_noop': True}"
    assert "'playground_serve': {'jpeg': True, 'noop': True, 'path': 'stock'" in selftest["metadata"]
    assert log["ready"]["path"] == "stock"  # RS_PATH unset: the stock openpi policy
