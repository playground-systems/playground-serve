"""The local endpoint works without Runpod credentials or network calls."""

import os
import signal
import socket
import subprocess
import time

import pytest

from playground_serve import cli, connect, example_obs


def test_local_server_accepts_a_warmup_observation(monkeypatch, tmp_path):
    pytest.importorskip("inspect_robots_agent")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "OPENROUTER_API_KEY": "test-key"}
    process = subprocess.Popen([os.path.join(os.path.dirname(os.sys.executable), "pgs"), "local", "opus_5_5", "--port", str(port)],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail(f"local server exited: {process.communicate()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("local server did not start")
        obs = example_obs()
        for key in ("observation/image", "observation/wrist_image"):
            obs[key][:] = 0
        with connect(f"ws://127.0.0.1:{port}") as policy:
            result = policy.infer(obs)
        assert result["actions"].shape == (5, 7)
        assert result["agent"]["warm_up"] is True
    finally:
        os.killpg(process.pid, signal.SIGINT)
        process.communicate(timeout=10)


def test_local_rejects_vla_without_runpod_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert cli.main(["local", "pi05_libero"]) == 1
    assert "LLM agents only" in capsys.readouterr().err
