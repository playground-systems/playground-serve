"""Run the existing agent server locally, without provisioning a Runpod endpoint."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import models
from .launcher import LauncherError


def run(model: str, port: int, *, out=print) -> int:
    try:
        spec = models.get(model)
    except (KeyError, ValueError) as e:
        raise LauncherError(str(e).strip("'\"")) from None
    if spec.family != "llm_agent":
        raise LauncherError("local mode currently supports LLM agents only; use `pgs models` to see them")
    if sys.version_info < (3, 11):
        raise LauncherError("local LLM agents need Python 3.11 or later")
    if not 1 <= port <= 65535:
        raise LauncherError("--port must be between 1 and 65535")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise LauncherError("set OPENROUTER_API_KEY to run an LLM agent")
    if any(importlib.util.find_spec(name) is None for name in ("inspect_robots", "inspect_robots_agent", "httpx")):
        raise LauncherError('local agent dependencies are missing: install playground-serve with its "llm" extra')
    root = Path(__file__).resolve().parent
    agent, common, worker = root / "families" / "llm_agent", root / "families" / "common", root / "worker"
    env = os.environ.copy()
    env["PGS_AGENT_MODEL"] = spec.name
    env["PYTHONPATH"] = os.pathsep.join([str(agent), str(common), str(worker), env.get("PYTHONPATH", "")])
    with tempfile.TemporaryDirectory(prefix="pgs-local-") as tmp:
        env["RS_READY_FILE"] = str(Path(tmp) / "ready")
        out(f"Starting {spec.name} locally at ws://127.0.0.1:{port}; press Ctrl-C to stop. "
            "OpenRouter token charges apply.")
        return subprocess.call([sys.executable, str(agent / "serve.py"), "--host", "127.0.0.1", "--port", str(port)], env=env)
