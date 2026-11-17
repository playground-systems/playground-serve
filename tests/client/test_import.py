r"""The client imports without the launcher's modules (so it runs on Python 3.11), and `pgs` on an older Python says in
one sentence that it needs 3.11.

The client suite on Python 3.11, installed independently of the development lock:

    uv venv --python 3.11 /tmp/pgs-py311
    uv pip install --python /tmp/pgs-py311/bin/python -e . "pytest>=8.3" \
        "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client"
    /tmp/pgs-py311/bin/python -m pytest tests/client tests/test_connect.py tests/test_status.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
LAUNCHER_MODULES = {"playground_serve.launcher", "playground_serve.runpod_api", "playground_serve.cli",
                    "playground_serve.models", "playground_serve.bundle"}


def test_client_import_does_not_load_the_launcher():
    code = ("import sys\n"
            "from playground_serve import ActionQueue, Status, connect, example_obs, status\n"
            "assert example_obs()['observation/image'].shape == (224, 224, 3)\n"
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('playground_serve'))))\n")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(SRC)}).stdout.split()
    assert "playground_serve.client" in out and "playground_serve.resolve" in out
    assert not LAUNCHER_MODULES & set(out)


def test_pgs_before_311_says_so_in_one_sentence(monkeypatch):
    from playground_serve import _pgs

    monkeypatch.setattr(sys, "version_info", (3, 8, 20, "final", 0))
    with pytest.raises(SystemExit) as e:
        _pgs.main()
    assert e.value.code == "playground-serve needs Python 3.11 or later for its security-patched image decoder."


@pytest.mark.skipif(sys.version_info < (3, 11), reason="pgs runs on Python 3.11 or later")
def test_pgs_runs_the_cli_from_311(monkeypatch, capsys):
    from playground_serve import _pgs

    monkeypatch.setattr(sys, "argv", ["pgs", "--version"])
    with pytest.raises(SystemExit) as e:
        _pgs.main()
    assert e.value.code == 0 and capsys.readouterr().out.startswith("playground-serve ")
