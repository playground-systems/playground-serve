"""Local memory of the resources Playground Serve created: {"owner": "o<8 hex>", "models": {<model>: {"endpoint": {...},
"volume": {...}}}}.

One state location per project if you like: $PLAYGROUND_SERVE_HOME/state.json, by default
~/.config/playground-serve/state.json. Its random owner id is part of the name of every resource it creates, so two
state locations on one Runpod account never act on each other's resources (launcher.py). The account is the truth:
`up`, `down` and `delete` re-check these ids against it; `status` and the client only read this file.
"""

from __future__ import annotations

import json
import os
import pathlib

HOME_ENV = "PLAYGROUND_SERVE_HOME"


def config_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config") / "playground-serve"


def default_path() -> pathlib.Path:
    home = os.environ.get(HOME_ENV, "").strip()
    return (pathlib.Path(home).expanduser() if home else config_dir()) / "state.json"


def key_path() -> pathlib.Path:
    """Optional file holding the Runpod API key, read when neither an argument nor RUNPOD_API_KEY gives one. Per user,
    not per project: PLAYGROUND_SERVE_HOME does not move it."""
    return config_dir() / "key"


def save_key(key: str) -> pathlib.Path:
    """Write the Runpod API key to key_path(), readable by this user only (0600)."""
    path = key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.strip() + "\n")
    os.chmod(path, 0o600)  # an existing file keeps its old mode through os.open
    return path


def load(path: pathlib.Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    data.setdefault("models", {})
    return data


def save(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
