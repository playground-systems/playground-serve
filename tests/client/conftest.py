from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pytest

FRAMES = Path(__file__).resolve().parent / "data" / "libero_frames.npz"  # four LIBERO frames (NOTICE)
IMAGE_KEYS = ("observation/image", "observation/wrist_image")


@pytest.fixture(scope="session")
def observations() -> list[dict]:
    """Four LIBERO observations in the layout of openpi's LIBERO example (256x256 renders, 8-D state, prompt)."""
    d = np.load(FRAMES)
    return [{"observation/image": d["image"][i], "observation/wrist_image": d["wrist_image"][i],
             "observation/state": d["state"][i], "prompt": str(d["prompt"])} for i in range(len(d["image"]))]


def wait_for(condition, timeout: float = 30.0) -> bool:
    """Poll `condition` until it holds or `timeout` (generous: a loaded machine runs threads late) runs out."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RecordingPolicy:
    """An openpi-style policy that keeps every observation it receives."""

    metadata = {"policy": "recording"}

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def infer(self, obs: dict) -> dict:
        self.seen.append(obs)
        return {"actions": np.ones((10, 7), np.float32)}

    def reset(self) -> None:
        pass
