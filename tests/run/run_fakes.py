"""A kinematic stand-in for the SO-101 and a stand-in policy in MolmoAct2-SO100_101's units, for the `pgs run` tests."""

from __future__ import annotations

import threading
import time

import numpy as np

from playground_serve.run import so101
from playground_serve.run.loop import Robot

MAPPING_SPAN = so101.OLD_GRIPPER_TICKS  # gripper scale 1: model gripper == robot gripper
START = np.array([0.0, -60.0, 60.0, 50.0, 0.0, 10.0])  # robot units, inside the limits below


class FakeRobot(Robot):
    """Follows every target at once; records what the loop does to it. Joint range +-100 degrees, gripper 0 to 100."""

    joints = so101.JOINTS
    max_speed = so101.MAX_SPEED
    gripper_span_ticks = MAPPING_SPAN

    def __init__(self, start=START, lower=None, upper=None) -> None:
        self.state = np.array(start, dtype=np.float64)
        self.lower = np.array(lower if lower is not None else [-100.0] * 5 + [0.0])
        self.upper = np.array(upper if upper is not None else [100.0] * 5 + [100.0])
        self.sent: list = []  # (time, target)
        self.holds = self.reads = 0
        self.connected = False
        self.torque_off = None  # set by disconnect
        self.image = np.zeros((48, 64, 3), np.uint8)

    def connect(self) -> None:
        self.connected = True

    def read(self):
        self.reads += 1
        return self.state.copy(), {"top": self.image, "side": self.image}

    def send(self, target) -> None:
        self.sent.append((time.monotonic(), np.array(target, dtype=np.float64)))
        self.state = np.array(target, dtype=np.float64)

    def hold(self) -> None:
        self.holds += 1

    def disconnect(self, torque_off: bool = True) -> None:
        self.connected = False
        self.torque_off = torque_off

    def targets(self) -> np.ndarray:
        return np.array([t for _, t in self.sent])


class FakePolicy:
    """Answers each observation with a chunk in model units: its state plus `step` per action (a steady move), plus
    `offset`, after delays[i] seconds (the last delay repeats). `fail_after`: calls after that many raise, as a lost
    connection does."""

    metadata = {"model": "molmoact2_so101", "action_hz": 30}

    def __init__(self, horizon=30, step=(0.5, 0, 0, 0, 0, 0), offset=(0, 0, 0, 0, 0, 0), delays=(0.01,),
                 fail_after=None) -> None:
        self.horizon, self.step, self.offset = horizon, np.array(step, float), np.array(offset, float)
        self.delays, self.fail_after = list(delays), fail_after
        self.observations: list = []
        self.timing: dict = {}
        self._lock = threading.Lock()

    def infer(self, obs: dict) -> dict:
        with self._lock:
            n = len(self.observations)
            self.observations.append(obs)
        if self.fail_after is not None and n >= self.fail_after:
            time.sleep(0.01)
            raise ConnectionError("connection lost")
        delay = self.delays[min(n, len(self.delays) - 1)]
        time.sleep(delay)
        state = np.asarray(obs["state"], dtype=np.float64)
        actions = state + np.arange(1, self.horizon + 1)[:, None] * self.step + self.offset
        self.timing = {"rtt_ms": delay * 1e3 + 1, "infer_ms": delay * 1e3}
        return {"actions": actions.astype(np.float32), "policy_timing": {"model_ms": delay * 1e3}}


def mapping(instruction: str = "put the cube in the bowl") -> so101.MolmoAct2SO101:
    return so101.MolmoAct2SO101(MAPPING_SPAN, instruction)
