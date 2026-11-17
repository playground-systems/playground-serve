"""Embodiment cards: the robot and its action space, as data that the simulator side sends at an episode's start.

A card says two things. `agent` is the interface the language model drives, in Inspect Robots' terms (an action Box
with ActionSemantics, cameras and state fields; inspect_robots.spaces), so Inspect Robots' agent policy builds its
tools from it. `native` is the environment's own wire: the request's state vector and the action vector that each reply
row must be, and how the server turns the agent's targets into those rows. Two native kinds exist:

- "same": the environment takes the agent's actions as they are (absolute joint targets, or an absolute hand pose with
  a rot6d rotation and a 0 to 1 gripper). The request's state vector is the agent's reference state field.
- "eef_delta": the environment takes per-step deltas of the hand (LIBERO's OSC_POSE: position and axis-angle deltas
  scaled to [-1, 1], and a gripper command), and its state holds the hand's position, its axis-angle rotation and
  the two finger positions. The agent then drives an absolute hand pose (x, y, z, rot6d, gripper 0 open to 1
  closed), and the server servoes each target with a proportional controller, then holds the last target until the
  hand arrives or stops making progress before it asks the model again.

The card's fields are listed in CARD_FIELDS and in docs/AGENTS.md. A card is checked with `load`; any problem raises
CardError with the field named.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CARD_VERSION = 1
HERE = pathlib.Path(__file__).resolve().parent
BUILTIN_FILE = HERE / "cards.json"
CARD_FIELDS = ("card", "name", "morphology", "docs", "control_hz", "chunk", "prompt", "cameras", "agent", "native")
ROT6D_LABELS = ("r00", "r10", "r20", "r01", "r11", "r21")


class CardError(ValueError):
    pass


@dataclass
class Card:
    name: str
    docs: str
    control_hz: float
    chunk: int  # rows of each reply: the actions the client runs before its next request
    prompt: str  # the request key of the task's text
    cameras: dict[str, str]  # agent camera name -> request key ("a/b" walks nested maps)
    agent: dict[str, Any]
    native: dict[str, Any]
    morphology: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def action_size(self) -> int:
        return int(self.native["action_size"])

    @property
    def kind(self) -> str:
        return self.native["kind"]


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise CardError(what)


def _slice(value: Any, what: str) -> tuple[int, int]:
    _need(isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value) and 0 <= value[0] < value[1],
          f"{what} must be [start, end] with 0 <= start < end")
    return int(value[0]), int(value[1])


def load(card: Any) -> Card:
    """A card from a dict, a JSON string, or the name of a built-in card (cards.json)."""
    if isinstance(card, (bytes, bytearray)):
        card = card.decode()
    if isinstance(card, str):
        text = card.strip()
        if not text.startswith("{"):
            found = builtin().get(text)
            if found is None:
                raise CardError(f"no built-in card {text!r}; built-in: {', '.join(sorted(builtin()))}")
            return load(found)
        try:
            card = json.loads(text)
        except json.JSONDecodeError as e:
            raise CardError(f"the card is not JSON: {e}") from None
    _need(isinstance(card, dict), "a card is a JSON object")
    unknown = sorted(set(card) - set(CARD_FIELDS) - {"about"})
    _need(not unknown, f"unknown card fields {unknown}; fields: {', '.join(CARD_FIELDS)}")
    _need(card.get("card") == CARD_VERSION, f"card must be {CARD_VERSION} (the card format's version)")
    for key in ("name", "prompt"):
        _need(isinstance(card.get(key), str) and card[key].strip() != "", f"{key} must be a non-empty string")
    hz, chunk = card.get("control_hz"), card.get("chunk")
    _need(isinstance(hz, (int, float)) and not isinstance(hz, bool) and math.isfinite(hz) and hz > 0, "control_hz must be > 0")
    _need(isinstance(chunk, int) and not isinstance(chunk, bool) and 1 <= chunk <= 200, "chunk must be an integer from 1 to 200")
    cameras = card.get("cameras")
    _need(isinstance(cameras, dict) and cameras and all(isinstance(k, str) and isinstance(v, str) for k, v in cameras.items()),
          "cameras must map each camera name to its request key")
    agent, native = card.get("agent"), card.get("native")
    _need(isinstance(agent, dict), "agent must be an object")
    _need(isinstance(native, dict), "native must be an object")
    mode = agent.get("control_mode")
    _need(mode in ("joint_pos", "eef_abs_pose"), "agent.control_mode must be joint_pos or eef_abs_pose")
    labels, low, high = agent.get("labels"), agent.get("low"), agent.get("high")
    _need(isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels), "agent.labels must name each dimension")
    n = len(labels)
    for name, v in (("low", low), ("high", high)):
        _need(isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) for x in v), f"agent.{name} needs {n} numbers")
    _need(all(a <= b for a, b in zip(low, high)), "agent.low must be at most agent.high in every dimension")
    max_step = agent.get("max_step")
    _need(max_step is None or (isinstance(max_step, list) and len(max_step) == n), f"agent.max_step needs {n} entries or null")
    _need(agent.get("rotation_repr", "none") in ("none", "rot6d"), "agent.rotation_repr must be none or rot6d")
    _need(agent.get("gripper", "none") in ("none", "continuous", "binary"), "agent.gripper must be none, continuous or binary")
    _need(agent.get("frame", "base") in ("base", "world"), "agent.frame must be base or world")
    kind = native.get("kind")
    _need(kind in ("same", "eef_delta"), "native.kind must be same or eef_delta")
    _need(isinstance(native.get("state"), str), "native.state must name the request key of the state vector")
    size = native.get("action_size")
    _need(isinstance(size, int) and size > 0, "native.action_size must be a positive integer")
    if kind == "same":
        _need(size == n, f"native.kind same: native.action_size ({size}) must equal the agent's {n} dimensions")
    else:
        _need(mode == "eef_abs_pose" and agent.get("rotation_repr") == "rot6d" and n == 10
              and labels[:3] == ["x", "y", "z"] and tuple(labels[3:9]) == ROT6D_LABELS,
              "native.kind eef_delta needs the agent's eef_abs_pose interface x, y, z, r00, r10, r20, r01, r11, r21, gripper")
        d = native.get("eef_delta")
        _need(isinstance(d, dict), "native.eef_delta must be an object")
        for key in ("state_position", "state_rotation", "state_fingers", "action_position", "action_rotation"):
            _slice(d.get(key), f"native.eef_delta.{key}")
        _need(isinstance(d.get("action_gripper"), int), "native.eef_delta.action_gripper must be an index")
        for key in ("position_scale", "rotation_scale", "finger_full", "gripper_open", "gripper_closed"):
            _need(isinstance(d.get(key), (int, float)), f"native.eef_delta.{key} must be a number")
    for key, value in native.get("extra_state", {}).items():
        _slice(value, f"native.extra_state.{key}")
    return Card(name=card["name"], docs=str(card.get("docs") or ""), control_hz=float(hz), chunk=int(chunk),
                prompt=card["prompt"], cameras=dict(cameras), agent=dict(agent), native=dict(native),
                morphology=str(card.get("morphology") or ""), raw=card)


_BUILTIN: dict[str, dict] | None = None


def builtin() -> dict[str, dict]:
    """The built-in cards (cards.json): for a client that sends a card's name, or none at all."""
    global _BUILTIN
    if _BUILTIN is None:
        _BUILTIN = {c["name"]: c for c in json.loads(BUILTIN_FILE.read_text())["cards"]}
    return _BUILTIN


def guess(request: dict) -> Card | None:
    """The built-in card whose prompt, state and camera keys a request has, for a client that sends no card."""
    for raw in builtin().values():
        c = load(raw)
        if c.prompt in request and lookup(request, c.native["state"]) is not None and all(
                lookup(request, k) is not None for k in c.cameras.values()):
            return c
    return None


def lookup(request: dict, key: str) -> Any:
    """request[key], or the nested value for a key written a/b when the flat key is absent."""
    if key in request:
        return request[key]
    node: Any = request
    for part in key.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def image(value: Any) -> np.ndarray | None:
    """An HxWx3 uint8 image from a request value (HWC or CHW, uint8 or floats in [0, 1]), or None."""
    if not isinstance(value, np.ndarray) or value.ndim != 3:
        return None
    a = value
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if np.issubdtype(a.dtype, np.floating):
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.ascontiguousarray(a)


# ------------------------------------------------------------------ rotations
def rotation(aa: np.ndarray) -> np.ndarray:
    """Axis-angle to a rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-9:
        return np.eye(3)
    k = np.asarray(aa, dtype=float) / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def axis_angle(R: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.sin(angle) < 1e-3:  # near pi: k k^T from R's symmetric part, from its largest column; sign from the skew part
        c = math.cos(angle)
        B = ((R + R.T) / 2 - c * np.eye(3)) / (1 - c)
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        if float(axis @ v) < 0:
            axis = -axis
        return axis / (np.linalg.norm(axis) or 1) * angle
    return v / (2 * math.sin(angle)) * angle


def rot6d(R: np.ndarray) -> np.ndarray:
    """The first two columns of R: r00, r10, r20, r01, r11, r21."""
    return np.concatenate([R[:, 0], R[:, 1]])


def from_rot6d(r: np.ndarray) -> np.ndarray:
    a, b = np.asarray(r[:3], dtype=float), np.asarray(r[3:6], dtype=float)
    x = a / (np.linalg.norm(a) or 1.0)
    b = b - x * float(x @ b)
    y = b / (np.linalg.norm(b) or 1.0)
    return np.stack([x, y, np.cross(x, y)], axis=1)
