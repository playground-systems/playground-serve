"""One episode of the LLM agent: Inspect Robots' agent policy bound to the episode's card, and the driver that turns
its targets into the rows each reply carries.

The agent loop is inspect-robots-agent's LLMAgentPolicy, unchanged (pinned in requirements.txt): each act() shows the
model the state as labelled text and the camera images, runs model turns until one tool call is valid (move_to or
move_joints, sleep, take_pic with images on demand, done, give_up), and returns an Inspect Robots ActionChunk of
interpolated targets. Its model layer speaks each provider's own wire (Anthropic Messages, OpenAI Chat Completions and
Responses, Google's Interactions and Live APIs) and any OpenAI-compatible server through a base URL.

This module adds what a remote simulator needs around it:
- Settings: the model and its limits, read once from the endpoint's environment (never from a request).
- Driver: queues the agent's targets and gives `card.chunk` native rows per request, with the hold, the settle and the
  conversion of the card's native kind (llm_card.py).
- Session: one episode; calls act() only when the driver has played the last motion, adds env_step to the
  observation as Inspect Robots' rollout does, stops at the call and USD caps, and reports usage and the new
  transcript messages (image-free) in each reply's "agent" map.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

import llm_card
import llm_pricing
from llm_card import Card

KEY_ENVS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY",
            "GROQ_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY")
# Wire errors that the agent raises after its retries (inspect_robots_agent: "LLM request rejected" for a 4xx,
# "LLM request failed after N attempts"): the provider or the key is at fault, so the request fails as a server error
# (the run's policy_error) instead of the arm holding still through every later episode.
WIRE_FAILURES = ("LLM request rejected", "LLM request failed")


class AgentFailure(RuntimeError):
    pass


# The agent's evaluation notes on every episode (the user's own notes, PGS_AGENT_NOTES, follow them). They are about the
# setting and the use of the step budget, never about a task or a benchmark.
DEFAULT_NOTES = ("The robot, the objects and their physics are simulated; the camera images are rendered. Plan each "
                 "motion against the step budget in the observation: look at every camera before the first move, combine "
                 "axes in one move where the path is clear, and after closing the gripper read its state to see whether it "
                 "holds something.")


@dataclass(frozen=True)
class Settings:
    model: str = "anthropic/claude-haiku-5-5"
    wire: str = ""  # "" = messages for anthropic/*, else chat
    base_url: str | None = None
    api_key_env: str | None = None
    effort: str | None = "medium"
    max_calls: int = 40  # model calls per episode
    max_usd_episode: float | None = 1.0
    max_usd_total: float | None = None  # this process, over every episode
    images: str = "always"  # or on_demand: no images unless the model calls take_pic
    image_horizon: int | None = None  # None keeps every image (append-only history; see NOTES.md)
    notes: str | None = DEFAULT_NOTES  # the agent's eval_notes: DEFAULT_NOTES, then the endpoint's additions
    max_speed_frac: float = 0.1
    keys: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def resolved_wire(self) -> str:
        return self.wire or ("messages" if self.model.startswith("anthropic/") and not self.base_url else "chat")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        num = lambda k, d: (float(e[k]) if e.get(k, "").strip() not in ("", "none") else (None if e.get(k, "").strip() == "none" else d))  # noqa: E731
        key_names = set(KEY_ENVS) | ({e["PGS_AGENT_API_KEY_ENV"]} if e.get("PGS_AGENT_API_KEY_ENV") else set())
        horizon = e.get("PGS_AGENT_IMAGE_HORIZON", "").strip()
        s = cls(model=e.get("PGS_AGENT_MODEL") or cls.model, wire=e.get("PGS_AGENT_WIRE", ""),
                base_url=e.get("PGS_AGENT_BASE_URL") or None, api_key_env=e.get("PGS_AGENT_API_KEY_ENV") or None,
                effort=(e.get("PGS_AGENT_EFFORT") or "medium") if e.get("PGS_AGENT_EFFORT", "") != "default" else None,
                max_calls=int(e.get("PGS_AGENT_MAX_CALLS") or 40),
                max_usd_episode=num("PGS_AGENT_MAX_USD_EPISODE", 1.0), max_usd_total=num("PGS_AGENT_MAX_USD", None),
                images=e.get("PGS_AGENT_IMAGES") or "always",
                image_horizon=int(horizon) if horizon not in ("", "none") else None,
                notes="\n".join(x for x in (DEFAULT_NOTES, e.get("PGS_AGENT_NOTES")) if x), max_speed_frac=float(e.get("PGS_AGENT_MAX_SPEED_FRAC") or 0.1),
                keys={k: e[k] for k in key_names if e.get(k)})
        s.check(e)
        return s

    def check(self, env: Mapping[str, str] | None = None) -> None:
        if self.base_url and not (self.base_url.startswith("https://") or os.environ.get("PGS_AGENT_ALLOW_HTTP") == "1"):
            raise ValueError("PGS_AGENT_BASE_URL must be https:// (PGS_AGENT_ALLOW_HTTP=1 for a local server on this machine)")
        capped = self.max_usd_episode is not None or self.max_usd_total is not None
        if capped and llm_pricing.price(self.model, env) is None:
            raise ValueError(f"{self.model} has no price here, so a USD cap cannot be kept: set PGS_AGENT_PRICE, or "
                             "PGS_AGENT_MAX_USD_EPISODE=none and PGS_AGENT_MAX_USD=none")

    def public(self) -> dict[str, Any]:
        """The settings without keys, for the metadata and logs."""
        return {"model": self.model, "wire": self.resolved_wire, "base_url": self.base_url, "effort": self.effort,
                "max_calls": self.max_calls, "max_usd_episode": self.max_usd_episode, "max_usd_total": self.max_usd_total,
                "images": self.images, "image_horizon": self.image_horizon, "notes": bool(self.notes and self.notes != DEFAULT_NOTES)}


def make_policy(s: Settings, transport: Any = None):
    """inspect-robots-agent's LLMAgentPolicy for one episode. The keys go in its env mapping, so each policy sends
    only the key of its own provider, and only to that provider's endpoint (or the endpoint's base URL)."""
    from inspect_robots_agent.policy import LLMAgentPolicy

    return LLMAgentPolicy(model=s.model, base_url=s.base_url, api_key_env=s.api_key_env, wire=s.resolved_wire,
                          wire_capture=False, max_llm_calls=s.max_calls, effort=s.effort, images=s.images,
                          image_horizon=s.image_horizon, eval_notes=s.notes, max_speed_frac=s.max_speed_frac,
                          transport=transport, env=dict(s.keys), transcript_echo=False)


def embodiment_info(card: Card, sizes: Mapping[str, tuple[int, int]], extra: Mapping[str, tuple[int, ...]] | None = None):
    """The card's agent interface as Inspect Robots' EmbodimentInfo, with each camera's size from the first request."""
    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec

    a = card.agent
    n = len(a["labels"])
    step = a.get("max_step")
    box = Box(shape=(n,), low=np.asarray(a["low"], dtype=np.float64), high=np.asarray(a["high"], dtype=np.float64),
              semantics=ActionSemantics(control_mode=a["control_mode"], rotation_repr=a.get("rotation_repr", "none"),
                                        gripper=a.get("gripper", "none"), frame=a.get("frame", "base"),
                                        dim_labels=tuple(a["labels"]),
                                        max_step=None if step is None else tuple(None if v is None else float(v) for v in step)))
    fields = [StateField(k, shape) for k, shape in {**state_shapes(card), **(extra or {})}.items()]
    cams = tuple(CameraSpec(name, *sizes[name]) for name in card.cameras if name in sizes)
    return EmbodimentInfo(name=card.name, action_space=box, observation_space=ObservationSpace(cameras=cams, state=StateSpec(tuple(fields))),
                          control_hz=card.control_hz, is_simulated=True, docs=card.docs or None, morphology=card.morphology or None,
                          environment_id="playground-serve-llm-agent")


def reference_field(card: Card) -> str:
    return card.native.get("state_field") or ("joint_pos" if card.agent["control_mode"] == "joint_pos" else "eef_state")


def state_shapes(card: Card) -> dict[str, tuple[int, ...]]:
    shapes = {reference_field(card): (len(card.agent["labels"]),)}
    if card.kind == "eef_delta":
        s, e = card.native["eef_delta"]["state_fingers"]
        shapes["gripper_qpos"] = (e - s,)
    for k, (s, e) in card.native.get("extra_state", {}).items():
        shapes[k] = (e - s,)
    return shapes


def extra_shapes(card: Card, request: dict) -> dict[str, tuple[int, ...]]:
    """The shapes of the card's extra_keys state fields, read from a request."""
    out = {}
    for k, key in card.native.get("extra_keys", {}).items():
        v = llm_card.lookup(request, key)
        if v is not None:
            out[k] = tuple(np.asarray(v).reshape(-1).shape)
    return out


class Driver:
    """The agent's targets as native rows, `card.chunk` per request."""

    def __init__(self, card: Card) -> None:
        self.card = card
        self.queue: list[np.ndarray] = []
        self.held: np.ndarray | None = None  # the last target sent: what the arm holds once the queue is empty
        self.settle_left = 0
        self.best, self.stall, self.grip_wait = math.inf, 0, 0
        d = card.native.get("eef_delta") or {}
        self.d = {"track": 0.2, "gain": 1.0, "arrive_m": 0.004, "arrive_deg": 2.0, "settle_steps": 60, "stall_steps": 15, "grip_steps": 15, **d}

    def state(self, request: dict) -> np.ndarray:
        value = llm_card.lookup(request, self.card.native["state"])
        if value is None:
            raise ValueError(f"the request has no {self.card.native['state']!r} (the card's native.state)")
        return np.asarray(value, dtype=np.float64).reshape(-1)

    def _pose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        d = self.d
        p = x[slice(*d["state_position"])]
        R = llm_card.rotation(x[slice(*d["state_rotation"])])
        f = x[slice(*d["state_fingers"])]
        opening = float(np.clip((f[0] - f[-1]) / d["finger_full"], 0, 1)) if len(f) > 1 else float(np.clip(f[0] / d["finger_full"], 0, 1))
        return p, R, opening

    def agent_state(self, x: np.ndarray, request: dict | None = None) -> dict[str, np.ndarray]:
        """The agent's state fields from the native state vector (and the request, for the card's extra_keys)."""
        out: dict[str, np.ndarray] = {}
        if self.card.kind == "same":
            out[reference_field(self.card)] = x[: len(self.card.agent["labels"])].copy()
        else:
            p, R, opening = self._pose(x)
            g = float(self.held[9]) if self.held is not None else (0.0 if opening > 0.5 else 1.0)
            out["eef_state"] = np.concatenate([p, llm_card.rot6d(R), [g]])
            out["gripper_qpos"] = x[slice(*self.d["state_fingers"])].copy()
        for k, (s, e) in self.card.native.get("extra_state", {}).items():
            out[k] = x[s:e].copy()
        for k, key in self.card.native.get("extra_keys", {}).items():
            v = llm_card.lookup(request or {}, key)
            if v is not None:
                out[k] = np.asarray(v, dtype=np.float64).reshape(-1)
        return out

    def load(self, targets: list[np.ndarray]) -> None:
        if not targets:
            return
        before = self.held
        self.queue = [np.asarray(t, dtype=np.float64) for t in targets]
        if self.card.kind == "eef_delta":
            self.settle_left = int(self.d["settle_steps"])
            self.best, self.stall = math.inf, 0
            last = self.queue[-1]
            changed = before is None or abs(float(last[9]) - float(before[9])) > 0.25
            self.grip_wait = int(self.d["grip_steps"]) if changed and before is not None else 0

    def busy(self, x: np.ndarray) -> bool:
        """Whether the last motion still plays or settles; updates the settle from the measured state."""
        if self.queue:
            return True
        if self.settle_left <= 0 or self.held is None or self.card.kind != "eef_delta":
            return False
        p, R, _ = self._pose(x)
        dist = float(np.linalg.norm(self.held[:3] - p))
        angle = math.degrees(float(np.linalg.norm(llm_card.axis_angle(llm_card.from_rot6d(self.held[3:9]) @ R.T))))
        score = dist + 0.0005 * angle
        self.stall = self.stall + self.card.chunk if score > self.best - 0.001 else 0
        self.best = min(self.best, score)
        if (dist < self.d["arrive_m"] and angle < self.d["arrive_deg"] and self.grip_wait <= 0) or self.stall >= self.d["stall_steps"]:
            self.settle_left = 0
        return self.settle_left > 0

    def hold_row(self, x: np.ndarray) -> np.ndarray:
        if self.card.kind == "same":
            return (self.held if self.held is not None else x[: self.card.action_size]).copy()
        d = self.d
        row = np.zeros(self.card.action_size)
        g = float(self.held[9]) if self.held is not None else (0.0 if self._pose(x)[2] > 0.5 else 1.0)
        row[d["action_gripper"]] = d["gripper_open"] + (d["gripper_closed"] - d["gripper_open"]) * g
        return row

    def rows(self, x: np.ndarray) -> np.ndarray:
        n = self.card.chunk
        out = np.zeros((n, self.card.action_size))
        if self.card.kind == "same":
            for k in range(n):
                if self.queue:
                    self.held = self.queue.pop(0)
                out[k] = self.held if self.held is not None else x[: self.card.action_size]
            return out
        d = self.d
        p, R, _ = self._pose(x)
        ps, rs, track, gain = float(d["position_scale"]), float(d["rotation_scale"]), float(d["track"]), float(d["gain"])
        for k in range(n):
            if self.queue:
                self.held = self.queue.pop(0)
            elif self.settle_left > 0:
                self.settle_left -= 1
            if self.held is None:
                out[k] = self.hold_row(x)
                continue
            if not self.queue:
                self.grip_wait -= 1  # the gripper's new command has reached its target: count its closing time
            target = self.held
            a = np.clip(gain * (target[:3] - p) / ps, -1, 1)
            p = p + a * ps * track
            r = np.clip(gain * llm_card.axis_angle(llm_card.from_rot6d(target[3:9]) @ R.T) / rs, -1, 1)
            R = llm_card.rotation(r * rs * track) @ R
            out[k, slice(*d["action_position"])] = a
            out[k, slice(*d["action_rotation"])] = r
            g = float(np.clip(target[9], 0, 1))
            out[k, d["action_gripper"]] = d["gripper_open"] + (d["gripper_closed"] - d["gripper_open"]) * g
        return out


class Ledger:
    """USD over every episode of this process, for PGS_AGENT_MAX_USD."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.usd = 0.0

    def add(self, usd: float) -> float:
        with self.lock:
            self.usd += usd
            return self.usd


class Session:
    def __init__(self, key: str, card: Card, settings: Settings, ledger: Ledger,
                 policy_factory: Callable[[Settings], Any] = make_policy) -> None:
        self.key, self.card, self.settings, self.ledger = key, card, settings, ledger
        self.factory = policy_factory
        self.driver = Driver(card)
        self.policy = None
        self.steps = 0
        self.finished: dict[str, str] | None = None
        self.error: str | None = None
        self.usd = 0.0
        self.calls = 0
        self.usage: dict[str, float] = {}
        self.seconds = 0.0
        self.instruction = ""
        self.touched = time.monotonic()
        self.lock = threading.Lock()

    def _observation(self, request: dict, x: np.ndarray):
        from inspect_robots.types import Observation

        images = {}
        for name, key in self.card.cameras.items():
            img = llm_card.image(llm_card.lookup(request, key))
            if img is not None:
                images[name] = img
        return Observation(images=images, state=self.driver.agent_state(x, request), instruction=self.instruction,
                           extra={"env_step": self.steps})

    def _start(self, request: dict, obs) -> None:
        from inspect_robots.scene import Scene
        from inspect_robots.task import TaskEnvelope

        self.policy = self.factory(self.settings)
        self.policy.bind(embodiment_info(self.card, {k: v.shape[:2] for k, v in obs.images.items()}, extra_shapes(self.card, request)))
        steps = request.get("max_steps") or self.card.max_steps
        if isinstance(steps, (int, np.integer)) and steps > 0:
            self.policy.bind_task(TaskEnvelope(name=self.card.name, max_steps=int(steps)))
        self.policy.reset(Scene(id=self.key, instruction=self.instruction))

    def step(self, request: dict) -> dict:
        with self.lock:
            self.touched = time.monotonic()
            x = self.driver.state(request)
            turns = None
            if self.finished is None and not self.driver.busy(x):
                turns = self._decide(request, x)
            rows = self.driver.rows(x)
            self.steps += self.card.chunk
            return {"actions": rows.astype(np.float32), "agent": self.report(turns)}

    def _decide(self, request: dict, x: np.ndarray) -> list | None:
        s = self.settings
        if s.max_usd_episode is not None and self.usd >= s.max_usd_episode:
            self.finished = {"kind": "budget", "reason": f"the episode's USD cap ({s.max_usd_episode}) is spent"}
            return None
        if s.max_usd_total is not None and self.ledger.usd >= s.max_usd_total:
            self.finished = {"kind": "budget", "reason": f"the server's USD cap ({s.max_usd_total}) is spent"}
            return None
        self.instruction = str(llm_card.lookup(request, self.card.prompt) or "")
        obs = self._observation(request, x)
        if self.policy is None:
            self._start(request, obs)
        started = time.monotonic()
        try:
            chunk = self.policy.act(obs)
        except RuntimeError as e:
            self._account()
            if str(e).startswith(WIRE_FAILURES):
                raise AgentFailure(str(e)) from None
            self.finished, self.error = {"kind": "give_up", "reason": "the agent stopped: " + str(e)[:300]}, str(e)[:300]
            return self.policy.transcript_delta()
        finally:
            self.seconds += time.monotonic() - started
        self._account()
        self.driver.load([np.asarray(a.data, dtype=np.float64) for a in chunk.actions])
        if chunk.terminal is not None:
            self.finished = {"kind": chunk.terminal.kind, "reason": chunk.terminal.reason}
        return self.policy.transcript_delta()

    def _account(self) -> None:
        p = self.policy
        if p is None:
            return
        totals = dict(getattr(p, "_usage_totals", {}) or {})  # inspect-robots-agent 0.30.0: summed per trial
        self.calls = int(getattr(p, "_calls_used", 0) or 0)
        usd = llm_pricing.usd(self.settings.model, totals)
        if usd is not None and usd > self.usd:
            self.ledger.add(usd - self.usd)
            self.usd = usd
        self.usage = totals

    def report(self, turns: list | None = None) -> dict:
        priced = llm_pricing.price(self.settings.model) is not None
        out: dict[str, Any] = {"episode": self.key, "card": self.card.name, "model": self.settings.model, "calls": self.calls,
                               "tokens": llm_pricing.tokens(self.usage), "usd": round(self.usd, 6) if priced else None,
                               "model_seconds": round(self.seconds, 2), "steps": self.steps}
        if self.finished:
            out["finished"] = self.finished
        if self.error:
            out["error"] = self.error
        if turns:
            out["turns"] = turns
        return out
