"""The LLM agent family without a model: embodiment cards, the driver's conversion, episodes through Inspect Robots'
agent policy against recorded tool calls (llm_fake.py) on three wire formats, the caps, and the server over the wire.

Needs inspect-robots and inspect-robots-agent (families/llm_agent/requirements.txt); skipped without them."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("inspect_robots_agent")

from playground_serve import bundle, models  # noqa: E402

PKG = pathlib.Path(models.__file__).resolve().parent
FAMILY = PKG / "families" / "llm_agent"
for d in (FAMILY, PKG / "families" / "common", PKG / "worker"):
    if str(d) not in sys.path:
        sys.path.append(str(d))

import llm_card  # noqa: E402
import llm_fake  # noqa: E402
import llm_pricing  # noqa: E402
import llm_session  # noqa: E402
import serve  # noqa: E402

TURNS = [
    {"tool": "move_to", "arguments": {"targets": {"z": 1.0}, "note": "lower toward the bowl"}},
    {"tool": "move_to", "arguments": {"targets": {"gripper": 1}, "note": "close on the rim"}},
    {"tool": "done", "arguments": {"summary": "the bowl is held", "hindsight": "none"}},
]


class Libero:
    """A stand-in for LIBERO behind OpenPIProxy: 224 x 224 images, the 8-D state, deltas that move the hand
    track * scale per unit, fingers that follow the gripper's sign."""

    def __init__(self, episode: str | None = "ep1") -> None:
        self.state = np.array([-0.2, 0.0, 1.17, np.pi, 0.0, 0.0, 0.039, -0.039])
        self.rng = np.random.default_rng(0)
        self.episode = episode

    def request(self, card: str | dict | None = None) -> dict:
        r = {"observation/image": self.rng.integers(1, 255, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": self.rng.integers(1, 255, (224, 224, 3), dtype=np.uint8),
             "observation/state": self.state.astype(np.float32), "prompt": "pick up the black bowl"}
        if self.episode:
            r["episode"] = self.episode
        if card is not None:
            r["embodiment"] = card
        return r

    def play(self, actions: np.ndarray) -> None:
        for row in actions:
            self.state[:3] += row[:3] * 0.05 * 0.2
            self.state[6:8] = [0.039, -0.039] if row[6] < 0 else [0.02, -0.02]


def policy(turns=TURNS, env=None, wire_model="anthropic/claude-haiku-5-5"):
    fake = llm_fake.FakeProvider({"turns": turns})
    e = {"PGS_AGENT_MODEL": wire_model, "ANTHROPIC_API_KEY": "fake", "OPENAI_API_KEY": "fake", **(env or {})}
    settings = llm_session.Settings.from_env(e)
    return serve.AgentPolicy(settings, lambda s: llm_session.make_policy(s, transport=fake.transport())), fake


def run(pol, sim, n=40, conn="c1", card=None):
    replies = []
    for _ in range(n):
        r = pol.infer(sim.request(card), conn)
        sim.play(r["actions"])
        replies.append(r)
    return replies


# ------------------------------------------------------------------ cards
def test_builtin_card_and_its_checks():
    card = llm_card.load("libero_franka")
    assert card.kind == "eef_delta" and card.action_size == 7 and card.chunk == 5
    assert llm_card.load(json.dumps(card.raw)).raw == card.raw
    assert llm_card.guess(Libero().request()).name == "libero_franka"
    bad = {**card.raw, "native": {**card.raw["native"], "kind": "torque"}}
    with pytest.raises(llm_card.CardError, match="native.kind"):
        llm_card.load(bad)
    with pytest.raises(llm_card.CardError, match="unknown card fields"):
        llm_card.load({**card.raw, "colour": "red"})
    with pytest.raises(llm_card.CardError, match="no built-in card"):
        llm_card.load("nothing")
    same = {**card.raw, "name": "joints", "agent": {"control_mode": "joint_pos", "labels": ["a", "b"], "low": [-1, -1], "high": [1, 1]},
            "native": {"kind": "same", "state": "observation/state", "action_size": 3}}
    with pytest.raises(llm_card.CardError, match="must equal"):
        llm_card.load(same)


def test_rot6d_round_trip():
    R = llm_card.rotation(np.array([np.pi, 0, 0])) @ llm_card.rotation(np.array([0, 0, 0.4]))
    assert np.allclose(llm_card.from_rot6d(llm_card.rot6d(R)), R)
    assert np.allclose(llm_card.rotation(llm_card.axis_angle(R)), R, atol=1e-6)


def test_driver_servoes_to_a_target_and_settles():
    card = llm_card.load("libero_franka")
    d = llm_session.Driver(card)
    sim = Libero()
    x = sim.state.copy()
    start = d.agent_state(x)["eef_state"]
    target = start.copy()
    target[2] = 1.0
    d.load([start + (target - start) * f for f in np.linspace(0.1, 1, 10)])
    for _ in range(40):
        if not d.busy(sim.state):
            break
        sim.play(d.rows(sim.state))
    assert not d.busy(sim.state)
    assert abs(sim.state[2] - 1.0) < 0.006
    hold = d.rows(sim.state)
    assert np.all(np.abs(hold[:, :6]) < 0.2) and np.all(hold[:, 6] == -1)  # holds, gripper open


# ------------------------------------------------------------------ episodes through the agent
def test_episode_on_the_messages_wire():
    pol, fake = policy()
    sim = Libero()
    replies = run(pol, sim)
    calls = [r["agent"]["calls"] for r in replies]
    assert calls[0] == 1 and calls[-1] == 3
    assert abs(sim.state[2] - 1.0) < 0.006  # lowered by the first call's move_to
    last = replies[-1]["agent"]
    assert last["finished"] == {"kind": "done", "reason": "the bowl is held"}
    assert last["tokens"]["input"] == 3000 and last["usd"] == pytest.approx(3 * (1000 * 0.10 + 60 * 0.50) / 1e6)
    first = fake.requests[0]
    assert first["model"] == "claude-haiku-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "low"} and "temperature" not in first and "tool_choice" not in first
    assert "LIBERO tabletop" in first["system"][0]["text"]  # the card's docs
    assert {t["name"] for t in first["tools"]} == {"move_to", "sleep", "done", "give_up"}
    images = [b for m in first["messages"] for b in (m["content"] if isinstance(m["content"], list) else []) if b.get("type") == "image"]
    assert len(images) == 2
    # Append-only history (image_horizon None): each request's messages begin with the request before's.
    strip = lambda node: ({k: strip(v) for k, v in node.items() if k != "cache_control"} if isinstance(node, dict)  # noqa: E731
                          else [strip(v) for v in node] if isinstance(node, list) else node)
    for before, after in zip(fake.requests, fake.requests[1:]):
        assert strip(after["messages"][: len(before["messages"])]) == strip(before["messages"])
    turns = [t for r in replies for t in r["agent"].get("turns") or []]
    assert turns and not any("base64" in json.dumps(t) for t in turns)  # transcript without images


@pytest.mark.parametrize("model,env,path", [
    ("openai/gpt-5.5", {}, "/chat/completions"),
    ("openai/gpt-5.5", {"PGS_AGENT_WIRE": "responses"}, "/responses"),
    ("qwen/qwen3-vl", {"PGS_AGENT_BASE_URL": "https://vllm.example.com/v1", "PGS_AGENT_API_KEY_ENV": "OPENAI_API_KEY",
                       "PGS_AGENT_MAX_USD_EPISODE": "none"}, "/chat/completions"),
])
def test_other_wires(model, env, path):
    pol, fake = policy(env={"PGS_AGENT_PRICE": "1,2", **env} if "PRICE" not in env and model.startswith("openai") else env,
                       wire_model=model)
    sim = Libero()
    replies = run(pol, sim, n=30)
    assert replies[-1]["agent"]["finished"]["kind"] == "done"
    assert abs(sim.state[2] - 1.0) < 0.006
    assert all(r["tools"] for r in fake.requests)
    assert fake.requests and ("input" in fake.requests[0] or "messages" in fake.requests[0])


def test_episodes_by_id_and_by_connection():
    pol, fake = policy(turns=[TURNS[0]] * 10)
    a, b = Libero("a"), Libero("b")
    pol.infer(a.request(), "c1")
    pol.infer(a.request(), "c2")  # a reconnect continues episode a
    pol.infer(b.request(), "c1")
    assert set(pol.sessions) == {"e:a", "e:b"} and len(fake.requests) == 2
    anon = Libero(episode=None)
    pol.infer(anon.request(), "c3")
    assert "c:c3:pick up the black bowl" in pol.sessions
    pol.close_connection("c3")
    assert "c:c3:pick up the black bowl" not in pol.sessions


def test_warm_up_request_calls_no_model():
    pol, fake = policy()
    r = Libero().request()
    r["observation/image"][:] = 0
    r["observation/wrist_image"][:] = 0
    out = pol.infer(r, "c1")
    assert out["agent"]["warm_up"] and out["actions"].shape == (5, 7) and not fake.requests and not pol.sessions


def test_caps():
    pol, fake = policy(turns=[TURNS[0]] * 50, env={"PGS_AGENT_MAX_USD_EPISODE": "0.0002"})
    replies = run(pol, Libero(), n=30)
    assert replies[-1]["agent"]["finished"]["kind"] == "budget" and len(fake.requests) == 2
    pol, fake = policy(turns=[{"text": "thinking out loud"}] * 5)
    reply = pol.infer(Libero().request(), "c1")
    assert reply["agent"]["finished"]["kind"] == "give_up" and "no tool call" in reply["agent"]["error"]


def test_settings():
    with pytest.raises(ValueError, match="no price"):
        llm_session.Settings.from_env({"PGS_AGENT_MODEL": "openai/gpt-5.5"})
    assert llm_session.Settings.from_env({"PGS_AGENT_MODEL": "openai/gpt-5.5", "PGS_AGENT_MAX_USD_EPISODE": "none"}).max_usd_episode is None
    with pytest.raises(ValueError, match="https"):
        llm_session.Settings.from_env({"PGS_AGENT_BASE_URL": "http://10.0.0.2/v1"})
    s = llm_session.Settings.from_env({"ANTHROPIC_API_KEY": "sk-secret", "HF_TOKEN": "hf_x"})
    assert set(s.keys) == {"ANTHROPIC_API_KEY"} and "sk-secret" not in json.dumps(s.public())


def test_pricing_reads_each_wire():
    assert llm_pricing.tokens({"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5}) == \
        {"input": 10, "output": 5, "cache_read": 90, "cache_write": 0}
    assert llm_pricing.tokens({"prompt_tokens": 100, "completion_tokens": 7}) == {"input": 100, "output": 7, "cache_read": 0, "cache_write": 0}
    assert llm_pricing.usd("anthropic/claude-opus-5-5", {"input_tokens": 1_000_000}) == pytest.approx(4.0)


def test_a_provider_failure_fails_the_request():
    import httpx

    settings = llm_session.Settings.from_env({"ANTHROPIC_API_KEY": "bad"})
    deny = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": {"type": "authentication_error"}}))
    pol = serve.AgentPolicy(settings, lambda s: llm_session.make_policy(s, transport=deny))
    with pytest.raises(llm_session.AgentFailure, match="rejected"):
        pol.infer(Libero().request(), "c1")


# ------------------------------------------------------------------ over the wire
def test_server_over_the_wire():
    from websockets.sync.client import connect

    import pgs_msgpack_numpy as m

    pol, _ = policy()
    server = serve.AgentServer(pol, host="127.0.0.1", port=0).start()
    try:
        sim = Libero()
        with connect(f"ws://127.0.0.1:{server.port}/ws", max_size=None) as ws:
            meta = m.unpackb(ws.recv())
            assert meta["playground_serve"]["agent"] is True and meta["image_mode"] == "jpeg"
            assert "keys" not in json.dumps(meta["playground_serve"]["agent_settings"])
            for _ in range(3):
                ws.send(m.packb(sim.request("libero_franka")))
                reply = m.unpackb(ws.recv())
                assert reply["actions"].shape == (5, 7) and reply["agent"]["card"] == "libero_franka"
            bad = sim.request({"card": 1})
            ws.send(m.packb(bad))
            assert "CardError" in ws.recv()
    finally:
        server.stop()


def test_bundle_ships_the_family():
    files = bundle.worker_files("llm_agent")
    assert {"serve.py", "llm_card.py", "llm_session.py", "llm_pricing.py", "cards.json", "requirements.txt", "start.sh",
            "pgs_server.py", "pgs_msgpack_numpy.py", "server_decode.py", "health.py"} <= set(files)
    assert len(bundle.pack(files)) < 60_000
