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
            r["max_steps"] = 220
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
    assert first["output_config"] == {"effort": "medium"} and "temperature" not in first and "tool_choice" not in first
    system = first["system"][0]["text"]
    assert "Franka Panda" in system and llm_session.DEFAULT_NOTES in system  # the card's docs, the default notes
    assert "Step budget: step 0/220" in json.dumps(first["messages"])  # max_steps from the request
    assert {t["name"] for t in first["tools"]} == {"move_to", "sleep", "zoom", "done", "give_up"}
    assert "state[gripper_holding]" in json.dumps(first["messages"])
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
    ("openai/gpt-6-luna", {"PGS_AGENT_WIRE": "chat"}, "/chat/completions"),
    ("openai/gpt-5.3-codex", {}, "/responses"),  # OpenAI's default wire here
    ("qwen/qwen3-vl", {"PGS_AGENT_BASE_URL": "https://vllm.example.com/v1", "PGS_AGENT_API_KEY_ENV": "OPENAI_API_KEY",
                       "PGS_AGENT_MAX_USD_EPISODE": "none"}, "/chat/completions"),
])
def test_other_wires(model, env, path):
    pol, fake = policy(env=env, wire_model=model)
    sim = Libero()
    replies = run(pol, sim, n=30)
    last = replies[-1]["agent"]
    assert last["finished"]["kind"] == "done"
    assert last["tokens"]["input"] == 3000 and (last["usd"] or 0) > 0 or model.startswith("qwen")  # metered on every wire
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
    assert "c:c3:0:pick up the black bowl" in pol.sessions
    pol.close_connection("c3")
    assert "c:c3:0:pick up the black bowl" not in pol.sessions


def test_warm_up_request_calls_no_model():
    pol, fake = policy()
    r = Libero(episode=None).request()
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
        llm_session.Settings.from_env({"PGS_AGENT_BASE_URL": "http://example.com/v1"})
    s = llm_session.Settings.from_env({"ANTHROPIC_API_KEY": "sk-secret", "HF_TOKEN": "hf_x"})
    assert set(s.keys) == {"ANTHROPIC_API_KEY"} and "sk-secret" not in json.dumps(s.public())


def test_pricing_reads_each_wire():
    assert llm_pricing.wire_tokens({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 800}, "output_tokens": 50}) == \
        {"input": 200, "output": 50, "cache_read": 800, "cache_write": 0}
    assert llm_pricing.wire_tokens({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}, "completion_tokens": 7}) == \
        {"input": 40, "output": 7, "cache_read": 60, "cache_write": 0}
    assert llm_pricing.usd_of("openai/gpt-6-luna", {"input": 1_000_000, "output": 0, "cache_read": 1_000_000, "cache_write": 0}) == pytest.approx(0.11)
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


# ------------------------------------------------------------------ custom environments' cards
PANDA_ENV_CARD = {
    "version": 1, "environment": {"id": "env_1", "revision": 3, "hash": "abc", "name": "mug shelf"},
    "task": {"number": 1, "instruction": "put the mug on the shelf", "seed": 7, "episode": 0},
    "robot": {"name": "panda", "arms": 1, "arm": "Franka Panda, 7 joints", "gripper": "Franka hand", "note": ""},
    "frame": "world: metres, z up, the table top at z = 0.75", "controlHz": 20,
    "action": {"layout": "eef_abs_pose_rot6d", "size": 10, "labels": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"],
               "low": [0.15, -0.55, 0.755, -1, -1, -1, -1, -1, -1, 0], "high": [1.05, 0.55, 1.45, 1, 1, 1, 1, 1, 1, 1],
               "maxStep": [0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25], "chunk": "the rows run in order"},
    "observation": {"cameras": [{"name": "agentview", "width": 64, "height": 64, "about": "the reference camera"},
                                {"name": "wrist", "width": 64, "height": 64, "about": "on the hand"}],
                    "state": [{"name": "eef_state", "size": 10, "labels": []}, {"name": "gripper_qpos", "size": 2, "labels": []}],
                    "prompt": "the task's instruction"},
    "limits": {"maxSteps": 1100, "seconds": 60}, "success": ["the mug rests on the shelf"],
}


def test_custom_environment_card_and_reset():
    card = llm_card.from_env_card(json.dumps(PANDA_ENV_CARD))
    assert card.kind == "same" and card.chunk == 10 and card.max_steps == 1100 and card.native["state"] == "state/eef_state"
    assert "the mug rests on the shelf" in card.docs and card.cameras == {"agentview": "images/agentview", "wrist": "images/wrist"}
    pol, fake = policy(turns=[{"tool": "move_to", "arguments": {"targets": {"z": 0.9}, "note": "lower"}}] * 4)
    rng = np.random.default_rng(0)
    pose = np.array([0.5, 0.0, 1.0, 1, 0, 0, 0, -1, 0, 0.0])

    def req(first=False):
        r = {"images": {c: rng.integers(1, 255, (64, 64, 3), dtype=np.uint8) for c in ("agentview", "wrist")},
             "state": {"eef_state": pose.copy(), "gripper_qpos": np.array([0.04, -0.04])}, "prompt": "put the mug on the shelf"}
        return {**r, "embodiment_card": json.dumps(PANDA_ENV_CARD), "reset": True} if first else r

    out = pol.infer(req(True), "c1")
    assert out["actions"].shape == (10, 10) and np.all(np.diff(out["actions"][:, 2]) <= 0)  # absolute targets going down
    body = fake.requests[0]
    assert "Environment step budget" in body["system"][0]["text"] and "the mug rests on the shelf" in body["system"][0]["text"]
    assert "state[gripper_qpos]" in json.dumps(body["messages"])
    pol.infer(req(), "c1")
    pol.infer(req(True), "c1")  # reset: a new episode on the same connection
    assert sorted(pol.sessions) == ["c:c1:1:put the mug on the shelf", "c:c1:2:put the mug on the shelf"]


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


def test_custom_environment_two_arm_card():
    labels = [f"left_joint{j}" for j in range(1, 7)] + ["left_gripper"] + [f"right_joint{j}" for j in range(1, 7)] + ["right_gripper"]
    env = {**PANDA_ENV_CARD, "controlHz": 50,
           "action": {"layout": "aloha_joints", "size": 14, "labels": labels, "low": [-3.2] * 6 + [0] + [-3.2] * 6 + [0],
                      "high": [3.2] * 6 + [1] + [3.2] * 6 + [1], "maxStep": None, "chunk": "the first 25 rows run, one per control step"},
           "observation": {"cameras": [{"name": "cam_high", "width": 224, "height": 224, "about": ""}],
                           "state": [{"name": "joint_pos", "size": 14, "labels": labels}], "prompt": ""}}
    card = llm_card.from_env_card(env)
    assert card.agent["control_mode"] == "joint_pos" and card.chunk == 25 and card.native["state_field"] == "joint_pos"
    info = llm_session.embodiment_info(card, {"cam_high": (224, 224)})
    assert info.action_space.semantics.control_mode == "joint_pos" and info.control_hz == 50


DATA = pathlib.Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize("name,size,chunk,mode", [("env_card_panda", 10, 10, "eef_abs_pose"), ("env_card_yam", 14, 25, "joint_pos")])
def test_the_custom_environments_cards(name, size, chunk, mode):
    """The cards that Playground Bench's custom environments make for their two robots (cloud/envs/card.ts
    embodimentCard, written to tests/families/data): an episode answers absolute rows inside the card's box."""
    env = json.loads((DATA / f"{name}.json").read_text())
    card = llm_card.from_env_card(env)
    assert (card.action_size, card.chunk, card.agent["control_mode"], card.max_steps) == (size, chunk, mode, env["limits"]["maxSteps"])
    target = {"z": 0.9} if mode == "eef_abs_pose" else {"left_joint1": 0.3}
    pol, fake = policy(turns=[{"tool": "move_to" if mode == "eef_abs_pose" else "move_joints", "arguments": {"targets": target, "note": "move"}}] * 3)
    rng = np.random.default_rng(0)
    cams = {c["name"]: rng.integers(1, 255, (c["height"], c["width"], 3), dtype=np.uint8) for c in env["observation"]["cameras"]}
    low, high = np.array(env["action"]["low"]), np.array(env["action"]["high"])
    if mode == "eef_abs_pose":
        state = {"eef_state": np.array([0.5, 0.0, 1.0, 1, 0, 0, 0, -1, 0, 0.0]), "gripper_qpos": np.array([0.04, -0.04])}
    else:
        state = {"joint_pos": np.clip(np.zeros(14), low, high)}
    out = pol.infer({"images": cams, "state": state, "prompt": "put the red block in the bowl", "embodiment_card": json.dumps(env), "reset": True}, "c1")
    rows = out["actions"]
    assert rows.shape == (chunk, size) and np.all(rows >= low - 1e-6) and np.all(rows <= high + 1e-6)
    assert fake.requests and "Step budget: step 0/" in json.dumps(fake.requests[0]["messages"])


def test_a_plain_scene_card():
    plain = {"version": 1, "controlHz": 50, "action": {"layout": "custom", "size": 2, "labels": ["force_x_n", "force_y_n"], "low": [-5, -5], "high": [5, 5], "chunk": ""},
             "observation": {"cameras": [{"name": "front", "width": 224, "height": 224}], "state": [{"name": "block_xy", "size": 2}]}}
    with pytest.raises(llm_card.CardError, match="absolute targets only"):
        llm_card.from_env_card(plain)
    absolute = {**plain, "action": {**plain["action"], "kind": "absolute", "labels": ["slider_x", "slider_y"]}}
    card = llm_card.from_env_card(absolute)
    assert card.agent["control_mode"] == "joint_pos" and card.native["state_field"] == "block_xy"


def test_zoom_and_motion_reports():
    turns = [{"tool": "zoom", "arguments": {"camera": "wrist", "box": [0.25, 0.4, 0.75, 0.9], "note": "check the rim"}},
             {"tool": "zoom", "arguments": {"camera": "wrist", "box": [0.5, 0.5, 0.52, 0.9], "note": "too thin"}},
             TURNS[0], TURNS[2]]
    pol, fake = policy(turns=turns)
    sim = Libero()
    big = sim.request()
    big["observation/wrist_image"] = np.random.default_rng(1).integers(1, 255, (448, 448, 3), dtype=np.uint8)
    first = pol.infer(sim.request(), "c1")
    assert np.abs(first["actions"][:, :6]).max() < 0.05 and first["agent"]["render"] == {"wrist": [448, 448]}
    sim.play(first["actions"])
    r = sim.request()
    r["observation/wrist_image"] = big["observation/wrist_image"]  # the client rendered the wrist larger, as asked
    r["motion_report"] = ["the arm held still"]
    second = pol.infer(r, "c1")
    body = json.dumps(fake.requests[1])
    assert "camera 'zoom'" in body and "the arm held still" in body
    assert "box must have" in body or "box must have" in json.dumps(fake.requests[2])  # the thin box was refused
    assert second["agent"]["zooms"] == 1
