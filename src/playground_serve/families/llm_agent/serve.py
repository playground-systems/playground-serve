"""The LLM agent worker: a language model as a robot policy, behind Playground Serve's websocket wire (pgs_server.py).

A simulator connects as to any Playground Serve model: one metadata frame, then one request (an observation map) and one
reply ({"actions": (chunk, action_size)}) at a time. Behind it, each episode runs Inspect Robots' LLM agent policy
(inspect-robots-agent, llm_session.py) on the episode's embodiment card (llm_card.py). The server needs no GPU.

Request keys, besides the observation in the card's own keys (its cameras, its native state and its prompt):
    "embodiment"  the card: a map, its JSON text, or a built-in card's name (cards.json). Send it with every request,
                  or at least with an episode's first; without one the server looks for a built-in card whose keys the
                  request has (LIBERO's openpi layout: libero_franka).
    "episode"     an id that is new for every episode (any string). Episode state is kept by this id, so a client that
                  reconnects continues its episode. Without it, an episode is the connection and its prompt: a new
                  prompt on the same connection starts a new episode.
    "max_steps"   optional: the episode's step limit, which the model is told.
A request whose images are all black is a warm-up: it gets a holding chunk, and no episode starts.

Reply: "actions" float32 (chunk, action_size) in the card's native layout, and "agent": {episode, card, model, calls,
tokens {input, output, cache_read, cache_write}, usd (null for a model with no price), model_seconds, steps,
finished {kind, reason} once the agent called done or give_up or a cap stopped it, error, and turns: the transcript
messages added by this request, without images}. A model call happens only when the last motion has played and
settled, so most replies take no model time. A provider failure after the agent's retries (a rejected key, an outage)
ends the request with the server's error frame; the client's run then fails as the policy's error.

Metadata: {"playground_serve": {..., "agent": true, "agent_settings": the settings without keys}, "image_mode": "jpeg",
"image_size": "native"}: the client sends each camera at its own size as JPEG.

Endpoint env (llm_session.Settings.from_env): PGS_AGENT_MODEL (provider/model, as Inspect Robots names it, for example
anthropic/claude-haiku-5-5), PGS_AGENT_WIRE, PGS_AGENT_BASE_URL and PGS_AGENT_API_KEY_ENV (an OpenAI-compatible
server), PGS_AGENT_EFFORT, PGS_AGENT_MAX_CALLS, PGS_AGENT_MAX_USD_EPISODE, PGS_AGENT_MAX_USD, PGS_AGENT_IMAGES,
PGS_AGENT_IMAGE_HORIZON, PGS_AGENT_NOTES, PGS_AGENT_PRICE, and the provider's key (ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY, OPENROUTER_API_KEY, or the variable PGS_AGENT_API_KEY_ENV names). Nothing here logs a prompt, an image
or a key: the log lines hold counts.

    PYTHONPATH=<this folder>:<families/common>:<worker> python serve.py --port 8000 [--fake script.json]
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import itertools
import logging
import os
import pathlib
import threading
import time
import traceback

import numpy as np

import llm_card
import llm_session
import pgs_server
from pgs_server import ConnectionClosed, CloseCode, log, msgpack_numpy

logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO line names each request's URL; nothing of ours needs it
IDLE_S = 1800  # an episode untouched this long is dropped
WORKERS = 32  # requests served at once (each waits on its model call, not on this CPU)


def warm_up(request: dict) -> bool:
    images = [v for v in _walk(request) if isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8]
    return bool(images) and all(int(v.max()) == 0 for v in images)


def _walk(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    else:
        yield node


class AgentPolicy:
    """The policy object the server calls, with the connection's id: sessions by episode, settings from the env."""

    def __init__(self, settings: llm_session.Settings, factory=llm_session.make_policy) -> None:
        self.settings, self.factory = settings, factory
        self.sessions: dict[str, llm_session.Session] = {}
        self.lock = threading.Lock()
        self.ledger = llm_session.Ledger()
        self.metadata = {"playground_serve": {"agent": True, "agent_settings": settings.public()},
                         "image_mode": "jpeg", "image_size": "native"}
        self.last_card: dict[str, llm_card.Card] = {}

    def _card(self, request: dict, conn: str) -> llm_card.Card:
        given = request.get("embodiment")
        if given is not None:
            card = llm_card.load(given)
        else:
            card = self.last_card.get(conn) or llm_card.guess(request)
            if card is None:
                raise llm_card.CardError("the request has no embodiment card and matches no built-in card: send "
                                         "\"embodiment\" (a card, or a built-in card's name: "
                                         + ", ".join(sorted(llm_card.builtin())) + ")")
        self.last_card[conn] = card
        return card

    def infer(self, request: dict, conn: str) -> dict:
        card = self._card(request, conn)
        if warm_up(request):
            x = llm_session.Driver(card).state(request)
            rows = np.stack([llm_session.Driver(card).hold_row(x)] * card.chunk)
            return {"actions": rows.astype(np.float32), "agent": {"warm_up": True, "card": card.name}}
        episode = request.get("episode")
        prompt = str(llm_card.lookup(request, card.prompt) or "")
        key = f"e:{episode}" if episode not in (None, "") else f"c:{conn}:{prompt}"
        with self.lock:
            self._expire()
            session = self.sessions.get(key)
            if session is None or session.card.raw != card.raw:
                session = self.sessions[key] = llm_session.Session(key, card, self.settings, self.ledger, self.factory)
                log("episode_start", card=card.name, keyed="episode" if key.startswith("e:") else "connection")
        reply = session.step(request)
        if reply["agent"].get("turns"):
            a = reply["agent"]
            log("agent_call", calls=a["calls"], tokens=a["tokens"], usd=a["usd"], finished=(a.get("finished") or {}).get("kind"))
        return reply

    def _expire(self) -> None:
        now = time.monotonic()
        for k in [k for k, s in self.sessions.items() if now - s.touched > IDLE_S]:
            del self.sessions[k]

    def close_connection(self, conn: str) -> None:
        with self.lock:
            for k in [k for k in self.sessions if k.startswith(f"c:{conn}:")]:
                del self.sessions[k]
            self.last_card.pop(conn, None)


class AgentServer(pgs_server.Server):
    """pgs_server.Server with the connection's id passed to the policy, and many requests at once."""

    _ids = itertools.count(1)

    def __init__(self, policy: AgentPolicy, host: str = "0.0.0.0", port: int = 8000) -> None:
        super().__init__(policy, host, port, executor=concurrent.futures.ThreadPoolExecutor(WORKERS, thread_name_prefix="agent"))

    def _infer_on(self, msg: bytes, conn: str) -> dict:
        t = time.monotonic()
        request = pgs_server.decode_images(msgpack_numpy.unpackb(msg))
        decoded = time.monotonic()
        result = self.policy.infer(request, conn)
        done = time.monotonic()
        return {**result, "server_timing": {"infer_ms": (done - t) * 1000, "decode_ms": (decoded - t) * 1000}}

    async def _handler(self, ws):
        conn = str(next(self._ids))
        opened, path, noops, prev_total = time.time(), ws.request.path, 0, None
        log("ws_open", path=path, conn=conn)
        await ws.send(msgpack_numpy.packb(self.metadata))
        try:
            while True:
                msg = await ws.recv()
                start = time.monotonic()
                if pgs_server.is_noop(msg):
                    await ws.send(pgs_server.NOOP_REPLY)
                    noops += 1
                    continue
                try:
                    reply = await asyncio.wrap_future(self._pool.submit(self._infer_on, msg, conn))
                except Exception as e:
                    # The traceback names the cause; the server log gets its type only (a message may quote a prompt).
                    log("request_failed", conn=conn, error=type(e).__name__)
                    await ws.send(traceback.format_exc())
                    await ws.close(CloseCode.INTERNAL_ERROR, "Internal server error. Traceback included in previous frame.")
                    return
                if prev_total is not None:
                    reply["server_timing"]["prev_total_ms"] = prev_total * 1000
                await ws.send(msgpack_numpy.packb(reply))
                prev_total = time.monotonic() - start
        except ConnectionClosed:
            pass
        finally:
            self.policy.close_connection(conn)
            log("ws_close", path=path, conn=conn, s=round(time.time() - opened, 1), code=ws.close_code, noops=noops)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--fake", help="a recorded tool-call script (llm_fake.py): no model, no key")
    args = parser.parse_args()
    env = dict(os.environ)
    factory = llm_session.make_policy
    if args.fake:
        import llm_fake

        fake = llm_fake.FakeProvider(pathlib.Path(args.fake))
        env.setdefault("PGS_AGENT_MODEL", "anthropic/claude-haiku-5-5")
        env["ANTHROPIC_API_KEY"] = env["OPENAI_API_KEY"] = "fake"  # the transport below never sends a request out
        factory = lambda s: llm_session.make_policy(s, transport=fake.transport())  # noqa: E731
    settings = llm_session.Settings.from_env(env)
    if not settings.keys and not settings.base_url:
        raise SystemExit("no provider key in the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, "
                         "OPENROUTER_API_KEY, or the variable PGS_AGENT_API_KEY_ENV names)")
    log("process_start", settings=settings.public(), fake=bool(args.fake))
    server = AgentServer(AgentPolicy(settings, factory), args.host, args.port).start()
    pathlib.Path(pgs_server.READY_FILE).touch()
    log("ready", port=server.port)
    threading.Event().wait()


if __name__ == "__main__":
    main()
