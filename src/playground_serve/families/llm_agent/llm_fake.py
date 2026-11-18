"""A fake model provider for tests and dry runs: an httpx transport that answers the agent's requests with a recorded
sequence of tool calls, in the wire format the request came in. No network, no key.

    {"turns": [{"tool": "move_to", "arguments": {"targets": {"z": 1.05}, "note": "lift"}},
               {"text": "a turn without a tool call"},
               {"tool": "done", "arguments": {"summary": "placed", "hindsight": "none"}}],
     "usage": {"input": 1200, "output": 80}}

Each request gets the next turn; after the last one every request gets `done`. Wires: messages (Anthropic's
/messages), chat (/chat/completions) and responses (/responses). Every request body is kept in `requests`, image data
replaced by its length, so a test can check what the model was sent.
"""

from __future__ import annotations

import copy
import json
import pathlib
import threading
from typing import Any

import httpx

DONE = {"tool": "done", "arguments": {"summary": "the recorded turns are over", "hindsight": "none"}}


class FakeProvider:
    def __init__(self, script: dict | str | pathlib.Path) -> None:
        if not isinstance(script, dict):
            script = json.loads(pathlib.Path(script).read_text())
        self.turns: list[dict] = list(script.get("turns") or [])
        u = script.get("usage") or {}
        self.usage_in, self.usage_out = int(u.get("input", 1000)), int(u.get("output", 60))
        self.cost = u.get("cost")  # OpenRouter's usage.cost per response, when the script gives one
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.n = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _next(self) -> tuple[dict, int]:
        with self.lock:
            turn = self.turns[self.n] if self.n < len(self.turns) else DONE
            self.n += 1
            return turn, self.n

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(_strip(body))
        if turn_error := (self.turns[self.n].get("error") if self.n < len(self.turns) else None):
            self.n += 1
            return httpx.Response(turn_error.get("status", 200), json={"error": {"code": turn_error.get("code", 502), "message": turn_error.get("message", "provider failed")}})
        turn, n = self._next()
        path = request.url.path
        if path.endswith("/messages"):
            if "tool" in turn:
                content = [{"type": "tool_use", "id": f"toolu_{n:04d}", "name": turn["tool"], "input": turn.get("arguments", {})}]
                stop = "tool_use"
            else:
                content, stop = [{"type": "text", "text": turn.get("text", "")}], "end_turn"
            return httpx.Response(200, json={"id": f"msg_{n:04d}", "type": "message", "role": "assistant", "model": body.get("model"),
                                             "content": content, "stop_reason": stop,
                                             "usage": {"input_tokens": self.usage_in, "output_tokens": self.usage_out,
                                                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}})
        if path.endswith("/chat/completions"):
            message: dict[str, Any] = {"role": "assistant", "content": turn.get("text")}
            if "tool" in turn:
                message["tool_calls"] = [{"id": f"call_{n:04d}", "type": "function",
                                          "function": {"name": turn["tool"], "arguments": json.dumps(turn.get("arguments", {}))}}]
            return httpx.Response(200, json={"id": f"chatcmpl-{n}", "object": "chat.completion", "model": body.get("model"),
                                             "choices": [{"index": 0, "message": message,
                                                          "finish_reason": "tool_calls" if "tool" in turn else "stop"}],
                                             "usage": {"prompt_tokens": self.usage_in, "completion_tokens": self.usage_out,
                                                       **({"cost": self.cost} if self.cost is not None else {})}})
        if path.endswith("/responses"):
            output = ([{"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n:04d}", "name": turn["tool"],
                        "arguments": json.dumps(turn.get("arguments", {})), "status": "completed"}] if "tool" in turn else
                      [{"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": turn.get("text", ""), "annotations": []}]}])
            return httpx.Response(200, json={"id": f"resp_{n}", "object": "response", "status": "completed", "model": body.get("model"),
                                             "output": output, "usage": {"input_tokens": self.usage_in, "output_tokens": self.usage_out,
                                                                         "total_tokens": self.usage_in + self.usage_out}})
        return httpx.Response(404, json={"error": f"the fake provider has no {path}"})


def _strip(body: Any) -> Any:
    """A request body with every image's data replaced by its length."""
    body = copy.deepcopy(body)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k in ("data", "url", "image_url") and isinstance(v, str) and len(v) > 200:
                    node[k] = f"<{len(v)} chars>"
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node

    return walk(body)
