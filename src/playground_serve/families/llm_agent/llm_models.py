"""The agent's models through OpenRouter (https://openrouter.ai/api/v1, one key for every provider): each model's
OpenRouter slug, title, list prices and the request settings this harness sends, and the transport that adds them.

The nine models are the hosted service's offer (Playground Bench's LLM_SERVICE_MODELS holds the same ids and prices).
Read 2026-11-17 from OpenRouter's public model list (GET /api/v1/models: every one takes image input, `tools`,
`tool_choice` and `reasoning`; none lists `parallel_tool_calls`) and its docs:
- reasoning: one `reasoning` object, `effort` none, minimal, low, medium, high, xhigh or max. OpenAI models take it as a
  share of the output limit, Claude as a thinking budget (max_tokens times 0.5 at medium, 0.8 at high; max_tokens must
  stay above it), Gemini 3 and later as its thinking level (xhigh becomes high). Reasoning comes back in the message's
  `reasoning` and `reasoning_details`, which the agent replays unchanged with its tool calls.
- caching: Claude caches only at `cache_control` breakpoints (a top-level `cache_control` caches automatically; writes
  1.25 times input, reads 0.1 times); OpenAI, Gemini 2.5 and later, and Moonshot cache by themselves (reads 0.1 to 0.5 times
  input); Meta's is not documented. Reads and writes are reported in usage.prompt_tokens_details.
- usage: every response's `usage` carries `cost` in credits (USD) without asking; `usage: {include: true}` is
  deprecated. The server's USD caps use that reported cost, and the price table only when a response has none.
- routing: `provider.require_parameters` true sends a request only to a provider that takes every parameter in it
  (`tools`, `reasoning`, `max_tokens`); `data_collection` "deny" excludes providers that may store and train on prompts.
- errors: {"error": {"code", "message", "metadata"}} with the HTTP status of the code: 401 key, 402 credit, 403
  moderation (metadata.reasons), 408 timeout, 429 rate limit, 502 the model or its provider failed, 503 no provider meets
  the routing; an error may also come with HTTP 200 as a body with `error` and no `choices`.

Prices are USD per million tokens: input, output, cache read, cache write (where OpenRouter lists none, input).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

BASE_URL = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class OpenRouterModel:
    slug: str
    title: str  # as the service's users name it
    prices: tuple[float, float, float, float]
    effort: str = "medium"  # the `reasoning.effort` sent
    max_tokens: int = 16000  # output limit per call (thinking included)
    cache: bool = False  # add a top-level cache_control (Claude caches only at breakpoints)
    data_collection: str = "deny"
    note: str = ""


MODELS: dict[str, OpenRouterModel] = {m.slug: m for m in (
    OpenRouterModel("anthropic/claude-fable-5.2", "Fable 5.2", (8.00, 40.00, 0.20, 10.00), cache=True),
    OpenRouterModel("anthropic/claude-opus-5.5", "Opus 5.5", (4.00, 20.00, 0.20, 5.00), cache=True),
    OpenRouterModel("google/gemini-4-carbon", "Gemini 4 Carbon", (2.00, 10.00, 0.10, 0.1875)),
    OpenRouterModel("google/gemini-4-argon", "Gemini 4 Argon", (2.00, 10.00, 0.10, 0.1875)),
    OpenRouterModel("google/gemini-3.8-flash", "Gemini 3.8 Flash (high)", (0.825, 4.125, 0.0825, 0.0417), effort="high"),
    OpenRouterModel("openai/gpt-6-astra", "GPT-6 Astra", (10.00, 50.00, 1.00, 12.50)),
    OpenRouterModel("openai/gpt-6.1-sol", "GPT 6.1 Sol", (2.00, 10.00, 0.10, 2.50)),
    OpenRouterModel("meta/muse-wonder", "Muse Wonder 1.0", (4.00, 20.00, 0.20, 4.00)),
    OpenRouterModel("moonshotai/kimi-k3.1-pro", "Kimi 3.1", (0.2403, 8.4023, 0.20, 0.2403),
                    note="OpenRouter lists Kimi K3.1 Pro only; 'Kimi 3.1' is served as it"),
)}

# OpenRouter's error classes by HTTP status, for the failure the server raises.
ERROR_CLASSES = {401: "key", 402: "credit", 403: "moderation", 408: "timeout", 429: "rate_limit", 502: "provider",
                 503: "no_provider"}


def error_class(text: str) -> str:
    """The class of an agent wire failure ("LLM request ... HTTP 402: ..."), from its HTTP status."""
    for code, name in ERROR_CLASSES.items():
        if f"HTTP {code}" in text:
            return name
    return "provider"


class OpenRouterTransport:
    """An httpx transport for requests to OpenRouter: it adds this harness's settings of the model to each chat request
    (the `reasoning` object in place of the agent's `reasoning_effort`, the output limit, the routing, Claude's cache
    breakpoint) and keeps the last error that came with HTTP 200, then passes the request on."""

    def __init__(self, model: OpenRouterModel, inner: Any) -> None:
        self.model, self.inner = model, inner
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def handle_request(self, request):
        import httpx

        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content or b"{}")
            m = self.model
            body["reasoning"] = {"effort": body.pop("reasoning_effort", None) or m.effort}
            body.setdefault("max_tokens", m.max_tokens)
            body["provider"] = {"require_parameters": True, "data_collection": m.data_collection, "allow_fallbacks": True}
            if m.cache:
                body["cache_control"] = {"type": "ephemeral"}
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("content-length",)}
            request = httpx.Request(request.method, request.url, headers=headers, content=json.dumps(body).encode())
        response = self.inner.handle_request(request)
        if response.status_code == 200 and "json" in response.headers.get("content-type", "application/json"):
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict) and not payload.get("choices"):
                e = payload["error"]
                with self.lock:
                    self.last_error = f"HTTP {e.get('code', 502)}: {str(e.get('message'))[:300]}"
        return response

    def close(self) -> None:
        self.inner.close()
