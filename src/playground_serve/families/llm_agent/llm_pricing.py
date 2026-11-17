"""Token prices and the cost of an agent's usage, for the spend caps and the usage each reply reports.

PRICES are list prices in USD per million tokens: input, output, cache read, cache write (5-minute). Claude's were read
2026-11-16 from Anthropic's model table (cache writes are 1.25 times input; Claude Haiku 5.5's cache read is not
listed there and is taken as a tenth of input, as for the other models). Haiku 5.5's prices hold for prompts up to
100K tokens; the server compacts nothing, and an episode's prompt stays far below that at LIBERO's image sizes. A model
missing here has no price: the server reports its tokens and no USD, and refuses a USD cap for it unless the endpoint
gives PGS_AGENT_PRICE ("input,output[,cache_read,cache_write]" per million tokens).

OpenAI's were read 2026-11-16 from developers.openai.com/api/docs/pricing and each model's page: gpt-6-luna lists cache
writes at 0.125 (1.25 times input); for the others no cache-write price is listed and 1.25 times input is assumed.
gpt-5.3-codex (the Codex model of the API, deprecated there, Responses API only) has no explicit cache writes.

Usage keys differ by wire: the Messages API reports input_tokens (uncached), cache_read_input_tokens,
cache_creation_input_tokens and output_tokens; Chat Completions reports prompt_tokens (cached included),
prompt_tokens_details.cached_tokens and completion_tokens; Responses reports input_tokens (cached included),
input_tokens_details.cached_tokens and output_tokens (reasoning included). `wire_tokens` reads all three from the
provider's own usage object; the server meters each response that way (llm_session.Meter), whatever the agent keeps.
"""

from __future__ import annotations

import os
from typing import Mapping

PRICES: dict[str, tuple[float, float, float, float]] = {
    "anthropic/claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "anthropic/claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "anthropic/claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
    "openai/gpt-6-astra": (10.00, 50.00, 1.00, 12.50),
    "openai/gpt-6.1-sol": (2.00, 10.00, 0.10, 2.50),
    "openai/gpt-6.1-terra": (1.00, 5.00, 0.05, 1.25),
    "openai/gpt-6-luna": (0.10, 0.50, 0.01, 0.125),
    "openai/gpt-5.3-codex": (1.75, 14.00, 0.175, 1.75),
}
DEFAULT_MODELS = tuple(PRICES)  # the models the hosted service offers on its own key


def price(model: str, env: Mapping[str, str] | None = None) -> tuple[float, float, float, float] | None:
    given = (env if env is not None else os.environ).get("PGS_AGENT_PRICE", "").strip()
    if given:
        parts = [float(x) for x in given.split(",")]
        if len(parts) not in (2, 4) or any(p < 0 for p in parts):
            raise ValueError("PGS_AGENT_PRICE: write input,output[,cache_read,cache_write] in USD per million tokens")
        return (parts[0], parts[1], parts[2] if len(parts) == 4 else parts[0], parts[3] if len(parts) == 4 else parts[0])
    return PRICES.get(model)


def tokens(usage: Mapping[str, float]) -> dict[str, int]:
    """input (uncached), output, cache_read and cache_write tokens of one usage total, whatever the wire."""
    g = lambda k: int(usage.get(k) or 0)  # noqa: E731
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage or (
            "input_tokens" in usage and "prompt_tokens" not in usage and "input_tokens_details" not in usage):
        return {"input": g("input_tokens"), "output": g("output_tokens"), "cache_read": g("cache_read_input_tokens"),
                "cache_write": g("cache_creation_input_tokens")}
    prompt = g("prompt_tokens") or g("input_tokens")
    cached = g("cached_tokens") or g("prompt_tokens_details.cached_tokens")
    return {"input": max(0, prompt - cached), "output": g("completion_tokens") or g("output_tokens"),
            "cache_read": cached, "cache_write": 0}


def wire_tokens(raw: object) -> dict[str, int]:
    """input (uncached), output, cache_read and cache_write tokens of one provider response's usage object."""
    if not isinstance(raw, dict):
        return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    n = lambda v: int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0  # noqa: E731
    d = lambda k: raw.get(k) if isinstance(raw.get(k), dict) else {}  # noqa: E731
    if "cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw:  # Messages: input is uncached
        return {"input": n(raw.get("input_tokens")), "output": n(raw.get("output_tokens")),
                "cache_read": n(raw.get("cache_read_input_tokens")), "cache_write": n(raw.get("cache_creation_input_tokens"))}
    if "prompt_tokens" in raw:  # Chat Completions
        cached, write = n(d("prompt_tokens_details").get("cached_tokens")), n(d("prompt_tokens_details").get("cache_write_tokens"))
        return {"input": max(0, n(raw["prompt_tokens"]) - cached - write), "output": n(raw.get("completion_tokens")),
                "cache_read": cached, "cache_write": write}
    cached, write = n(d("input_tokens_details").get("cached_tokens")), n(d("input_tokens_details").get("cache_write_tokens"))
    return {"input": max(0, n(raw.get("input_tokens")) - cached - write), "output": n(raw.get("output_tokens")),
            "cache_read": cached, "cache_write": write}


def usd_of(model: str, t: dict[str, int], env: Mapping[str, str] | None = None) -> float | None:
    p = price(model, env)
    if p is None:
        return None
    return (t["input"] * p[0] + t["output"] * p[1] + t["cache_read"] * p[2] + t["cache_write"] * p[3]) / 1e6


def usd(model: str, usage: Mapping[str, float], env: Mapping[str, str] | None = None) -> float | None:
    p = price(model, env)
    if p is None:
        return None
    t = tokens(usage)
    return (t["input"] * p[0] + t["output"] * p[1] + t["cache_read"] * p[2] + t["cache_write"] * p[3]) / 1e6
