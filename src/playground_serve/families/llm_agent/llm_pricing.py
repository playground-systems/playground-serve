"""Token prices and the cost of an agent's usage, for the spend caps and the usage each reply reports.

PRICES are list prices in USD per million tokens: input, output, cache read, cache write (5-minute). Claude's were read
2026-11-16 from Anthropic's model table (cache writes are 1.25 times input; Claude Haiku 5.5's cache read is not
listed there and is taken as a tenth of input, as for the other models). Haiku 5.5's prices hold for prompts up to
100K tokens; the server compacts nothing, and an episode's prompt stays far below that at LIBERO's image sizes. A model
missing here has no price: the server reports its tokens and no USD, and refuses a USD cap for it unless the endpoint
gives PGS_AGENT_PRICE ("input,output[,cache_read,cache_write]" per million tokens).

Usage keys differ by wire: the Messages API reports input_tokens (uncached), cache_read_input_tokens,
cache_creation_input_tokens and output_tokens; Chat Completions and Responses report prompt_tokens or input_tokens
(cached included) and completion_tokens or output_tokens.
"""

from __future__ import annotations

import os
from typing import Mapping

PRICES: dict[str, tuple[float, float, float, float]] = {
    "anthropic/claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "anthropic/claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "anthropic/claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
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


def usd(model: str, usage: Mapping[str, float], env: Mapping[str, str] | None = None) -> float | None:
    p = price(model, env)
    if p is None:
        return None
    t = tokens(usage)
    return (t["input"] * p[0] + t["output"] * p[1] + t["cache_read"] * p[2] + t["cache_write"] * p[3]) / 1e6
