"""Token prices and the cost of an agent's usage, for the spend caps and the usage each reply reports.

The prices are OpenRouter's list prices of the nine agent models (llm_models.MODELS), USD per million tokens: input,
output, cache read, cache write. Each OpenRouter response also reports its own cost (usage.cost, in USD credits); the
server uses that where every response had it, and these prices only otherwise.

Usage keys: Chat Completions reports prompt_tokens (cached included), prompt_tokens_details.cached_tokens and
cache_write_tokens, and completion_tokens (reasoning included). `wire_tokens` also reads the Messages and Responses
shapes, so a recorded response of either counts the same way.
"""

from __future__ import annotations

from typing import Mapping

import llm_models


def price(model: str, env: Mapping[str, str] | None = None) -> tuple[float, float, float, float] | None:
    """USD per million tokens of an agent model (its registry name, or "openrouter/<name>")."""
    m = llm_models.MODELS.get(model.removeprefix("openrouter/"))
    return m.prices if m else None


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


def tokens(usage: Mapping[str, float]) -> dict[str, int]:
    """The agent's own summed usage (a policy made without the meter) in the same four counts."""
    return wire_tokens(dict(usage))


def usd_of(model: str, t: dict[str, int], env: Mapping[str, str] | None = None) -> float | None:
    p = price(model, env)
    if p is None:
        return None
    return (t["input"] * p[0] + t["output"] * p[1] + t["cache_read"] * p[2] + t["cache_write"] * p[3]) / 1e6
