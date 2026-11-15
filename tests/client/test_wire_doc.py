"""docs/WIRE.md's constants table equals the code's constants, so the spec for a second client cannot drift."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from playground_serve import resolve
from playground_serve.client import client, codec
from playground_serve.client.mock_server import Config

WIRE = Path(__file__).resolve().parents[2] / "docs" / "WIRE.md"
DEFAULTS = {name: p.default for name, p in inspect.signature(client.Policy.__init__).parameters.items()}
CODE = {
    "OPEN_TIMEOUT_S": DEFAULTS["open_timeout"],
    "WAIT_S": DEFAULTS["wait"],
    "RETRY_STATUS": sorted(client.RETRY_STATUS),
    "RETRY_400_BODY": client.RETRY_400.decode(),
    "BACKOFF_S": list(client.BACKOFF_S),
    "RECV_TIMEOUT_S": DEFAULTS["recv_timeout"],
    "RETRIES": DEFAULTS["retries"],
    "KEEPALIVE_S": DEFAULTS["keepalive"],
    "NOOP": codec.unpackb(client.NOOP),
    "NOOP_TIMEOUT_S": client.NOOP_TIMEOUT,
    "FEATURE_KEYS": list(client.FEATURE_KEYS),
    "MODES": list(codec.MODES),
    "IMAGE_SIZE": client.IMAGE_SIZE,
    "JPEG_QUALITY": DEFAULTS["jpeg_quality"],
    "JPEG_KEY": codec.JPEG_KEY.decode(),
    "SPLIT_KEY": client.SPLIT_KEY,
    "SPLIT_PATH": client.SPLIT_PATH,
    "DIRECT_OPEN_TIMEOUT_S": client.DIRECT_OPEN_TIMEOUT,
    "HOT_QUERY": client.HOT_QUERY,
    "HEALTH_URL": resolve.HEALTH_URL,
    "COLD_WAIT_S": list(resolve.COLD_WAIT_S),
    "WARM_WAIT_S": list(resolve.WARM_WAIT_S),
    "URL_ENV": resolve.URL_ENV,
}


def parse(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return [parse(part.strip()) for part in text.split(",")] if "," in text else text


def documented() -> dict:
    rows = re.findall(r"^\| `(\w+)` \| `([^`]*)` \|", WIRE.read_text(), flags=re.M)
    return {name: parse(value) for name, value in rows}


def test_constants_table_matches_the_code():
    assert documented() == CODE


def test_every_constant_named_in_the_text_is_in_the_table():
    text = WIRE.read_text().split("## Constants")[0]
    named = set(re.findall(r"`([A-Z][A-Z0-9_]+)(?:\[\d\])?`", text))
    assert named <= set(CODE), named - set(CODE)


def test_short_and_the_mock_uses_the_current_feature_key():
    assert len(WIRE.read_text().splitlines()) < 120
    assert Config().metadata()[client.FEATURE_KEYS[0]] == {"jpeg": True, "noop": True}
