"""`playground_serve.status`: warm, cold or missing from Runpod's worker counts, through a fake HTTP layer that records
every request: one GET of the health URL with the bearer key, nothing that reaches a worker."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from playground_serve import resolve, state
from playground_serve.resolve import COLD_WAIT_S, URL_ENV, WARM_WAIT_S, EndpointNotFound, Status, status

KEY = "test-key-0123456789"
HEALTH = "https://api.runpod.ai/v2/{}/health"
ACCOUNT = "https://api.runpod.io/v2/serverless"


def counts(**kw) -> dict:
    return {"idle": 0, "initializing": 0, "ready": 0, "running": 0, "throttled": 0, "unhealthy": 0, **kw}


class FakeResponse:
    def __init__(self, code: int, raw: bytes) -> None:
        self.status, self._raw = code, raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


class FakeHttp:
    """Stands in for urllib.request.urlopen: answers by URL, records (method, url, headers) of every request."""

    def __init__(self, answers: dict) -> None:
        self.answers, self.requests = answers, []

    def __call__(self, request, timeout=None):
        self.requests.append((request.get_method(), request.full_url, dict(request.header_items())))
        code, body = self.answers.get(request.full_url, (599, b"not a URL this test expects"))
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        if code >= 400:
            raise urllib.error.HTTPError(request.full_url, code, "error", {}, io.BytesIO(raw))
        return FakeResponse(code, raw)

    @property
    def urls(self) -> list:
        return [url for _, url, _ in self.requests]


@pytest.fixture
def http(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    fake = FakeHttp({})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def in_state(model: str, endpoint_id: str) -> None:
    state.save(state.default_path(), {"models": {model: {"endpoint": {"id": endpoint_id, "name": "n",
                                                                     "url": f"wss://{endpoint_id}.api.runpod.ai/ws"}}}})


def test_warm_when_a_worker_runs(http):
    in_state("pi05_libero", "ep1")
    http.answers[HEALTH.format("ep1")] = (200, {"jobs": {}, "workers": counts(running=1)})
    s = status("pi05_libero")
    assert s == Status("warm", WARM_WAIT_S, counts(running=1), "ep1", "wss://ep1.api.runpod.ai/ws")
    (method, url, headers), = http.requests  # one request, and not to the gateway (which would start a worker)
    assert (method, url) == ("GET", HEALTH.format("ep1")) and headers["Authorization"] == f"Bearer {KEY}"
    assert headers["User-agent"].startswith("playground-serve/")


@pytest.mark.parametrize("workers", [counts(idle=1, ready=1), counts(throttled=1), counts(initializing=1), {}])
def test_cold_when_none_runs(http, workers):
    http.answers[HEALTH.format("ep1")] = (200, {"workers": workers})
    s = status("wss://ep1.api.runpod.ai/ws")
    assert (s.state, s.wait_s, s.workers, s.endpoint_id) == ("cold", COLD_WAIT_S, workers, "ep1")
    assert http.urls == [HEALTH.format("ep1")]


def test_endpoint_only_key_with_the_env_url(http, monkeypatch):
    """A key that may only call the endpoint: the account refuses every read, the health call answers."""
    http.answers[ACCOUNT] = (403, {"detail": "you do not have permission to perform this action"})
    http.answers[HEALTH.format("ep9")] = (200, {"workers": counts(idle=1, ready=1)})
    monkeypatch.setenv(URL_ENV, "wss://ep9.api.runpod.ai/ws")
    assert status("pi05_libero").state == "cold"
    assert http.urls == [HEALTH.format("ep9")]


def test_endpoint_only_key_without_url_says_to_pass_it(http):
    http.answers[ACCOUNT] = (403, {"detail": "you do not have permission to perform this action"})
    with pytest.raises(EndpointNotFound, match=r"HTTP 403.*PLAYGROUND_SERVE_URL"):
        status("pi05_libero")
    assert http.urls == [ACCOUNT]  # one refused account read, no health call


def test_missing(http):
    http.answers[ACCOUNT] = (200, {"endpoints": [{"id": "d1", "name": "playground-serve-pi05-droid-cccccc"}]})
    assert status("pi05_libero") == Status("missing", None, {})
    assert http.urls == [ACCOUNT]
    http.answers[HEALTH.format("gone1")] = (404, {"detail": "not found"})
    s = status("wss://gone1.api.runpod.ai/ws")
    assert (s.state, s.wait_s, s.endpoint_id) == ("missing", None, "gone1")


def test_found_on_the_account(http):
    http.answers[ACCOUNT] = (200, {"endpoints": [{"id": "lib1", "name": "playground-serve-pi05-libero-aaaaaa"}]})
    http.answers[HEALTH.format("lib1")] = (200, {"workers": counts(running=1)})
    assert status("pi05_libero").state == "warm"
    assert http.urls == [ACCOUNT, HEALTH.format("lib1")]


def test_errors(http, monkeypatch):
    http.answers[HEALTH.format("ep1")] = (403, {"detail": "forbidden"})
    with pytest.raises(PermissionError, match="may not call this endpoint"):
        status("wss://ep1.api.runpod.ai/ws")
    http.answers[HEALTH.format("ep1")] = (200, b"<html>")
    with pytest.raises(ConnectionError, match="unexpected answer"):
        status("wss://ep1.api.runpod.ai/ws")
    with pytest.raises(ValueError, match="not a Runpod endpoint URL"):
        status("ws://127.0.0.1:8000")
    monkeypatch.delenv("RUNPOD_API_KEY")
    with pytest.raises(ValueError, match="no Runpod API key"):
        status("wss://ep1.api.runpod.ai/ws")
    assert http.urls == [HEALTH.format("ep1")] * 2


def test_api_key_argument_and_export(http):
    http.answers[HEALTH.format("ep1")] = (200, {"workers": counts()})
    status("wss://ep1.api.runpod.ai/ws", api_key="other-key")
    assert http.requests[0][2]["Authorization"] == "Bearer other-key"
    import playground_serve

    assert playground_serve.status is resolve.status
