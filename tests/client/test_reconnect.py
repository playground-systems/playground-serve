"""The client survives dropped sockets, a server restart and a server that is not up yet; auth is checked."""

import threading
import time

import pytest
from conftest import free_port, wait_for
from websockets.exceptions import InvalidStatus

from playground_serve.client import ServerError, connect
from playground_serve.client.mock_server import Config, MockServer


def test_reconnects_after_server_drops_socket(observations):
    cfg = Config(drop_every=2)  # every second request: abort the TCP connection without replying
    with MockServer(cfg) as mock, connect(mock.url) as client:
        for i in range(6):
            assert client.infer(observations[i % 4])["actions"].shape == (10, 7)
        assert client.reconnects == 5  # every call after the first lost its socket once and was resent
        assert cfg.requests == 11


def test_reconnects_after_server_restart(observations):
    port = free_port()
    with MockServer(port=port) as mock:
        client = connect(mock.url)
        client.infer(observations[0])
    with MockServer(port=port):
        client.infer(observations[1])
    assert client.reconnects == 1
    client.close()


def test_connect_waits_for_server(observations):
    port = free_port()
    mock = MockServer(port=port)
    timer = threading.Timer(1.0, mock.__enter__)
    timer.start()
    t = time.monotonic()
    try:
        with connect(f"ws://127.0.0.1:{port}", wait=60) as client:
            assert time.monotonic() - t >= 0.9
            client.infer(observations[0])
    finally:
        timer.join()
        mock.__exit__(None, None, None)


def test_server_error_then_recovers(observations):
    with MockServer() as mock, connect(mock.url) as client:
        with pytest.raises(ServerError, match="expected an array or a string"):
            client.infer({"bad": {"not": "an array"}})
        assert client.infer(observations[0])["actions"].shape == (10, 7)
        assert client.reconnects == 1


def test_bearer_auth(observations):
    with MockServer(cfg := Config(api_key="test-key")) as mock:
        with connect(mock.url, api_key="test-key") as client:
            client.infer(observations[0])
        with pytest.raises(InvalidStatus, match="401"):
            connect(mock.url)
        assert cfg.attempts == 2  # a 401 is not retried


FAILED_TO_ASSIGN = '{"status":400,"title":"Bad Request","detail":"failed to assign worker"}'


def test_retries_the_gateways_400_failed_to_assign_worker(observations):
    cfg = Config(upgrades=[(400, FAILED_TO_ASSIGN)])
    with MockServer(cfg) as mock, connect(mock.url) as client:
        assert client.infer(observations[0])["actions"].shape == (10, 7)
    assert cfg.attempts == 2


def test_any_other_400_is_not_retried():
    cfg = Config(upgrades=[(400, "Bad Request: no such route")])
    with MockServer(cfg) as mock:
        with pytest.raises(InvalidStatus, match="400"):
            connect(mock.url)
    assert cfg.attempts == 1


def test_a_held_upgrade_is_cut_at_open_timeout_and_retried_at_once(observations, monkeypatch):
    """The gateway holds an upgrade while the worker starts and may never hand it over (the mock holds it an hour):
    short attempts, no backoff."""
    from playground_serve.client import client as client_module

    pauses = []  # the client's backoff sleeps between attempts
    monkeypatch.setattr(client_module, "time", type("Clock", (), {
        "monotonic": staticmethod(time.monotonic), "perf_counter": staticmethod(time.perf_counter),
        "sleep": staticmethod(lambda s: pauses.append(s) or time.sleep(s))}))
    cfg = Config(upgrades=["hold", "hold"])
    with MockServer(cfg) as mock:
        t = time.monotonic()
        with connect(mock.url, open_timeout=0.5) as client:
            elapsed = time.monotonic() - t
            client.infer(observations[0])
    assert cfg.attempts == 3 and elapsed >= 0.9 and pauses == []  # each held attempt waited 0.5 s, none backed off


def test_an_unanswered_noop_reconnects_at_once(observations, monkeypatch):
    """A worker that is gone answers nothing: after NOOP_TIMEOUT the keep-alive opens a new socket right away."""
    from playground_serve.client import client as client_module

    monkeypatch.setattr(client_module, "NOOP_TIMEOUT", 0.3)
    cfg = Config(noop_reply=False)
    with MockServer(cfg) as mock, connect(mock.url, keepalive=0.2) as client:
        assert wait_for(lambda: client.connects >= 2 and client._ws is not None)  # without waiting for an inference
        assert client.infer(observations[0])["actions"].shape == (10, 7)
