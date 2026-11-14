"""The default image mode follows the server's metadata; the keep-alive no-op keeps to its timing and never shares the
socket with an infer call."""

import random
import threading
import time

from conftest import IMAGE_KEYS, RecordingPolicy, free_port
from openpi_websocket_policy_server import WebsocketPolicyServer

from playground_serve.client import connect
from playground_serve.client.mock_server import Config, MockServer


def test_default_mode_from_metadata(observations):
    with MockServer() as mock:  # advertises {"playground_serve": {"jpeg": True, "noop": True}}
        with connect(mock.url) as client:
            assert client.features == {"jpeg": True, "noop": True} and client.mode == "jpeg"
            client.infer(observations[0])
            assert client.timing["payload_bytes"] < 40_000
        with connect(mock.url, mode="raw") as client:  # an explicit mode wins
            assert client.mode == "raw"
    with MockServer(Config(stock=True)) as mock, connect(mock.url) as client:
        assert client.features == {} and client.mode == "resize"
        client.infer(observations[0])

    policy = RecordingPolicy()  # a stock openpi server: no playground_serve metadata
    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={"config": "pi05_libero"})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with connect(f"ws://127.0.0.1:{port}", wait=10, keepalive=0.05) as client:
        assert client.mode == "resize"
        client.infer(observations[0])
        assert all(policy.seen[0][key].shape == (224, 224, 3) for key in IMAGE_KEYS)
        time.sleep(0.3)
        assert client.noops == 0 and len(policy.seen) == 1  # no no-op reaches a server that did not advertise it


def test_keepalive_timing(observations):
    cfg = Config()
    with MockServer(cfg) as mock, connect(mock.url, keepalive=0.2) as client:
        time.sleep(1.1)  # idle: a no-op 0.2, 0.4, ... s after the last message
        assert 4 <= cfg.noops <= 6 and cfg.noops - 1 <= client.noops <= cfg.noops  # one may be in flight
        before = cfg.noops
        for i in range(20):  # a request every 0.05 s: no no-op
            client.infer(observations[i % 4])
            time.sleep(0.05)
        assert cfg.noops == before and cfg.requests == 20
        time.sleep(0.5)
        assert cfg.noops >= before + 1
    with MockServer(cfg := Config()) as mock, connect(mock.url, keepalive=None) as client:
        time.sleep(0.5)
        assert cfg.noops == 0 and client._keepalive_thread is None


def test_reconnect_to_stock_server_rechecks_features(observations):
    port = free_port()
    with MockServer(port=port) as mock:
        client = connect(mock.url, keepalive=0.1)
        client.infer(observations[0])
        assert client.mode == "jpeg"
    with MockServer(cfg := Config(stock=True), port=port):
        client.infer(observations[1])  # reconnects; the automatic mode follows the new server
        assert client.reconnects == 1 and client.mode == "resize"
        time.sleep(0.4)
        assert cfg.requests == 1 and cfg.noops == 0  # no no-op to a server that does not advertise it
    client.close()


class CountingSocket:
    """Wraps the client's websocket and records how many requests were in flight at once."""

    def __init__(self, ws) -> None:
        self.ws, self.in_flight, self.max_in_flight = ws, 0, 0

    def send(self, data) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.ws.send(data)

    def recv(self, timeout=None):
        reply = self.ws.recv(timeout=timeout)
        self.in_flight -= 1
        return reply

    def __getattr__(self, name):
        return getattr(self.ws, name)


def test_keepalive_never_interleaves(observations):
    # A keep-alive due every 10 ms, replies after 2 to 15 ms, infer calls at random moments: without the lock, no-ops
    # and requests would overlap on the socket many times.
    cfg = Config(infer_ms=2, net_p50_ms=5, net_p95_ms=10, seed=1)
    rng = random.Random(0)
    with MockServer(cfg) as mock, connect(mock.url, keepalive=0.01) as client:
        client._ws = counting = CountingSocket(client._ws)
        for i in range(60):
            out = client.infer(observations[i % 4])
            assert "rs_noop" not in out and out["actions"].shape == (10, 7)
            time.sleep(rng.uniform(0, 0.03))
        assert counting.max_in_flight == 1 and client.reconnects == 0
        assert cfg.requests == 60 and cfg.noops > 10
        assert cfg.noops - 1 <= client.noops <= cfg.noops  # one no-op may be on its way back
