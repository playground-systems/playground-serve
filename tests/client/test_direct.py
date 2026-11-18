"""The direct port: with a server that advertises one, requests go straight to it over TLS with the certificate and
token from the metadata; the gateway socket stays open for the keep-alive and takes over when the direct one fails."""

import shutil
import socket
import time

import pytest
from conftest import wait_for

from playground_serve.client import client as client_module
from playground_serve.client import connect
from playground_serve.client.mock_server import Config, MockServer
from playground_serve.worker import direct

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command")


def test_requests_go_direct_and_split_follows(observations):
    cfg = Config(direct=True, split=True)
    with MockServer(cfg) as mock, connect(mock.url) as client:
        assert client._direct is not None and client._aux is not None
        for i in range(6):
            assert client.infer(observations[i % 4])["actions"].shape == (10, 7)
            assert client.timing["path"] == "direct"
        assert cfg.requests == cfg.direct_requests == 6 and cfg.aux_frames == 6
        assert client.reconnects == 0


def test_gateway_when_turned_off_or_not_advertised(observations):
    with MockServer(cfg := Config(direct=True)) as mock, connect(mock.url, direct=False) as client:
        client.infer(observations[0])
        assert client._direct is None and client.timing["path"] == "gateway" and cfg.direct_requests == 0
    with MockServer(cfg := Config()) as mock, connect(mock.url) as client:
        client.infer(observations[0])
        assert client._direct is None and client.timing["path"] == "gateway" and cfg.requests == 1


@pytest.mark.parametrize("wrong", ["cert", "token", "port"])
def test_a_direct_port_that_cannot_be_trusted_or_reached_leaves_the_gateway(observations, wrong, caplog):
    cfg = Config(direct=True, split=True)
    with MockServer(cfg) as mock:
        other = {"cert": direct.certificate()[0], "token": "not-the-token", "port": 1}[wrong]
        cfg.direct_info[wrong] = other  # what a client would be told
        with connect(mock.url) as client:
            assert client._direct is None and "no direct connection" in caplog.text
            assert client.infer(observations[0])["actions"].shape == (10, 7)
            assert client.timing["path"] == "gateway" and cfg.direct_requests == 0 and cfg.aux_frames == 1


def test_a_direct_port_that_never_answers_costs_one_open_timeout(observations, caplog, monkeypatch):
    opens = []  # (url, open_timeout) of each socket the client opens
    real_open = client_module.Policy._open
    monkeypatch.setattr(client_module.Policy, "_open", lambda self, url=None, aux=False, **o: opens.append(
        (url, o.get("open_timeout", self.open_timeout))) or real_open(self, url, aux, **o))
    cfg = Config(direct=True)
    with MockServer(cfg) as mock, socket.create_server(("127.0.0.1", 0)) as silent:  # takes the TCP connect, no TLS
        cfg.direct_info["port"] = silent.getsockname()[1]
        t = time.monotonic()
        with connect(mock.url, open_timeout=1) as client:
            assert time.monotonic() - t >= 0.9  # it waited for the TLS answer until the open timeout cut it
            assert [o for o in opens if o[0]] == [(f"wss://127.0.0.1:{silent.getsockname()[1]}/ws", 1)]  # once
            assert client._direct is None and "no direct connection" in caplog.text
            assert client.infer(observations[0])["actions"].shape == (10, 7)
            assert client.timing["path"] == "gateway" and cfg.direct_requests == 0


def test_a_lost_direct_connection_falls_back_without_a_reconnect(observations):
    cfg = Config(direct=True, split=True)
    with MockServer(cfg) as mock, connect(mock.url) as client:
        client.infer(observations[0])
        client._direct.close()
        assert client.infer(observations[1])["actions"].shape == (10, 7)
        assert client.timing["path"] == "gateway" and client._direct is None and client._aux is None
        assert cfg.requests == 2 and cfg.direct_requests == 1 and client.reconnects == 0
        client.close()
        client.infer(observations[2])  # a reconnect opens the direct connection again
        assert client.timing["path"] == "direct" and cfg.direct_requests == 2


def test_keepalive_uses_both_connections(observations):
    cfg = Config(direct=True)
    with MockServer(cfg) as mock:
        client = connect(mock.url, keepalive=0.1)
        assert wait_for(lambda: client.noops >= 2)
        assert cfg.noops >= 2 * client.noops - 1 and client._direct is not None
        client.infer(observations[0])
        assert client.timing["path"] == "direct"
        client.close()
        assert client._direct is None


def test_worker_side_address_and_token():
    env = {"RS_DIRECT": "8002", "RUNPOD_PUBLIC_IP": "127.0.0.1", "RUNPOD_TCP_PORT_8002": "48245"}
    assert direct.address(env) == ("127.0.0.1", 48245, 8002)
    assert direct.address({**env, "RS_DIRECT": ""}) is None and direct.address({"RS_DIRECT": "8002"}) is None
    info, context, port = direct.setup(env)
    assert (info["host"], info["port"], port) == ("127.0.0.1", 48245, 8002) and "BEGIN CERTIFICATE" in info["cert"]
    assert context.num_tickets == 0  # see direct.certificate()
    assert direct.allowed(f"Bearer {info['token']}", info["token"])
    assert not direct.allowed(None, info["token"]) and not direct.allowed("Bearer x", info["token"])
    assert direct.setup({}) is None


def test_hot_marks_every_socket_and_is_off_by_default(observations):
    cfg = Config(direct=True, split=True)
    with MockServer(cfg) as mock:
        with connect(mock.url, hot=True) as client:
            client.infer(observations[0])
            assert client.timing["path"] == "direct" and cfg.direct_requests == 1
        assert len(cfg.paths) == 3 and all("rs_hot=1" in path for path in cfg.paths)  # gateway, direct, second socket
        assert sum("rs_aux=" in path for path in cfg.paths) == 1
        del cfg.paths[:]
        with connect(mock.url + "/ws?x=1") as client:
            client.infer(observations[0])
        assert len(cfg.paths) == 3 and all(path.startswith("/ws?x=1") and "rs_hot" not in path for path in cfg.paths)
