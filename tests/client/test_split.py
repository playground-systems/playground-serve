"""Split requests: with a server that advertises `split`, a jpeg request goes as two frames on two sockets and the
server sees one whole observation; without it, or once the second socket is gone, it goes whole on one."""

import time

from playground_serve.client import client as client_mod
from playground_serve.client import connect
from playground_serve.client.codec import packb, unpackb
from playground_serve.client.mock_server import Config, MockServer


def test_two_frames_two_sockets_one_observation(observations):
    cfg = Config(split=True)
    with MockServer(cfg) as mock, connect(mock.url) as client:
        assert client.features["split"] is True and client._aux is not None
        whole = len(client.encode(observations[0]))
        main, part = client._frames(observations[0])
        assert part is not None and unpackb(main)["rs_split"] == unpackb(part)["rs_split"] == [client._sid, 1]
        assert set(unpackb(main)) | set(unpackb(part)) == set(observations[0]) | {"rs_split"}
        assert max(len(main), len(part)) < 11_500 < whole  # each half fits an idle connection's first flight
        for i in range(8):
            out = client.infer(observations[i % 4])  # the mock checks every value is an array or a string
            assert out["actions"].shape == (10, 7)
            assert whole - 2500 < client.timing["payload_bytes"] < whole + 2500
        assert cfg.requests == 8 and cfg.aux_frames == 8 and not cfg.parts


def test_one_socket_when_not_advertised_or_turned_off(observations):
    with MockServer(cfg := Config()) as mock, connect(mock.url) as client:
        client.infer(observations[0])
        assert client._aux is None and cfg.aux_frames == 0 and cfg.requests == 1
    with MockServer(cfg := Config(split=True)) as mock:
        with connect(mock.url, split=False) as client:
            client.infer(observations[0])
            assert client._aux is None and cfg.aux_frames == 0
        with connect(mock.url, mode="resize") as client:  # only jpeg requests are split
            client.infer(observations[0])
            assert client._aux is None and cfg.aux_frames == 0 and cfg.requests == 2


def test_falls_back_to_one_socket_when_the_second_is_lost(observations):
    cfg = Config(split=True)
    with MockServer(cfg) as mock, connect(mock.url) as client:
        client.infer(observations[0])
        client._aux.close()
        assert client.infer(observations[1])["actions"].shape == (10, 7)
        assert client._aux is None and cfg.aux_frames == 1 and cfg.requests == 2 and client.reconnects == 0
        client.close()
        client.infer(observations[2])  # a reconnect opens a second socket again, with a new session
        assert client._aux is not None and cfg.aux_frames == 2 and cfg.requests == 3


def test_keepalive_uses_both_sockets_and_close_closes_both(observations):
    cfg = Config(split=True)
    with MockServer(cfg) as mock:
        client = connect(mock.url, keepalive=0.1)
        aux = client._aux
        time.sleep(0.45)
        assert client.noops >= 2
        client.infer(observations[0])
        client.close()
        assert client._aux is None and aux.protocol.state.name in ("CLOSING", "CLOSED")
    assert client_mod.SPLIT_KEY == "rs_split" and unpackb(packb({client_mod.SPLIT_KEY: ["a", 1]})) == {"rs_split": ["a", 1]}
