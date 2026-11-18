"""A stand-in for openpi's websocket policy server, so tests and the benchmark run on a laptop with no GPU.

    python -m playground_serve.client.mock_server --port 8000 --infer-ms 50 --uplink-mbps 10 --rtt-ms 20
    python -m playground_serve.client.mock_server --port 8000 --infer-ms 79 --net-p50-ms 57 --net-p95-ms 150 --spike-prob 0.005

Protocol, as in openpi src/openpi/serving/websocket_policy_server.py (commit 215abfb217dbac7d5f1273282331b9b1866c0479):
on connect it sends the metadata as one msgpack frame; each request is a msgpack-numpy observation; each reply is the
output dict plus `server_timing` ({infer_ms, prev_total_ms}); an exception goes back as a text frame (the traceback)
and the socket closes with code 1011. GET /healthz answers 200. Compression is off and frame size is unlimited.

"Inference" decodes JPEG images (server_decode.decode_images, as the real server would), checks that every value is an
array or a string, sleeps --infer-ms, and returns zero actions of shape (10, 7) (pi05_libero's action horizon, LIBERO's
action size). The sleep does not block the event loop; openpi's server blocks it for the length of inference.

Config(direct=True) adds a worker's direct port (worker/direct.py): a second listener with TLS and a token, advertised
in the metadata.

Playground Serve features: the metadata carries {"playground_serve": {"jpeg": True, "noop": True}} (--stock leaves it out), and the
observation {"rs_noop": True} is answered with {"rs_noop": True} without inference (with --stock it reaches inference
and fails, as on a stock server).

Latency model (--net-p50-ms): before each reply, no-ops included, the server sleeps a simulated network round trip:
with probability --spike-prob a spike drawn uniformly from --spike-ms (default 500 to 1000 ms), otherwise a lognormal
draw with median --net-p50-ms whose 95th percentile is --net-p95-ms (sigma = ln(p95 / p50) / 1.645). --seed fixes it.

Network shaping: with --uplink-mbps, --downlink-mbps or --rtt-ms, clients reach the server through a TCP proxy that
delays each direction by half the RTT and paces it at the given rate (a fixed-rate link with a queue). TCP and TLS
handshakes, slow start, packet loss and jitter are not simulated.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import logging
import math
import random
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode

from .codec import packb, unpackb
from .server_decode import decode_images

log = logging.getLogger(__name__)


@dataclass
class Config:
    infer_ms: float = 0.0
    action_shape: tuple[int, int] = (10, 7)
    api_key: str | None = None  # if set, require "Authorization: Bearer <key>"
    drop_every: int | None = None  # abort the TCP connection instead of answering every Nth request
    uplink_mbps: float | None = None
    downlink_mbps: float | None = None
    rtt_ms: float = 0.0
    stock: bool = False  # no playground_serve metadata and no no-op: a stock openpi server
    feature_key: str = "playground_serve"  # "robo_serve": the pre-rename wire key, still accepted for compatibility
    net_p50_ms: float | None = None  # latency model, see the module docstring
    net_p95_ms: float = 150.0
    spike_prob: float = 0.0
    spike_ms: tuple[float, float] = (500.0, 1000.0)
    seed: int = 0
    # Gateway behaviors for tests: answers to the next upgrade attempts, in order: an (HTTP status, body) pair, or
    # "hold" (no answer, as Runpod's gateway holds an upgrade while a worker starts). Then upgrades go through.
    upgrades: list = field(default_factory=list)
    noop_reply: bool = True  # False: no-ops get no reply, as on a worker that is gone
    split: bool = False  # advertise and join split requests (a second socket at ?rs_aux=<session>)
    direct: bool = False  # a second listener as a worker's direct port: TLS with its own certificate, and a token
    direct_info: dict = field(default_factory=dict, init=False)  # what the metadata advertises once it listens
    direct_requests: int = field(default=0, init=False)  # inferences that arrived on the direct listener
    paths: list = field(default_factory=list, init=False)  # the request path of every socket opened, in order
    parts: dict = field(default_factory=dict, init=False)  # (session, request number) -> future of the second frame
    aux_frames: int = field(default=0, init=False)
    requests: int = field(default=0, init=False)
    attempts: int = field(default=0, init=False)
    noops: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    @property
    def shaped(self) -> bool:
        return bool(self.uplink_mbps or self.downlink_mbps or self.rtt_ms)

    def metadata(self) -> dict:
        meta = {"mock": True, "infer_ms": self.infer_ms, "uplink_mbps": self.uplink_mbps,
                "downlink_mbps": self.downlink_mbps, "rtt_ms": self.rtt_ms, "net_p50_ms": self.net_p50_ms,
                "net_p95_ms": self.net_p95_ms, "spike_prob": self.spike_prob, "spike_ms": list(self.spike_ms)}
        if not self.stock:
            meta[self.feature_key] = {"jpeg": True, "noop": True, **({"split": True} if self.split else {}),
                                      **({"direct": self.direct_info} if self.direct_info else {})}
        return meta

    def net_delay(self) -> float:
        """One simulated network round trip in seconds (0 without the latency model)."""
        if not self.net_p50_ms:
            return 0.0
        if self.rng.random() < self.spike_prob:
            return self.rng.uniform(*self.spike_ms) / 1e3
        sigma = math.log(self.net_p95_ms / self.net_p50_ms) / 1.6448536
        return self.rng.lognormvariate(math.log(self.net_p50_ms), sigma) / 1e3


def _infer(obs) -> None:
    if not isinstance(obs, dict):
        raise TypeError(f"observation must be a dict, got {type(obs).__name__}")
    for key, value in obs.items():
        if not isinstance(value, (np.ndarray, np.generic, str)):
            raise TypeError(f"observation[{key!r}] is a {type(value).__name__}; expected an array or a string")


def _part(cfg: Config, tag) -> asyncio.Future:
    return cfg.parts.setdefault(tuple(tag), asyncio.get_running_loop().create_future())


async def _handler(ws: ServerConnection, cfg: Config, direct: bool = False) -> None:
    cfg.paths.append(ws.request.path)
    if cfg.split and "rs_aux=" in ws.request.path:  # the second socket: frames are kept for the first, no replies
        with contextlib.suppress(ConnectionClosed):
            async for msg in ws:
                part = unpackb(msg)
                if "rs_split" in part:
                    cfg.aux_frames += 1
                    _part(cfg, part.pop("rs_split")).set_result(part)
        return
    await ws.send(packb(cfg.metadata()))
    prev_total = None
    while True:
        try:
            start = time.monotonic()
            obs = unpackb(await ws.recv())
            if not cfg.stock and isinstance(obs, dict) and obs.get("rs_noop") is True:
                cfg.noops += 1
                if not cfg.noop_reply:
                    continue
                await asyncio.sleep(cfg.net_delay())
                await ws.send(packb({"rs_noop": True}))
                continue
            if cfg.split and isinstance(obs, dict) and "rs_split" in obs:
                tag = obs.pop("rs_split")
                obs.update(await asyncio.wait_for(_part(cfg, tag), 5))
                del cfg.parts[tuple(tag)]
            cfg.requests += 1
            cfg.direct_requests += direct
            if cfg.drop_every and cfg.requests % cfg.drop_every == 0:
                ws.transport.abort()  # a dropped connection: no close frame, no reply
                return
            t = time.monotonic()
            obs = decode_images(obs)
            _infer(obs)
            await asyncio.sleep(cfg.infer_ms / 1e3)
            action = {"actions": np.zeros(cfg.action_shape, np.float32)}
            action["server_timing"] = {"infer_ms": (time.monotonic() - t) * 1e3}
            if prev_total is not None:
                action["server_timing"]["prev_total_ms"] = prev_total * 1e3
            await asyncio.sleep(cfg.net_delay())
            await ws.send(packb(action))
            prev_total = time.monotonic() - start
        except ConnectionClosed:
            return
        except Exception:
            await ws.send(traceback.format_exc())
            await ws.close(CloseCode.INTERNAL_ERROR, "Internal server error. Traceback included in previous frame.")
            return


def _process_request(cfg: Config):
    async def check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        cfg.attempts += 1
        if cfg.upgrades:
            answer = cfg.upgrades.pop(0)
            if answer == "hold":
                await asyncio.sleep(3600)
            return connection.respond(http.HTTPStatus(answer[0]), answer[1])
        if cfg.api_key and request.headers.get("Authorization") != f"Bearer {cfg.api_key}":
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    return check


class _Link:
    """One direction of a fixed-rate link: bytes queue behind earlier bytes, then take `delay` seconds to arrive."""

    def __init__(self, mbps: float | None, delay: float) -> None:
        self.rate = mbps * 1e6 / 8 if mbps else None  # bytes per second
        self.delay = delay
        self.free_at = 0.0

    def arrival(self, nbytes: int, now: float) -> float:
        start = max(now, self.free_at)
        self.free_at = start + (nbytes / self.rate if self.rate else 0.0)
        return self.free_at + self.delay


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, link: _Link) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def deliver() -> None:
        while True:
            due, data = await queue.get()
            if (wait := due - loop.time()) > 0:
                await asyncio.sleep(wait)
            if data is None:
                break
            writer.write(data)
            await writer.drain()

    task = asyncio.create_task(deliver())
    try:
        while data := await reader.read(16384):
            queue.put_nowait((link.arrival(len(data), loop.time()), data))
    except OSError:
        pass
    queue.put_nowait((link.arrival(0, loop.time()), None))
    with contextlib.suppress(OSError):
        await task
    writer.close()


async def _proxy(cfg: Config, target_port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    one_way = cfg.rtt_ms / 2e3
    try:
        srv_reader, srv_writer = await asyncio.open_connection("127.0.0.1", target_port)
    except OSError:
        writer.close()
        return
    up = asyncio.create_task(_pipe(reader, srv_writer, _Link(cfg.uplink_mbps, one_way)))
    down = asyncio.create_task(_pipe(srv_reader, writer, _Link(cfg.downlink_mbps, one_way)))
    await asyncio.wait([up, down], return_when=asyncio.FIRST_COMPLETED)
    await asyncio.sleep(cfg.rtt_ms / 1e3)  # let the other direction drain its last bytes
    for task in (up, down):
        task.cancel()
    writer.close()
    srv_writer.close()


async def start(cfg: Config, host: str = "127.0.0.1", port: int = 0) -> tuple[list, int]:
    """Start the server (and the shaping proxy, if any). Returns the servers and the port that clients use."""
    Image.preinit()  # as server_decode.DecodeImages does at server start
    server = await serve(lambda ws: _handler(ws, cfg), host, 0 if cfg.shaped else port, compression=None,
                         max_size=None, process_request=_process_request(cfg))
    ws_port = server.sockets[0].getsockname()[1]
    servers = [server]
    if cfg.direct:  # as worker/serve.py: the same handler on a second port, behind worker/direct.py's TLS and token
        import secrets

        from ..worker import direct

        pem, context = direct.certificate()
        token = secrets.token_urlsafe(16)

        def check(connection, request):
            if not direct.allowed(request.headers.get("Authorization"), token):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
            return None

        second = await serve(lambda ws: _handler(ws, cfg, True), host, 0, compression=None, max_size=None, ssl=context,
                             process_request=check)
        cfg.direct_info.update(host=host, port=second.sockets[0].getsockname()[1], cert=pem, token=token)
        servers.append(second)
    if not cfg.shaped:
        return servers, ws_port
    proxy = await asyncio.start_server(lambda r, w: _proxy(cfg, ws_port, r, w), host, port)
    return [*servers, proxy], proxy.sockets[0].getsockname()[1]


class MockServer:
    """The mock on a background thread: `with MockServer(Config(infer_ms=5)) as mock: connect(mock.url)`."""

    def __init__(self, cfg: Config | None = None, port: int = 0) -> None:
        self.cfg, self.port = cfg or Config(), port
        self.url = ""

    def __enter__(self) -> "MockServer":
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers, port = asyncio.run_coroutine_threadsafe(start(self.cfg, port=self.port), self._loop).result(10)
        self.url = f"ws://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc) -> None:
        async def stop() -> None:
            for server in self._servers:
                server.close()
            for server in self._servers:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(server.wait_closed(), 2)
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--infer-ms", type=float, default=0.0, help="fake inference time per request")
    p.add_argument("--uplink-mbps", type=float, help="client-to-server rate")
    p.add_argument("--downlink-mbps", type=float, help="server-to-client rate")
    p.add_argument("--rtt-ms", type=float, default=0.0, help="added round-trip latency")
    p.add_argument("--api-key", help="require Authorization: Bearer <key>")
    p.add_argument("--drop-every", type=int, help="abort the connection on every Nth request")
    p.add_argument("--stock", action="store_true", help="no playground_serve metadata and no no-op, like a stock server")
    p.add_argument("--net-p50-ms", type=float, help="latency model: median simulated round trip per reply")
    p.add_argument("--net-p95-ms", type=float, default=150.0, help="latency model: its 95th percentile")
    p.add_argument("--spike-prob", type=float, default=0.0, help="latency model: probability of a spike")
    p.add_argument("--spike-ms", type=float, nargs=2, default=(500.0, 1000.0), help="latency model: spike range")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = Config(infer_ms=a.infer_ms, api_key=a.api_key, drop_every=a.drop_every, uplink_mbps=a.uplink_mbps,
                 downlink_mbps=a.downlink_mbps, rtt_ms=a.rtt_ms, stock=a.stock, net_p50_ms=a.net_p50_ms,
                 net_p95_ms=a.net_p95_ms, spike_prob=a.spike_prob, spike_ms=tuple(a.spike_ms), seed=a.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    async def run() -> None:
        _, port = await start(cfg, a.host, a.port)
        log.info("mock openpi server on ws://%s:%d %s", a.host, port, cfg.metadata())
        await asyncio.Future()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
