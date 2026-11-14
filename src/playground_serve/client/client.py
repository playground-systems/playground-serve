"""A persistent websocket client for an openpi policy server (scripts/serve_policy.py), wire-compatible with openpi-client.

Protocol (openpi src/openpi/serving/websocket_policy_server.py at commit 215abfb217dbac7d5f1273282331b9b1866c0479): on
connect the server sends its metadata as one msgpack frame; each request is one binary frame holding the msgpack-numpy
observation; each reply is one binary frame holding the policy's output dict plus `server_timing` ({infer_ms,
prev_total_ms}). A text frame is a server traceback, after which the server closes the socket.

Playground Serve extensions, used only when the metadata carries {"playground_serve": {...}} (or the older key
"robo_serve", kept for compatibility with workers deployed before the rename): "jpeg": true means the server decodes
jpeg-mode images (server_decode.py); "noop": true means it answers the observation {"rs_noop": true} with
{"rs_noop": true} without running the policy, which the client sends as a keep-alive; "split": true means it joins a
request sent in two frames on two sockets (docs/WIRE.md, "Split requests").

Image settings from the metadata (any server may send them):
"image_size", [H, W] or an int, is the size the resize and jpeg modes send; "image_mode", one of MODES, is the default
mode. An explicit `size` or `mode` argument wins.
"""

from __future__ import annotations

import inspect
import logging
import secrets
import socket
import threading
import time

import numpy as np
from PIL import Image
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from .. import __version__
from .codec import JPEG_KEY, MODES, encode_images, packb, unpackb

log = logging.getLogger(__name__)
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# Runpod's gateway sometimes answers a held upgrade with this 400 once the worker is ready; the next attempt gets through.
# Any other 400 is a real error.
RETRY_400 = b"failed to assign worker"
NOOP = packb({"rs_noop": True})
# An unanswered no-op means the worker is gone (it is not answered later): reconnect at once.
NOOP_TIMEOUT = 5.0
# The current metadata key first; "robo_serve" is the pre-rename wire key, still accepted for compatibility.
FEATURE_KEYS = ("playground_serve", "robo_serve")
# A connect attempt that failed quickly waits BACKOFF_S[0] before the next one, doubling up to BACKOFF_S[1].
BACKOFF_S = (0.5, 10.0)
SPLIT_KEY = "rs_split"  # in both frames of a split request: [session id, request number]
SPLIT_PATH = "rs_aux"  # query key of the second socket's URL; its value is the session id
IMAGE_SIZE = 224  # resize and jpeg modes: images go as IMAGE_SIZE x IMAGE_SIZE unless `size` or the metadata says
# Client ping frames, without a ping timeout (the no-op is the liveness check). websockets 13.1, the last release for
# Python 3.8, has no pings in its threading client; the server's pings are answered either way.
PINGS = ({"ping_interval": 20, "ping_timeout": None}
         if "ping_interval" in inspect.signature(ws_connect).parameters else {})


class ServerError(RuntimeError):
    """The server answered with a text frame (its traceback)."""


class Policy:
    """`act(obs)` returns the action chunk as a numpy array (horizon, action_dim). `infer(obs)` returns the server's
    whole reply dict, as openpi's WebsocketClientPolicy does: the chunk is `["actions"]`.

    mode None (the default) picks, at every connect, the metadata's `image_mode` if the server sends one (jpeg only
    if it advertises jpeg), else `jpeg` when the server advertises it and `resize` otherwise; an explicit mode is always
    used. size None (the default) is the metadata's `image_size` ([H, W] or an int) if any, else IMAGE_SIZE (224); an
    explicit size (int or (H, W)) is always used. `size` holds the (H, W) in use. After each call `timing` holds payload_bytes, encode_ms (image transform +
    msgpack), rtt_ms (send to reply received), infer_ms (the server's, or None) and unpack_ms.

    keepalive: when the server advertises `noop` and nothing was sent for this many seconds, a background thread sends
    a no-op request, which also checks that the worker is still there: no reply within NOOP_TIMEOUT (5 s) and it
    reconnects at once. None or 0 turns it off. One request is in flight at a time: the keep-alive skips while `infer`
    runs, and an `infer` that starts while a no-op is in flight waits for that no-op's reply (one network round trip, no
    inference).

    split: when the server advertises `split` and the mode is jpeg, a second socket carries about half of each request
    (every second image, largest first), so that each half fits the first flight of a TCP connection that was idle.
    Through Runpod's gateway a 19.7 kB request sent 0.2 s or more after the previous one took one more round trip of
    the path (about 70 ms to US-CA-2); two frames of up to 11.5 kB on two sockets did not. The same bytes, the same
    inference; False keeps one socket. If the second socket cannot be opened or fails, requests go whole on the first.

    open_timeout caps each upgrade attempt (and the wait for the metadata frame). Runpod's gateway holds an upgrade while
    the worker starts but does not reliably hand it over when the worker turns ready (502 or 400 well after ready), so
    short attempts, retried at once when they time out, open the socket within seconds of ready.
    """

    def __init__(self, url: str, api_key: str | None = None, *, mode: str | None = None, size=None,
                 jpeg_quality: int = 90, open_timeout: float = 10.0, wait: float = 600.0, retries: int = 2,
                 recv_timeout: float | None = 300.0, keepalive: float | None = 20.0, split: bool = True) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} or None, not {mode!r}")
        self.url = url if url.startswith(("ws://", "wss://")) else f"ws://{url}"
        self._mode, self._size = mode, size
        self.mode = mode or "resize"
        self.size, self.jpeg_quality = _height_width(size), jpeg_quality
        self.open_timeout, self.wait, self.retries, self.recv_timeout = open_timeout, wait, retries, recv_timeout
        self.keepalive, self.split = keepalive, split
        self._aux: ClientConnection | None = None  # the second socket of split requests
        self._sid, self._seq = "", 0
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._ws: ClientConnection | None = None
        self._io = threading.RLock()  # held for each request/reply pair: one request in flight on the socket
        self._last_send = time.monotonic()
        self._stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self.metadata: dict = {}
        self.timing: dict = {}
        self.connects = 0
        self.noops = 0
        self.connect()

    @property
    def reconnects(self) -> int:
        return max(0, self.connects - 1)

    @property
    def features(self) -> dict:
        """The server's Playground Serve features, e.g. {"jpeg": True, "noop": True}; {} for a stock openpi server."""
        meta = self.metadata if isinstance(self.metadata, dict) else {}
        for key in FEATURE_KEYS:
            if isinstance(meta.get(key), dict):
                return meta[key]
        return {}

    def connect(self, _stop: threading.Event | None = None) -> None:
        """Open the socket and read the metadata frame, retrying for up to `wait` seconds: a serverless endpoint
        scaling from zero can refuse connections or answer 5xx for a minute or more. An attempt that timed out while
        the gateway held it is retried at once; quick refusals back off. `_stop` (the keep-alive's) ends the retries
        when the policy is closed."""
        with self._io:
            self._close_ws()
            deadline = time.monotonic() + self.wait
            delay = BACKOFF_S[0]
            while True:
                started = time.monotonic()
                try:
                    self._ws = self._open()
                    self.connects += 1
                    break
                except InvalidStatus as e:
                    status, body = e.response.status_code, bytes(e.response.body or b"")
                    if status not in RETRY_STATUS and not (status == 400 and RETRY_400 in body):
                        raise
                    err: Exception = e
                except socket.gaierror:
                    raise
                except (OSError, InvalidHandshake, ConnectionClosed) as e:  # refused, reset, timeout, handshake cut off
                    err = e
                held = time.monotonic() - started >= 0.9 * self.open_timeout
                pause = 0.0 if held else delay
                if time.monotonic() + pause > deadline or (_stop is not None and _stop.is_set()):
                    raise ConnectionError(f"could not connect to {self.url} within {self.wait:.0f} s: {err!r}") from err
                log.info("connect to %s failed (%r); retrying in %.1f s", self.url, err, pause)
                if not held:
                    time.sleep(delay)
                    delay = min(delay * 2, BACKOFF_S[1])
            jpeg = self.features.get("jpeg") is True
            meta = self.metadata if isinstance(self.metadata, dict) else {}
            if self._mode is None:
                wanted = meta.get("image_mode")
                ok = wanted in MODES and (wanted != "jpeg" or jpeg)
                self.mode = wanted if ok else "jpeg" if jpeg else "resize"
            if self._size is None:
                self.size = _height_width(meta.get("image_size"))
            if self.mode == "jpeg":
                Image.preinit()  # load Pillow's JPEG plugin now (~30-50 ms) instead of in the first request
            if self.split and self.mode == "jpeg" and self.features.get("split") is True:
                self._open_aux()
            self._last_send = time.monotonic()
            alive = self._keepalive_thread is not None and self._keepalive_thread.is_alive()
            if self.keepalive and self.features.get("noop") is True and not alive:
                self._stop = threading.Event()
                self._keepalive_thread = threading.Thread(target=self._keepalive_loop, args=(self._stop,),
                                                          name="playground-serve-keepalive", daemon=True)
                self._keepalive_thread.start()

    def _open_aux(self) -> None:
        """The second socket, to the same URL with ?rs_aux=<a new session id>. The server sends nothing on it."""
        self._sid, self._seq = secrets.token_hex(8), 0
        url = f"{self.url}{'&' if '?' in self.url else '?'}{SPLIT_PATH}={self._sid}"
        try:
            self._aux = self._open(url)
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            log.warning("no second socket to %s (%r); requests go whole on one socket", self.url, e)

    def _open(self, url: str | None = None) -> ClientConnection:
        # __enter__ returns the open connection in every websockets version: connect() returns the connection up to 17
        # (17.1 warns unless it is entered) and a context manager after; close() below ends it either way.
        ws = ws_connect(url or self.url, additional_headers=self._headers, compression=None, max_size=None,
                        open_timeout=self.open_timeout, user_agent_header=f"playground-serve/{__version__}",
                        **PINGS).__enter__()
        try:
            # websockets 17 sets this too; set it here so it holds in older versions.
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if url:
                return ws
            first = ws.recv(timeout=self.open_timeout)
            if isinstance(first, str):
                raise ConnectionError(f"{self.url} sent text instead of openpi's metadata frame: {first[:300]}")
            self.metadata = unpackb(first)
        except BaseException:
            ws.close()
            raise
        return ws

    def encode(self, obs: dict) -> bytes:
        return packb(encode_images(obs, self.mode, self.size, self.jpeg_quality))

    def _frames(self, obs: dict):
        """(the request frame, the second socket's frame or None). Split: every second image, largest first, goes
        to the second socket; both frames carry SPLIT_KEY: [session id, request number]."""
        enc = encode_images(obs, self.mode, self.size, self.jpeg_quality)
        if self._aux is not None and isinstance(enc, dict):
            images = sorted((k for k, v in enc.items() if isinstance(v, dict) and v.get(JPEG_KEY) is True),
                            key=lambda k: -len(enc[k][b"data"]))
            if len(images) >= 2:
                self._seq += 1
                tag, second = [self._sid, self._seq], images[1::2]
                return (packb({**{k: v for k, v in enc.items() if k not in second}, SPLIT_KEY: tag}),
                        packb({**{k: enc[k] for k in second}, SPLIT_KEY: tag}))
        return packb(enc), None

    def infer(self, obs: dict) -> dict:
        self._last_send = time.monotonic()  # before the lock: a keep-alive due now sees the activity and skips
        t0 = time.perf_counter()
        made = self.connects, self._aux
        data, part = self._frames(obs)
        t1 = time.perf_counter()
        with self._io:
            for attempt in range(self.retries + 1):
                if self._ws is None:
                    self.connect()  # has its own deadline
                if made != (self.connects, self._aux):  # a reconnect: other features, another second socket
                    made = self.connects, self._aux
                    data, part = self._frames(obs)
                try:
                    t2 = time.perf_counter()
                    self._last_send = time.monotonic()
                    if part is not None:
                        try:
                            self._aux.send(part)
                        except (ConnectionClosed, OSError) as e:  # go on with one socket
                            log.warning("second socket to %s lost (%r); requests go whole on one socket", self.url, e)
                            self._close_aux()
                            made = self.connects, None
                            data, part = self._frames(obs)
                    self._ws.send(data)
                    reply = self._ws.recv(timeout=self.recv_timeout)
                    t3 = time.perf_counter()
                    break
                except TimeoutError:  # the server is slow, not gone: do not send the request twice
                    self._close_ws()
                    raise
                except (ConnectionClosed, OSError) as e:
                    self._close_ws()
                    if attempt == self.retries:
                        raise ConnectionError(f"lost the connection to {self.url} {attempt + 1} times") from e
                    log.warning("connection to %s lost (%r); reconnecting", self.url, e)
            if isinstance(reply, str):
                self._close_ws()  # openpi's server closes the socket after an error
                raise ServerError(f"error in inference server:\n{reply}")
        result = unpackb(reply)
        t4 = time.perf_counter()
        server = result.get("server_timing") if isinstance(result, dict) else None
        self.timing = {
            "payload_bytes": len(data) + len(part or b""),
            "encode_ms": (t1 - t0) * 1e3,
            "rtt_ms": (t3 - t2) * 1e3,
            "infer_ms": server.get("infer_ms") if isinstance(server, dict) else None,
            "unpack_ms": (t4 - t3) * 1e3,
        }
        return result

    def act(self, obs: dict) -> np.ndarray:
        """The action chunk for `obs`: a numpy array of shape (horizon, action_dim), e.g. (10, 7) for LIBERO."""
        return np.asarray(self.infer(obs)["actions"])

    def _keepalive_loop(self, stop: threading.Event) -> None:
        delay = self.keepalive
        while not stop.wait(delay):
            delay = self._last_send + self.keepalive - time.monotonic()
            if delay > 0:
                continue  # a request went out while this thread slept
            delay = self.keepalive
            if self._ws is None or self.features.get("noop") is not True or not self._io.acquire(blocking=False):
                continue  # no socket (the next infer reconnects), a server without no-op, or a request in flight
            try:
                if not stop.is_set():
                    self._noop()
            finally:
                self._io.release()

    def _noop(self) -> None:
        ws = self._ws
        if ws is None:  # closed by close() from another thread
            return
        self._last_send = time.monotonic()
        try:
            if self._aux is not None:
                try:
                    self._aux.send(NOOP)  # keeps the second socket in use; the server does not answer it
                except (ConnectionClosed, OSError):
                    self._close_aux()
            ws.send(NOOP)
            reply = ws.recv(timeout=NOOP_TIMEOUT)
            ok = not isinstance(reply, str) and isinstance(msg := unpackb(reply), dict) and msg.get("rs_noop") is True
        except (ConnectionClosed, OSError, TimeoutError) as e:
            if not self._stop.is_set():  # not closed by close()
                log.warning("keep-alive to %s got no reply (%r): the worker is gone; reconnecting", self.url, e)
                self._reconnect()
            else:
                self._close_ws()
            return
        if not ok:
            log.warning("keep-alive to %s got an unexpected reply: %r; reconnecting", self.url, reply[:300])
            self._reconnect()
            return
        self.noops += 1

    def _reconnect(self) -> None:
        """From the keep-alive thread (which holds the lock): a new socket now, not at the next inference."""
        self._close_ws()
        try:
            self.connect(_stop=self._stop)
        except Exception as e:  # noqa: BLE001  (the next infer tries again)
            log.warning("reconnect to %s failed (%r); the next infer tries again", self.url, e)

    # openpi BasePolicy / WebsocketClientPolicy compatibility
    def get_server_metadata(self) -> dict:
        return self.metadata

    def reset(self) -> None:
        pass

    def _close_aux(self) -> None:
        aux, self._aux = self._aux, None
        if aux is not None:
            aux.close()

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        self._close_aux()
        if ws is not None:
            ws.close()

    def close(self) -> None:
        """Stop the keep-alive and close the socket. A later `infer` reconnects."""
        self._stop.set()
        self._close_ws()

    def __enter__(self) -> "Policy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _height_width(size) -> tuple:
    """(H, W) from an int, an [H, W] pair or None (IMAGE_SIZE); anything else is IMAGE_SIZE too."""
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2 and all(isinstance(n, int) and n > 0 for n in size):
        return (size[0], size[1])
    return (IMAGE_SIZE, IMAGE_SIZE)


def connect(url: str, api_key: str | None = None, **kwargs) -> Policy:
    """Connect to an openpi policy server at ws://host:port or wss://host/path. See `Policy` for the options."""
    return Policy(url, api_key, **kwargs)
