"""Playground Serve's websocket protocol and worker start-up, for models that are not openpi.

The openpi worker (src/playground_serve/worker/serve.py) wraps openpi's own WebsocketPolicyServer. Other models' code
has no such server in our protocol, so this module is that server, written to behave the same way on the wire:

- On connect the server sends one msgpack frame: the policy's metadata (model facts) merged with
  {"playground_serve": {"jpeg": True, "noop": True, "release": RS_RELEASE_ID, "path": "stock", "path_reason": ...}};
  a "playground_serve" map in the policy's own metadata (its "path" and "path_reason") is merged into that one.
  `pgs up` reports ready only once the worker it reaches sends the release id it just set.
- Each request is one binary frame, a msgpack-numpy observation dict. Images the client sent in jpeg mode
  ({b"__jpeg__": True, b"data": ...}) are decoded with server_decode.decode_images before the policy sees them.
- Each reply is one binary frame: the policy's output dict (it must hold "actions") plus
  "server_timing": {"infer_ms", "decode_ms", "prev_total_ms"}; infer_ms covers JPEG decode and policy.infer, as in the
  openpi worker, decode_ms the decode alone. A policy may add "policy_timing": {"<stage>_ms": ...} for its own stages.
- A message whose top-level dict holds "rs_noop": true is answered with {"rs_noop": true} and runs nothing.
- An exception goes back as a text frame (the traceback) and the socket closes with 1011, as openpi's server does.
- GET /healthz answers 200 (openpi's server does too); the Runpod health check is health.py on PORT_HEALTH.

One difference from openpi's server: inference does not run on the event loop, so a no-op or a ping on another socket is
answered while a long inference runs. Inferences still run one at a time, on one thread: in the worker that is the main
thread, the one that loaded and warmed up the model, because torch keeps the CUDA graphs of
torch.compile(mode="reduce-overhead") per thread (a graph recorded on one thread fails an assertion on another).

`run_worker` is the start-up sequence of the openpi worker: wait for the weights (RS_CKPT_DIR, which boot.sh points at
the volume's complete copy or at the container-disk directory stage.py fetch is downloading into), load, one warm-up
inference, a few timed ones, serve on PORT, touch the ready file (health.py then answers 200), and a self-test through
a localhost websocket. Each phase logs one JSON line with the seconds since boot.

A policy is any object with `metadata` (a dict of model facts) and `infer(obs) -> dict` holding "actions".
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http
import json
import logging
import os
import pathlib
import queue
import socket
import subprocess
import threading
import time
import traceback

import websockets.asyncio.server as ws_server
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode

import pgs_msgpack_numpy as msgpack_numpy  # not "msgpack_numpy": that is a pip package model code may import
from server_decode import decode_images

FEATURES = {"playground_serve": {"jpeg": True, "jpeg_stack": True, "noop": True,
                                 "release": os.environ.get("RS_RELEASE_ID", ""),
                                 "path": "stock", "path_reason": os.environ.get("RS_PATH_REASON") or "stock model"}}
PATH_REASON = "/tmp/rs_path_reason"  # why the fast path did not serve; boot.sh passes it to a stock rerun
NOOP_REPLY = msgpack_numpy.packb({"rs_noop": True})
BOOT = float(os.environ.get("RS_BOOT_T", time.time()))
READY_FILE = os.environ.get("RS_READY_FILE", "/tmp/rs_ready")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("websockets").setLevel(logging.WARNING)


def log(phase, **kw):
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z"
    print(json.dumps({"rs": phase, "utc": stamp, "s_since_boot": round(now - BOOT, 3), **kw}, default=str), flush=True)


def is_noop(msg) -> bool:
    if not (isinstance(msg, bytes) and b"rs_noop" in msg):  # byte search first: observations are up to 400 kB
        return False
    obs = msgpack_numpy.unpackb(msg)
    return isinstance(obs, dict) and (obs.get("rs_noop") is True or obs.get(b"rs_noop") is True)


class CallerThreadExecutor:
    """Runs submitted calls one at a time on the thread that calls run(): the worker's main thread."""

    def __init__(self):
        self._calls: queue.Queue = queue.Queue()

    def submit(self, fn, *args) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._calls.put((future, fn, args))
        return future

    def run(self, until: threading.Event | None = None) -> None:
        while until is None or not until.is_set():
            try:
                future, fn, args = self._calls.get(timeout=0.1)
            except queue.Empty:
                continue
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args))
                except BaseException as e:
                    future.set_exception(e)

    def shutdown(self, wait: bool = False) -> None:
        pass


class Server:
    """Serves one policy on host:port from a background thread. `start()` returns once the socket is listening.
    Inference runs on `executor` (default: one worker thread; run_worker passes the main thread's)."""

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, executor=None):
        self.policy, self.host, self.port = policy, host, port
        meta = dict(getattr(policy, "metadata", None) or {})
        own = meta.get("playground_serve") if isinstance(meta.get("playground_serve"), dict) else {
            k: meta[k] for k in ("path", "path_reason") if k in meta}  # the top-level keys some families set
        self.metadata = {**meta, "playground_serve": {**FEATURES["playground_serve"], **own}}
        self._pool = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    def _infer(self, msg: bytes) -> dict:
        t = time.monotonic()
        obs = decode_images(msgpack_numpy.unpackb(msg))
        decoded = time.monotonic()
        result = self.policy.infer(obs)
        if not isinstance(result, dict) or "actions" not in result:
            raise TypeError(f"policy.infer must return a dict with 'actions', got {type(result).__name__}")
        done = time.monotonic()
        return {**result, "server_timing": {"infer_ms": (done - t) * 1000, "decode_ms": (decoded - t) * 1000}}

    async def _handler(self, ws):
        opened, path, noops, prev_total = time.time(), ws.request.path, 0, None
        log("ws_open", path=path, peer=str(ws.remote_address))
        await ws.send(msgpack_numpy.packb(self.metadata))
        try:
            while True:
                msg = await ws.recv()
                start = time.monotonic()
                if is_noop(msg):
                    await ws.send(NOOP_REPLY)
                    noops += 1
                    continue
                try:
                    reply = await asyncio.wrap_future(self._pool.submit(self._infer, msg))
                except Exception:
                    await ws.send(traceback.format_exc())
                    await ws.close(CloseCode.INTERNAL_ERROR,
                                   "Internal server error. Traceback included in previous frame.")
                    raise
                if prev_total is not None:
                    reply["server_timing"]["prev_total_ms"] = prev_total * 1000
                await ws.send(msgpack_numpy.packb(reply))
                prev_total = time.monotonic() - start
        except ConnectionClosed:
            pass
        finally:
            log("ws_close", path=path, s=round(time.time() - opened, 1), code=ws.close_code, noops=noops)

    @staticmethod
    def _health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def _serve(self, started: concurrent.futures.Future):
        self._stop = asyncio.Event()
        try:
            async with ws_server.serve(self._handler, self.host, self.port, compression=None, max_size=None,
                                       process_request=self._health) as server:
                self.port = server.sockets[0].getsockname()[1]
                started.set_result(self.port)
                await self._stop.wait()
        except BaseException as e:
            if not started.done():
                started.set_exception(e)
            raise

    def start(self) -> "Server":
        started: concurrent.futures.Future = concurrent.futures.Future()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_until_complete, args=(self._serve(started),),
                                        name="pgs-server", daemon=True)
        self._thread.start()
        started.result(timeout=30)
        return self

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(10)
        self._pool.shutdown(wait=False)


def gpu_info(fields: str = "name,uuid,driver_version,memory.total") -> str:
    try:
        query = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"]
        return subprocess.run(query, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def wait_for_weights(ckpt_dir: str | None) -> None:
    """Wait until stage.py fetch has written ckpt_dir/.rs_complete; exit if it failed. No-op without a directory."""
    if not ckpt_dir:
        return
    import stage  # /rs/stage.py, the openpi worker's

    t = time.time()
    while not stage.complete(ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, ".rs_failed")):
            log("ckpt_failed", dir=ckpt_dir)
            raise SystemExit(1)
        time.sleep(0.2)
    log("ckpt_ready", dir=ckpt_dir, waited_s=round(time.time() - t, 1))


def wait_for_fetch(ckpt_dir: str | os.PathLike) -> bool:
    """Wait while boot.sh's stage.py fetch writes ckpt_dir; then True if the copy is complete. For a family's own
    download step (it fetches only if this is False), so that two never write the same directory at once."""
    import stage  # /rs/stage.py

    while stage.fetching(ckpt_dir) and not stage.complete(ckpt_dir):
        time.sleep(0.2)
    return stage.complete(ckpt_dir)


def selftest(port: int, make_example, calls: int = 3) -> dict:
    """Round trips through the localhost websocket: metadata, `calls` inferences, one no-op."""
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{port}/ws", compression=None, max_size=None) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
        rows = []
        for _ in range(calls):
            t = time.time()
            ws.send(msgpack_numpy.packb(make_example()))
            reply = ws.recv()
            if isinstance(reply, str):
                raise RuntimeError(f"self-test inference failed:\n{reply}")
            out = msgpack_numpy.unpackb(reply)
            rows.append({"rtt_ms": round((time.time() - t) * 1000, 1),
                         "infer_ms": round(out["server_timing"]["infer_ms"], 1)})
        t = time.time()
        ws.send(msgpack_numpy.packb({"rs_noop": True}))
        noop = msgpack_numpy.unpackb(ws.recv())
    record = {"metadata": str(meta)[:300], "actions_shape": list(out["actions"].shape), "calls": rows,
              "noop_reply": str(noop), "noop_ms": round((time.time() - t) * 1000, 2)}
    log("selftest_ws", **record)
    return record


def run_worker(load_policy, make_example, *, port: int | None = None, steady: int = 3, block: bool = True,
               ready_file: str = READY_FILE) -> Server:
    """The worker's start-up: load, warm up, serve on PORT, mark ready, self-test. Every inference, warm-up included,
    runs on the calling (main) thread, which then serves requests forever if `block` (else until the self-test ends)."""
    port = int(os.environ.get("PORT", "8000")) if port is None else port
    log("process_start", host=socket.gethostname(), gpu=gpu_info(), worker=os.environ.get("RUNPOD_POD_ID"),
        dc=os.environ.get("RUNPOD_DC_ID"), model=os.environ.get("RS_MODEL"), checkpoint=os.environ.get("RS_CHECKPOINT"))
    wait_for_weights(os.environ.get("RS_CKPT_DIR"))
    t = time.time()
    policy = load_policy()
    log("weights_loaded", s=round(time.time() - t, 2), metadata=getattr(policy, "metadata", {}))
    t = time.time()
    policy.infer(make_example())
    log("first_infer_done", ms=round((time.time() - t) * 1000, 1))
    times = []
    for _ in range(steady):
        t = time.time()
        policy.infer(make_example())
        times.append(round((time.time() - t) * 1000, 1))
    log("steady_infer", ms=times, gpu_memory_used=gpu_info("memory.used,memory.total"))
    executor = CallerThreadExecutor()
    server = Server(policy, port=port, executor=executor).start()
    pathlib.Path(ready_file).touch()
    features = server.metadata["playground_serve"]
    log("ready", port=server.port, path=features.get("path"), path_reason=features.get("path_reason"))
    done = threading.Event()

    def _selftest():
        try:
            selftest(server.port, make_example)
        except Exception as e:  # logged, not fatal: the worker keeps serving, as the openpi worker would
            log("selftest_failed", error=repr(e))
        finally:
            done.set()

    threading.Thread(target=_selftest, name="pgs-selftest", daemon=True).start()
    executor.run(until=None if block else done)
    return server
