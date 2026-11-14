"""Playground Serve worker: openpi's serve_policy for one model, with a warmup and one JSON log line per startup phase.

Runs inside the openpi checkout's venv (cwd = the checkout). The model comes from the endpoint env: RS_OPENPI_CONFIG
(an openpi config name) and RS_CHECKPOINT (a gs:// URI or a path on the worker). For a gs:// checkpoint boot.sh sets
RS_CKPT_DIR: the volume's complete copy, or the container-disk directory stage.py is downloading into at the same time;
serve.py waits for that download (its .rs_complete marker) after the imports. Other URIs go to openpi's own download
into OPENPI_DATA_HOME. It builds the policy exactly as `scripts/serve_policy.py policy:checkpoint` does, wraps it with
server_decode.DecodeImages (the client's jpeg mode), runs one warmup inference on openpi's example observation for
the policy's robot family (warmup.py; this is the XLA compile), starts openpi's own WebsocketPolicyServer on PORT (wire
protocol unchanged), then touches /tmp/rs_ready so health.py turns /ping to 200, and finally checks the server through
a localhost websocket client.

The JAX compilation cache is per model and per GPU type: RS_JAX_CACHE_ROOT/<RS_MODEL>/<GPU name>, set as
JAX_COMPILATION_CACHE_DIR before jax is imported.

Measurement hooks, inert for normal clients: an observation key "rs_seed" (int) fixes the sampler's noise for that
request; a websocket path containing "noping" turns off the server's ping frames for that connection; every
connection's open and close (code, duration, no-op count) is logged.

Keep-alive, backward compatible with openpi's protocol: a message whose top-level dict holds "rs_noop": true is
answered at once with {"rs_noop": true} and runs no inference. The metadata frame sent on connect advertises this as
{"playground_serve": {"jpeg": true, "noop": true, "path": ..., "path_reason": ...}}, merged into the policy's own
metadata; path is "fast" or "stock" (fastpath.py).

RS_PATH=fast (boot.sh, from fastpath.py): the same server around fast_policy.FastPolicy instead of openpi's policy, JAX
on the CPU only. The fast policy is loaded from the volume's checked copy (RS_FAST_VOLUME: model.pkl, norm stats,
Triton cache), or on a first start converted from the checkpoint in this process (fast_policy.restore and convert, the
arithmetic on the GPU), once boot.sh's `fast_policy.py warm` compiled the kernels in another process meanwhile. It
must pass the self-test (selftest.py) twice before anything is served, and once more in the state that serves. Once a
first start serves, realtime-vla's own converter writes model.pkl into RS_FAST_DIR in a low-priority process
(fast_policy.save: the same weights, compared tensor for tensor; the directory is flagged
.rs_pending until then) and the directory is marked complete for stage.py fill; later starts check it again. Any
failure before ready logs fast_failed,
writes the reason to fastpath.REASON_FILE and exits with fastpath.FALLBACK_EXIT; boot.sh then starts this file again
with RS_PATH=stock and RS_PATH_REASON.

The metadata's "release" is RS_RELEASE_ID, the digest of the endpoint's env map that `pgs up` set: `pgs up` reports
ready only once the worker it reaches sends the id it just set.
"""

import json
import logging
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time

BOOT = float(os.environ.get("RS_BOOT_T", time.time()))
MODEL = os.environ.get("RS_MODEL", "model")
CONFIG = os.environ["RS_OPENPI_CONFIG"]
CHECKPOINT = os.environ["RS_CHECKPOINT"]
FAST = os.environ.get("RS_PATH") == "fast"
if FAST:  # openpi's transforms import JAX; the GPU belongs to the CUDA graph
    os.environ["JAX_PLATFORMS"] = "cpu"
logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(name)s %(message)s")


def log(phase, **kw):
    now = time.time()
    print(json.dumps({"rs": phase, "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z",
                      "s_since_boot": round(now - BOOT, 3), **kw}), flush=True)


def cache_stats():
    d = pathlib.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "/nonexistent"))
    files = [p for p in d.rglob("*") if p.is_file()] if d.exists() else []
    return {"dir": str(d), "files": len(files), "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1)}


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unknown"


gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
if os.environ.get("RS_JAX_CACHE_ROOT") and not FAST:  # before any jax import: jax reads it once, at import
    os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(
        os.environ["RS_JAX_CACHE_ROOT"], slug(MODEL), slug(gpu.splitlines()[0].split(",")[0] if gpu else ""))
log("process_start", host=socket.gethostname(), gpu=gpu, worker=os.environ.get("RUNPOD_POD_ID"),
    dc=os.environ.get("RUNPOD_DC_ID"), model=MODEL, config=CONFIG, checkpoint=CHECKPOINT, jax_cache=cache_stats(),
    path="fast" if FAST else "stock")

# Fast path: built and checked before openpi's server stack is imported and before any JAX backend starts, the order of
# a realtime-vla-only worker; checked once more after those imports and the warm-up, before ready.
import jax  # noqa: E402
from server_decode import DecodeImages  # noqa: E402  (/rs/server_decode.py, a copy of the client's file)


class Hooks:
    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        if "rs_seed" in obs:  # same seed, same noise (openpi otherwise splits one key(0) per call)
            self._policy._rng = jax.random.key(int(obs.pop("rs_seed")))
        return self._policy.infer(obs)

    def __getattr__(self, name):
        return getattr(self._policy, name)


FEATURES = {"playground_serve": {"jpeg": True, "noop": True, "path": "stock",
                                 "path_reason": os.environ.get("RS_PATH_REASON") or "stock openpi policy",
                                 "release": os.environ.get("RS_RELEASE_ID", "")}}
CKPT_DIR = os.environ.get("RS_CKPT_DIR")

def wait_for_download(done):
    """Wait until done() says the checkpoint download (stage.py fetch) got far enough; exit if it failed."""
    while not done():
        if os.path.exists(os.path.join(CKPT_DIR, ".rs_failed")):
            log("ckpt_failed", dir=CKPT_DIR)
            sys.exit(1)
        time.sleep(0.2)


def fast_example():
    """A random LIBERO observation as the client sends it (uint8 images)."""
    import numpy as np

    return {"observation/state": np.random.rand(8), "prompt": "do something",
            "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)}


def fall_back(error: str, trace: str = "") -> None:
    """Never serve an unchecked fast path: log why, leave the reason for boot.sh, exit (the stock path takes over)."""
    import fastpath  # noqa: E402

    log("fast_failed", error=(trace or error)[-1500:])
    pathlib.Path(fastpath.REASON_FILE).write_text(f"fast path failed: {error}"[:500])
    sys.exit(fastpath.FALLBACK_EXIT)


def check_fast(served, ref, attempts) -> dict:
    """The self-test: the stored observation at its seed must give stock openpi's stored actions (each attempt after
    another request went through the same buffers); raises otherwise."""
    import selftest  # noqa: E402  (/rs/selftest.py)

    for attempt in attempts:
        ok, detail = selftest.check(served.infer(selftest.observation(ref))["actions"], ref)
        log("fast_selftest", attempt=attempt, ok=ok, **detail)
        if not ok:
            raise RuntimeError(f"self-test against stock openpi's stored actions failed: {detail}")
        served.infer(fast_example())
    return detail


def build_fast():
    """(FastPolicy, robot transforms, what the final check needs) after the self-test passed twice."""
    import shutil
    import traceback

    import fast_policy  # noqa: E402  (/rs/fast_policy.py)
    import selftest  # noqa: E402
    import stage  # noqa: E402
    from openpi import transforms as _transforms
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config

    local = pathlib.Path(os.environ.get("RS_FAST_DIR", "/rs_local/fast"))
    vol = os.environ.get("RS_FAST_VOLUME", "")
    try:
        os.environ["TRITON_CACHE_DIR"] = str(local / "triton_cache")  # read when triton is first imported
        cfg = _config.get_config(CONFIG)  # before the weights: on a first start while the checkpoint downloads
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        converted = False  # a first start: the weights are saved to RS_FAST_DIR after ready (save_fast)
        saved = next((pathlib.Path(d) for d in (vol, str(local)) if d and stage.complete(d)), None)
        if saved:  # a copy that passed this check on an earlier start (the volume's, or this container's own)
            src = saved
            if src != local and (src / "triton_cache").is_dir() and not (local / "triton_cache").is_dir():
                shutil.copytree(src / "triton_cache", local / "triton_cache")
            t = time.time()
            ckpt = fast_policy.load(str(src / "model.pkl"))
            load_s = round(time.time() - t, 1)
        else:
            src = local
            fast_policy.converter()  # its imports (orbax, transformers) while the checkpoint may still download
            if CKPT_DIR:
                wait_for_download(lambda: stage.complete(CKPT_DIR))
            t = time.time()
            log("fast_convert_start")
            params = fast_policy.restore(CKPT_DIR or CHECKPOINT)
            log("fast_restore", s=round(time.time() - t, 1))
            warm, tw = os.environ.get("RS_WARM_DONE"), time.time()  # boot.sh's kernel compile (fast_policy.py warm)
            while warm and not os.path.exists(warm) and time.time() - tw < 300:  # holds the GPU until it ends
                time.sleep(0.1)
            warm_wait_s = round(time.time() - tw, 1)
            ckpt, converted = fast_policy.convert(params, log=log), True
            del params
            load_s = round(time.time() - t, 1)
            log("fast_convert", s=load_s, warm_wait_s=warm_wait_s)
            shutil.copytree(pathlib.Path(CKPT_DIR or CHECKPOINT) / "assets", local / "assets", dirs_exist_ok=True)
            (local / ".rs_pending").touch()  # stage.py fill waits for model.pkl and the marker (save_fast)
        norm = _checkpoints.load_norm_stats(src / "assets", data_config.asset_id)
        outputs = _transforms.compose([*data_config.model_transforms.outputs,
                                       _transforms.Unnormalize(norm, use_quantiles=data_config.use_quantile_norm),
                                       *data_config.data_transforms.outputs])
        rt = fast_policy.Rtvla(ckpt, outputs)
        del ckpt
        __import__("torch").cuda.empty_cache()  # the copies the graph does not use: room for fast_policy.py save
        log("fast_built", load_s=load_s, capture_s=rt.capture_s, overlapped=rt.overlapped, from_volume=src != local)
        fast = fast_policy.FastPolicy(rt, cfg.model.action_horizon, cfg.model.action_dim, cfg.policy_metadata)
        served, ref = DecodeImages(Hooks(fast)), selftest.load(MODEL)
        served.infer({**fast_example(), "rs_seed": 0})  # one request first, as a realtime-vla-only worker does
        check_fast(served, ref, (1, 2))
        return fast, data_config.data_transforms.inputs, (served, ref, local if src == local else None, converted)
    except Exception as e:  # noqa: BLE001
        fall_back(str(e), traceback.format_exc())


def save_fast(local: pathlib.Path) -> None:
    """After ready, on a first start: the weights as model.pkl next to their norm stats and Triton cache (realtime-vla's
    converter in a low-priority process, fast_policy.save), then the complete-marker, so that stage.py fill copies them
    to the volume (later starts load them from there and check them again). A failure only means the next start
    converts again."""
    import fast_policy  # noqa: E402
    import stage  # noqa: E402

    t = time.time()
    try:
        r = subprocess.run(["nice", "-n", "19", sys.executable, fast_policy.__file__, "save", CKPT_DIR or CHECKPOINT,
                            str(local / "model.pkl")], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}")
        stage.mark(local, stage.sizes(local))
        log("fast_saved", s=round(time.time() - t, 1), bytes=(local / "model.pkl").stat().st_size)
    except Exception as e:  # noqa: BLE001
        log("fast_save_failed", error=repr(e)[:300])
    finally:
        (local / ".rs_pending").unlink(missing_ok=True)


if FAST:
    fast_built = build_fast()

sys.path.insert(0, "scripts")
import serve_policy  # noqa: E402  (openpi/scripts/serve_policy.py)
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402

log("imports_done", jax=jax.__version__, modules=len(sys.modules),
    packages=sorted({n.split(".")[0] for n, m in list(sys.modules.items()) if "site-packages" in (getattr(m, "__file__", "") or "")}))
log("jax_devices", devices=[str(d) for d in jax.devices()])

import warmup  # noqa: E402  (/rs/warmup.py)


NOOP_REPLY = msgpack_numpy.Packer().pack({"rs_noop": True})


class NoopFilter:
    """The socket as openpi's handler sees it: no-op messages are answered here and never reach the policy."""

    def __init__(self, ws):
        self._ws, self.noops = ws, 0

    def __getattr__(self, name):
        return getattr(self._ws, name)

    async def recv(self):
        while True:
            msg = await self._ws.recv()
            if isinstance(msg, bytes) and b"rs_noop" in msg:  # byte search first: observations are up to 400 kB
                obs = msgpack_numpy.unpackb(msg)
                if isinstance(obs, dict) and (obs.get("rs_noop") is True or obs.get(b"rs_noop") is True):
                    await self._ws.send(NOOP_REPLY)
                    self.noops += 1
                    continue
            return msg


class Server(websocket_policy_server.WebsocketPolicyServer):
    async def _handler(self, ws):
        t, path, filtered = time.time(), ws.request.path, NoopFilter(ws)
        if "noping" in path and ws.keepalive_task is not None:
            ws.keepalive_task.cancel()
        log("ws_open", path=path, peer=str(ws.remote_address), header_names=sorted(ws.request.headers.keys()))
        try:
            await super()._handler(filtered)
        finally:
            log("ws_close", path=path, s=round(time.time() - t, 1), code=ws.close_code, reason=ws.close_reason,
                noops=filtered.noops)


args = serve_policy.Args(port=int(os.environ.get("PORT", "8000")),
                         policy=serve_policy.Checkpoint(config=CONFIG, dir=CKPT_DIR or CHECKPOINT))


def precompile():
    """First start: compile while the checkpoint is still downloading. The same policy, built the same way but with
    zeros for weights (openpi's restore_params replaced for this one call: same tree, bfloat16, same sharding), runs
    the warm-up inference, so the compiled programs land in the JAX cache, where the real policy's first inference
    finds them. A failure here only means the real policy compiles as before."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    import openpi.models.model as _model
    from openpi.training import config as _config

    def zeros(params_path, *, dtype=None, **kwargs):
        shapes = nnx.state(nnx.eval_shape(_config.get_config(CONFIG).model.create, jax.random.key(0))).to_pure_dict()
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("x",)), jax.sharding.PartitionSpec())
        return jax.tree.map(lambda s: jnp.zeros(s.shape, dtype or s.dtype, device=sharding), shapes)

    real, _model.restore_params = _model.restore_params, zeros
    try:
        stand_in = serve_policy.create_policy(args)
    finally:
        _model.restore_params = real
    stand_in.infer(warmup.family(stand_in._input_transform.transforms)[1]())


if FAST:
    base, transforms, final = fast_built
else:
    if CKPT_DIR:
        import gc  # noqa: E402
        import stage  # noqa: E402  (/rs/stage.py)

        t = time.time()
        if not stage.complete(CKPT_DIR) and cache_stats()["files"] == 0:
            wait_for_download(lambda: os.path.exists(os.path.join(CKPT_DIR, stage.SMALL)))  # norm stats, metadata
            try:
                precompile()
                log("precompile_done", s=round(time.time() - t, 1), jax_cache=cache_stats())
            except Exception as e:  # noqa: BLE001
                log("precompile_failed", error=repr(e)[:500])
            gc.collect()
        t = time.time()
        wait_for_download(lambda: stage.complete(CKPT_DIR))
        log("ckpt_ready", dir=CKPT_DIR, waited_s=round(time.time() - t, 1))
    base = serve_policy.create_policy(args)
    transforms = base._input_transform.transforms
policy = DecodeImages(Hooks(base))
robot, make_example = warmup.family(transforms)
log("weights_loaded", robot=robot, path=FEATURES["playground_serve"]["path"])

t = time.time()
policy.infer(make_example())
log("first_infer_done", ms=round((time.time() - t) * 1000, 1), jax_cache=cache_stats())
steady = []
for _ in range(5):
    t = time.time()
    policy.infer(make_example())
    steady.append(round((time.time() - t) * 1000, 1))
log("steady_infer", ms=steady)
if FAST:  # once more in the state that will serve: after the server's imports and the warm-up requests
    import traceback  # noqa: E402

    served, ref, built_here, converted = final
    try:
        detail = check_fast(served, ref, (3,))
    except Exception as e:  # noqa: BLE001
        fall_back(str(e), traceback.format_exc())
    base._rng = None  # unseeded requests draw their noise on the GPU again
    FEATURES["playground_serve"].update(path="fast", path_reason=(
        f"realtime-vla graph on {gpu.split(',')[0]}, same bf16 weights; boot self-test passed (max "
        f"{detail['max_abs_diff']}, mean {detail['mean_abs_diff']} against stock openpi's stored actions)"))

server = Server(policy=policy, host="0.0.0.0", port=args.port, metadata={**(policy.metadata or {}), **FEATURES})
threading.Thread(target=server.serve_forever, daemon=True).start()
while True:
    try:
        socket.create_connection(("127.0.0.1", args.port), timeout=1).close()
        break
    except OSError:
        time.sleep(0.05)
pathlib.Path("/tmp/rs_ready").touch()
log("ready", port=args.port, path=FEATURES["playground_serve"]["path"],
    path_reason=FEATURES["playground_serve"]["path_reason"])

# Self-check through openpi's wire protocol on localhost (msgpack-numpy over the websocket server).
from websockets.sync.client import connect  # noqa: E402

with connect(f"ws://127.0.0.1:{args.port}/ws", compression=None, max_size=None) as ws:
    meta = msgpack_numpy.unpackb(ws.recv())
    rows = []
    for _ in range(5):
        t = time.time()
        ws.send(msgpack_numpy.Packer().pack(make_example()))
        out = msgpack_numpy.unpackb(ws.recv())
        rows.append({"rtt_ms": round((time.time() - t) * 1000, 1), "infer_ms": round(out["server_timing"]["infer_ms"], 1)})
    t = time.time()
    ws.send(msgpack_numpy.Packer().pack({"rs_noop": True}))
    noop = msgpack_numpy.unpackb(ws.recv())
    log("selftest_ws", metadata=str(meta)[:200], actions_shape=list(out["actions"].shape), calls=rows,
        noop_reply=str(noop), noop_ms=round((time.time() - t) * 1000, 2))

if FAST and converted:
    save_fast(built_here)
threading.Event().wait()
