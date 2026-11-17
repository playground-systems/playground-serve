"""The four commands (up, down, status, delete) on top of Runpod's REST v2.

Facts this module is built on (measured on Runpod; numbers in docs/BENCHMARKS.md):
- one load-balancing endpoint per model on the stock image, worker files in the env var RS_FILES, a network volume
  that holds the built environment, the checkpoint and the JAX cache (a first boot serves from its own disk and
  fills the volume afterwards, worker/stage.py);
- the gateway holds a request while the worker starts and does not reliably hand it over once the worker is ready;
- a PATCH with `env` replaces the whole env map, so every env update sends the complete map;
- endpoint names stay taken by a hidden template after a delete, so names carry a random suffix;
- with scaling requestCount 1 the gateway admits one websocket per worker, so requestCount is 4;
- a FlashBoot-resumed container loses its worker minutes into a session, so FlashBoot is off;
- on a freshly started container a worker stays while a websocket is open, even a silent one, and a request on a
  min-0 endpoint starts one; so by default no worker is kept (min 0) and a worker stops idleTimeout after its last
  request or client; `--warm` keeps min 1;
- creates go one at a time; a create that failed in transit is looked up by name before moving on;
- the data center of a new endpoint is the nearest with stock: measured from this machine through the gateway by a
  short-lived probe endpoint per candidate (region.py, cached in the state file for a day), or before any such
  measurement guessed from this machine's time zone, unless `--region` names one. The endpoint is created at once;
  the probe runs while its worker boots (it took about a minute before the create when it ran first);
- the fast model path (worker/fastpath.py) is the worker's default where it was measured; `--stock-path` sets
  RS_FAST=off, so openpi's own policy serves;
- a serverless worker gets a public address and a mapped port for every TCP port its endpoint lists (measured in
  US-CA-2: RUNPOD_PUBLIC_IP, RUNPOD_TCP_PORT_<port>), and a request on that port skips the gateway: 77 ms against
  91 ms for 19.7 kB from the US east coast. `--direct` lists DIRECT_PORT and sets DIRECT_KEY; the gateway socket
  still starts and keeps the worker;
- every env map carries RS_RELEASE_ID, a digest of the rest of it, and the openpi worker sends it in its metadata;
  `up` reports ready only once the worker the gateway connects to sends the id it just set. After a release change
  Runpod kept or started a worker of the old release (a client held it, or a new connection revived it), so such a
  worker is stopped (workers max 0) and one of the new release starts.

Ownership: every state location (state.py) has a random owner id, and every name it creates carries it:
playground-serve-<owner>-<model>-<6 hex>. REST v2 lists the name with every endpoint and volume, so any key that may
list them can read it (endpoints and volumes also have a `tags` list, but a `pgs` from before owner ids matches names
only, so the owner id in the name also keeps those away). Yours: the name carries this state's owner id, or this state
file holds the id. `up`, `down` and `delete` act only on yours (by id); `status` shows the others as not yours.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import secrets
import threading
import time
import dataclasses
from dataclasses import dataclass

from . import __version__, region
from .runpod_api import ApiError, encode

PREFIX = "playground-serve-"
# Names from before owner ids have no owner part: playground-serve-<model>-<6 hex>.
NAME = re.compile(r"playground-serve-(?:(o[0-9a-f]{8})-)?([a-z0-9-]+)-[0-9a-f]{6}")
RELEASE = __version__
LEVEL = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
LEVEL_NAME = {v: k for k, v in LEVEL.items()}
# Runpod list prices per GB-month (read 2026-11-13).
VOLUME_USD_PER_GB_MONTH = {"STANDARD": 0.07, "HIGH_PERFORMANCE": 0.14}
HOURS_PER_MONTH = 730
PORTS = ["8000/http", "8001/http"]  # websocket on PORT 8000, health on PORT_HEALTH 8001
# `pgs up --direct`: the worker also listens on this container port (TCP, mapped by Runpod to a port of the worker's
# public address; worker/direct.py), and RS_DIRECT tells it so. A client then sends its requests there.
DIRECT_PORT, DIRECT_KEY = 8002, "RS_DIRECT"
DISK_GB = 50
AGENT_DISK_GB = 10  # an LLM agent's CPU worker: Runpod gives cpu3c 10 GB of container disk per vCPU
# Default: no worker kept. A worker that a request or a client started stays IDLE_MIN after the last one ends (long
# enough for a first start's volume fill, 84 to 139 s after ready), then stops and billing stops.
IDLE_MIN = 5
DOWN = {"min": 0, "max": 1, "idleTimeout": 5}  # `pgs down`: a running worker stops within about a minute
SCALING = {"type": "REQUEST_COUNT", "requestCount": 4}
# A FlashBoot-resumed container lost its worker about 2 minutes into every session tried, whatever the client sent;
# a freshly started one stayed for the whole session. So every start boots
# a fresh container.
FLASHBOOT = "OFF"
TIMEOUT_MS = 3_600_000
# A later start reads the environment and weights from the volume; on a slow host that took 15 min 38 s for a MolmoAct2
# entry (29 GB at about 45 MB/s plus a container restart), so the limit leaves room above that.
FIRST_BOOT_TIMEOUT_S, BOOT_TIMEOUT_S = 30 * 60, 25 * 60
# The gateway holds a request while the worker starts, but does not reliably hand it over once the worker is ready
# (forwarded 0.7 s after ready once; another time a 10 s hold that spanned ready timed out): short attempts, back to back.
GATEWAY_HOLD_S = 3
PROGRESS_EVERY_S = 15  # a worker log read blocks for LOG_READ_S, so the wait reads it only this often
LOG_READ_S = 2
FILL_TIMEOUT_S = 20 * 60
RELEASE_KEY = "RS_RELEASE_ID"
# Runpod refuses a create or update request that it expands past 102400 bytes (HTTP 413, "Expanded upstream request
# exceeds the 102400 byte limit"). Measured 2026-11-16 with dummy endpoints: a 100,300-byte create request was accepted
# and a 100,400-byte one refused. `up` measures its largest request before it creates anything (_check_size).
MAX_REQUEST_BYTES = 100_300
ISSUES = "https://github.com/playground-systems/playground-serve/issues"
RECYCLE_WAIT_S = 90  # after workers max 0, until no worker of the endpoint is listed as running (5 s measured)
PROBE_GRACE_S = 2  # at ready, how long a region probe still running may finish before it is stopped
NEARER_MS = 10  # a measured data center this much nearer than the endpoint's gets a hint how to move
KEY_PAGE = "https://console.runpod.io/user/credentials"
KEY_PERMISSIONS = ("Permission: All, or Restricted with Read/Write for Serverless and for Network volumes (the robot "
                   "client alone needs only access to its endpoint).")
BILLING_PAGE = "https://console.runpod.io/user/billing"

FIRST_BOOT_NOTE = ("First start on this volume: the worker builds the Python environment and downloads the "
                   "checkpoint on its own disk (about 1 to 1.5 minutes), then saves both to the volume in the "
                   "background.")
FAMILY_FIRST_BOOT_NOTE = ("First start on this volume: the worker builds the model's Python environment and downloads "
                          "its checkpoint on its own disk (several minutes for a large checkpoint; MODELS.md has "
                          "measured times), then saves both to the volume in the background.")
FILL_NOTE = ("The worker is now saving the environment and checkpoint to the volume (a few minutes, while it serves); "
             "`pgs down` waits for that, so that later starts take about 1 minute.")
# What the worker is doing after it printed {"rs": <key>} (worker/boot.sh, stage.py and serve.py log lines).
PHASES = {
    "boot": "container started, preparing the environment",
    "path": "container started, preparing the environment",
    "rtvla_clone": "starting the fast model path",
    "build_start": "first start: building the Python environment and downloading the checkpoint (about 1 minute)",
    "ckpt_fetch_start": "downloading the checkpoint",
    "ckpt_fetch_done": "checkpoint downloaded",
    "build_done": "environment built, starting the policy server",
    "unpack_done": "environment unpacked, starting the policy server",
    "unpack_skipped": "starting the policy server",
    "process_start": "loading libraries",
    "imports_done": "loading the checkpoint",
    "jax_devices": "loading the checkpoint",
    "precompile_done": "model compiled, loading the checkpoint",
    "ckpt_ready": "loading the checkpoint",
    "weights_loaded": "compiling the model and running a warmup inference",
    "fast_warm_start": "first start: compiling the fast path's GPU kernels while the checkpoint downloads",
    "fast_warm": "fast path kernels compiled, waiting for the checkpoint",
    "fast_convert_start": "converting the checkpoint for the fast path",
    "fast_restore": "converting the checkpoint for the fast path",
    "fast_gpu_convert": "checkpoint converted, building the fast path's CUDA graph",
    "fast_convert": "checkpoint converted, building the fast path's CUDA graph",
    "fast_built": "fast path built, checking its actions against the stock path's",
    "fast_selftest": "fast path built, checking its actions against the stock path's",
    "fast_failed": "the fast path failed its check: starting the stock path instead",
    "fallback": "the fast path failed its check: starting the stock path instead",
    "model_loaded": "checkpoint loaded",
    # model families (families/<family>/serve.py)
    "cpu": "loading the model",
    "stock_loaded": "model loaded; building the fast path (the first start on a GPU type compiles for a few minutes)",
    "service_built": "model loaded, building the fast path or warming up",
    "serving_setup": "model loaded, warming up",
    "fast_pixels": "fast path built, checking its actions against the stock path's",
    "model_path": "checkpoint loaded; compiling the fast path (about 10 to 30 s) or warming up",
    "first_infer_done": "warmup done, opening the websocket server",
    "steady_infer": "warmup done, opening the websocket server",
    "ready": "server ready, waiting for the gateway",
}


class LauncherError(Exception):
    pass


class Refused(Exception):
    """Runpod declined a create (capacity, data center, name); the next candidate may work."""


@dataclass
class Candidate:
    pool: str
    dc: str
    stock: int
    price: float
    gpus: list[str]
    excluded: list[str]
    volume_types: list[str]

    def label(self) -> str:
        return f"{' / '.join(self.gpus)} (pool {self.pool}) in {self.dc}, serverless stock {LEVEL_NAME[self.stock]}"


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def new_owner() -> str:
    return "o" + secrets.token_hex(4)


def make_name(model: str, owner: str) -> str:
    return f"{PREFIX}{owner}-{slug(model)}-{secrets.token_hex(3)}"


def parse_name(name: str | None) -> tuple[str | None, str] | None:
    """(owner id or None for a name from before owner ids, model slug) of a Playground Serve name, else None."""
    m = NAME.fullmatch(name or "")
    return (m[1], m[2]) if m else None


def is_model_name(model: str, name: str | None) -> bool:
    """A Playground Serve name of `model`, whoever created it."""
    p = parse_name(name)
    return p is not None and p[1] == slug(model)


def workers(warm: bool = False, idle_min: int = IDLE_MIN) -> dict:
    return {"min": 1 if warm else 0, "max": 1, "idleTimeout": int(idle_min * 60)}


def same(current: dict | None, want: dict) -> bool:
    return all((current or {}).get(k) == v for k, v in want.items())


def idle_text(seconds) -> str:
    return f"{seconds // 60} min" if seconds and seconds % 60 == 0 else f"{seconds} s"


def mounted(ep: dict | None) -> list[str]:
    ids = [m.get("volumeId") for m in ((ep or {}).get("mounts") or {}).get("network") or []]
    return ids + list((ep or {}).get("networkVolumes") or [])


def newest(items: list[dict]) -> dict | None:
    return max(items, key=lambda x: x.get("createdAt") or "", default=None)


def cuda_floor(image: str, need: str = "") -> str | None:
    """'runpod/pytorch:1.5.0-cu1281-...' -> '12.8': hosts must run at least the image's CUDA, or `need` (a family's
    min_cuda, such as '13.0' for CUDA 13 wheels) where that is higher."""
    m = re.search(r"-cu(\d\d)(\d)", image)
    floor = f"{m[1]}.{m[2]}" if m else None
    if need and (floor is None or tuple(map(int, need.split("."))) > tuple(map(int, floor.split(".")))):
        return need
    return floor


def spec_cuda(spec) -> str | None:
    """The host CUDA floor of a model: its image's, or its family's min_cuda where that is higher."""
    return cuda_floor(spec.image, getattr(spec, "min_cuda", ""))


def pool_types(gpus: list[dict], pool: str, min_ram_gb: float) -> tuple[list[dict], list[str]]:
    """GPU types of a pool split into usable ones and the ids to exclude (host RAM below the model's need or unknown)."""
    types = [g for g in gpus if g.get("pool") == pool]
    ok = [g for g in types if g.get("minRam") is not None and g["minRam"] >= min_ram_gb]
    return ok, [g["id"] for g in types if g not in ok]


def candidates(gpus: list[dict], dcs: list[dict], spec, only_dc: str | None = None) -> list[Candidate]:
    """(pool, data center) pairs with serverless stock and network volumes: spec pool order, then stock, high first."""
    volume_dcs = {d["id"]: list(d.get("networkVolumeTypes") or []) for d in dcs if d.get("networkVolumeTypes")}
    out = []
    for pool in spec.gpu_pools:
        ok, excluded = pool_types(gpus, pool, spec.min_container_ram_gb)
        if not ok:
            continue
        stock: dict[str, int] = {}
        names: dict[str, list[str]] = {}
        for g in ok:
            for d in g.get("dataCenters") or []:
                level = LEVEL.get(d.get("availability"), 0)
                if level:
                    stock[d["id"]] = max(stock.get(d["id"], 0), level)
                    names.setdefault(d["id"], []).append(g.get("name") or g["id"])
        price = max(g.get("price", {}).get("serverless") or 0.0 for g in ok)
        for dc in sorted(stock, key=lambda k: (-stock[k], k)):
            if dc in volume_dcs and (only_dc is None or dc == only_dc):
                out.append(Candidate(pool, dc, stock[dc], price, names[dc], excluded, volume_dcs[dc]))
    return out


def release_id(env: dict) -> str:
    """Digest of an endpoint's env map without RELEASE_KEY: what a worker started from it reports as its release."""
    return hashlib.sha256(json.dumps({k: v for k, v in env.items() if k != RELEASE_KEY}, sort_keys=True).encode()
                          ).hexdigest()[:12]


def rs_line(line: str) -> dict | None:
    """A worker log line {"rs": <phase>, ...} as a dict, else None."""
    if not line.lstrip().startswith("{"):
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("rs"), str) else None


def fmt_elapsed(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def volume_month_usd(vol: dict) -> float:
    return vol.get("size", 0) * VOLUME_USD_PER_GB_MONTH.get(vol.get("type"), VOLUME_USD_PER_GB_MONTH["HIGH_PERFORMANCE"])


def base_url(ep: dict) -> str:
    return ((ep.get("requestUrls") or {}).get("base") or f"https://{ep['id']}.api.runpod.ai").rstrip("/")


def ws_url(ep: dict) -> str:
    return re.sub(r"^http", "ws", base_url(ep)) + "/ws"


def snippet(model: str, robot: str = "", family: str = "openpi") -> str:
    if robot == "libero" and family == "openpi":  # example_obs is openpi's LIBERO observation
        return (f'from playground_serve import connect, example_obs\npolicy = connect("{model}")\n'
                'print(policy.act(example_obs()).shape)')
    takes = "as the openpi config takes it" if family == "openpi" else "with the keys MODELS.md lists for it"
    return (f'from playground_serve import connect\npolicy = connect("{model}")\n'
            f"actions = policy.act(obs)  # obs: your robot's observation, {takes}")


def stops(w: dict) -> str:
    """When the GPU stops billing, for an endpoint's workers settings `w`."""
    if (w.get("min") or 0) >= 1:
        return "1 worker is kept running (--warm) until `pgs down`"
    return f"the worker stops by itself {idle_text(w.get('idleTimeout'))} after the last request (scale to zero)"


def no_credit(e: Exception) -> bool:
    """A refusal for lack of account credit. Runpod's wording was not seen live; these words ("balance", not the
    "balancer" of a load balancer) are a guess."""
    return re.search(r"\b(balance|credits?|funds)\b", str(e).lower()) is not None


def hf_token_file() -> str:
    """The Hugging Face token `hf auth login` saved on this machine, or "" (HF_TOKEN_PATH, $HF_HOME/token,
    ~/.cache/huggingface/token)."""
    import os
    import pathlib

    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    path = pathlib.Path(os.environ.get("HF_TOKEN_PATH") or os.path.join(home, "token"))
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def secret_env(spec) -> dict:
    """The values of the model's secrets (spec.secrets), from this machine: the environment variable of that name, and
    for HF_TOKEN also the token file `hf auth login` writes. They go only into the endpoint's env on the user's Runpod
    account; never into the state file, and pgs never prints them."""
    import os

    out = {}
    for name in getattr(spec, "secrets", ()) or ():
        value = os.environ.get(name, "").strip() or (hf_token_file() if name == "HF_TOKEN" else "")
        if not value:
            raise LauncherError(
                f"{spec.name} needs the secret {name}, and this machine has none: set the environment variable {name}"
                + (" or log in with `hf auth login` (it saves ~/.cache/huggingface/token); the Hugging Face account "
                   "must have accepted the terms of the gated repos the model loads (MODELS.md)" if name == "HF_TOKEN"
                   else "") + ". pgs puts it only into this endpoint's settings on your Runpod account.")
        out[name] = value
    return out


def advice(e: Exception) -> str:
    """What to do about a Runpod refusal that a new user commonly meets; "" when there is nothing specific to say."""
    from . import state

    status = getattr(e, "status", 0)
    if status == 401:
        return (f"Runpod does not accept this API key (wrong, revoked or disabled). Create one at {KEY_PAGE} (API "
                f"Keys), then set RUNPOD_API_KEY, or delete {state.key_path()} and run pgs again to enter it.")
    if status == 403:
        m = re.search(r"does not grant ([\w:.-]+)", str(e))
        need = f"the permission {m[1]}" if m else "a permission this command needs"
        return (f"This API key lacks {need}. Edit the key at {KEY_PAGE} (API Keys, pencil icon) or create a new one. "
                + KEY_PERMISSIONS)
    if no_credit(e):
        return f"Your Runpod account needs credit: add some at {BILLING_PAGE}, then run the command again."
    if "quota" in str(e).lower():
        return ("Your Runpod account's serverless worker quota is full (max workers over all endpoints). Lower max "
                "workers or delete endpoints you do not use (console, Serverless), or ask Runpod support to raise "
                "the quota.")
    return ""


def ask(question: str) -> bool:
    """A yes (y or yes) on the terminal; closed input (Ctrl-D) is a no. Without a terminal there is no one to ask:
    LauncherError, so a script stops with a sentence instead of a traceback."""
    import sys

    if not sys.stdin.isatty():
        raise LauncherError(f"{question.split('?')[0]}? pgs asks before it deletes, and this is not a terminal: run it "
                            "in one, or pass --yes to delete without asking. Nothing deleted.")
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


class Launcher:
    def __init__(self, api, state_path, *, get_spec=None, make_env=None, start_command=None, model_names=None,
                 out=print, confirm=None, sleep=time.sleep, clock=time.monotonic, poll: float = 5.0,
                 measure=region.echo_ms, now=time.time, here=region.here):
        from . import state

        self.api, self.state_path, self._state = api, state_path, state
        self.data = state.load(state_path)
        self.get_spec, self.make_env, self.model_names = get_spec, make_env, model_names
        self.start_command = start_command  # argv such as ["bash", "-c", script]; default bundle.START_COMMAND
        self.out, self.sleep, self.clock, self.poll = out, sleep, clock, poll
        self.confirm = confirm or ask
        self.measure, self.now = measure, now  # region probe: round trip of one probe endpoint; wall clock for its cache
        self.here = here  # () -> (lat, lon) of this machine's time zone or None: the guess before any probe

    # ----- helpers
    def _spec(self, model: str):
        if self.get_spec is None:
            from . import models

            self.get_spec = models.get
        try:
            return self.get_spec(model)
        except KeyError:
            raise LauncherError(f"unknown model {model!r}; known: {', '.join(self._names()) or 'none'}") from None

    def _env(self, spec) -> dict:
        if self.make_env is None:
            from . import bundle

            self.make_env = bundle.endpoint_env
        return self.make_env(spec, RELEASE)

    def _start(self) -> tuple[list[str], list[str]]:
        """(entrypoint, cmd) of the worker: unpack RS_FILES and run boot.sh."""
        if self.start_command is None:
            from . import bundle

            self.start_command = list(bundle.START_COMMAND)
        return self.start_command[:2], self.start_command[2:]

    def _key(self, model: str) -> str:
        """The name the state file and resource names use: the spec's name (custom CONFIG@CHECKPOINT gets a hash)."""
        if model in self.data["models"]:
            return model
        try:
            return self._spec(model).name
        except LauncherError:
            return model

    def _names(self) -> list[str]:
        if self.model_names is None:
            try:
                from . import models

                self.model_names = models.names
            except ImportError:
                return []
        return list(self.model_names())

    def _save(self) -> None:
        self._state.save(self.state_path, self.data)

    def _rec(self, model: str) -> dict:
        return self.data["models"].get(model, {})

    def _gpu(self, spec, c: Candidate) -> dict:
        gpu = {"pools": [c.pool], "count": 1}
        if spec_cuda(spec):
            gpu["minCudaVersion"] = spec_cuda(spec)
        if c.excluded:
            gpu["excludedTypes"] = c.excluded
        return gpu

    def _container(self, spec, env: dict) -> dict:
        entrypoint, cmd = self._start()
        ports = PORTS + ([f"{env[DIRECT_KEY]}/tcp"] if env.get(DIRECT_KEY) else [])
        return {"image": spec.image, "entrypoint": entrypoint, "cmd": cmd, "ports": ports, "disk": DISK_GB, "env": env}

    def _permission(self, e: ApiError) -> LauncherError:
        return LauncherError(f"Runpod refused this API key: {e}. {advice(e)}")

    def _create_one(self, create, body: dict, listing) -> dict:
        try:
            return create(body)
        except ApiError as e:
            if e.status in (401, 403):
                raise self._permission(e) from None
            if e.status == 422:
                raise LauncherError(f"Runpod rejected the request: {e}") from None
            if no_credit(e):  # no other candidate can help
                raise LauncherError(f"Runpod refused the create: {e}. {advice(e)}") from None
            if e.status == 0 or e.status >= 500:  # it may have gone through: look before moving on
                found = [x for x in listing() if x.get("name") == body["name"]]
                if found:
                    return found[0]
            raise Refused(str(e)) from None

    # ----- whose resources: read only
    @property
    def owner(self) -> str | None:
        return self.data.get("owner")

    def _ensure_owner(self) -> str:
        """This state's owner id, made and saved before the first resource that carries it is created."""
        if not self.owner:
            self.data["owner"] = new_owner()
            self._save()
        return self.data["owner"]

    def _whose(self, x: dict) -> str:
        owner = (parse_name(x.get("name")) or (None,))[0]
        return f"created by owner {owner}" if owner else "created before owner ids and not in this state file"

    def _find(self, model: str, endpoints: list[dict], volumes: list[dict]):
        """(endpoint, volume, (other endpoints, other volumes)) of `model` on the account. Yours: the id is in this
        state file or the name carries this state's owner id; the others are the remaining Playground Serve names of
        `model`. Calls no API and writes nothing."""
        rec, s = self._rec(model), slug(model)

        def split(items, kind):
            rid = (rec.get(kind) or {}).get("id")
            mine, theirs = [], []
            for x in items:
                p = parse_name(x.get("name"))
                if x["id"] == rid or (p and p[1] == s and self.owner and p[0] == self.owner):
                    mine.append(x)
                elif p and p[1] == s:
                    theirs.append(x)
            mine = sorted(mine, key=lambda x: x.get("createdAt") or "", reverse=True)
            return sorted(mine, key=lambda x: x["id"] != rid), theirs  # the state file's id, then the newest

        eps, their_eps = split(endpoints, "endpoint")
        ep = eps[0] if eps else None
        vols, their_vols = split(volumes, "volume")
        vol = next((v for v in volumes if v["id"] in mounted(ep)), None) or (vols[0] if vols else None)
        return ep, vol, (their_eps, [v for v in their_vols if not vol or v["id"] != vol["id"]])

    def _missing(self, model: str, endpoints: list[dict], volumes: list[dict]) -> list[tuple[str, dict]]:
        """Entries of this state file that the account listing lacks."""
        ids = {x["id"] for x in endpoints + volumes}
        return [(k, r) for k, r in self._rec(model).items() if r.get("id") not in ids]

    # ----- account reconciliation: up, down and delete only
    def reconcile(self, model: str, endpoints: list[dict], volumes: list[dict]):
        """_find, and the state file updated to match. An id this state file holds that the listing lacks is asked for
        by id: only a 404 forgets it, never a failed or partial read (any other error stops the command)."""
        for kind, r in self._missing(model, endpoints, volumes):
            try:
                (endpoints if kind == "endpoint" else volumes).append(
                    (self.api.get_endpoint if kind == "endpoint" else self.api.get_volume)(r["id"]))
            except ApiError as e:
                if e.status != 404:
                    raise
                self.out(f"{kind.capitalize()} {r.get('name')} ({r['id']}) is no longer on the account (Runpod answers "
                         "404 for it); forgetting it.")
                self.data["models"][model].pop(kind)
        ep, vol, others = self._find(model, endpoints, volumes)
        self._record(model, ep, vol)
        return ep, vol, others

    def _record(self, model: str, ep: dict | None, vol: dict | None) -> None:
        old = self._rec(model)
        new = {}
        if ep:
            new["endpoint"] = {"id": ep["id"], "name": ep["name"], "dc": (ep.get("dataCenterIds") or [None])[0],
                               "pool": ((ep.get("gpu") or {}).get("pools") or [None])[0], "url": ws_url(ep)}
        if vol:
            built = bool(old.get("volume", {}).get("ready")) and old["volume"].get("id") == vol["id"]
            new["volume"] = {"id": vol["id"], "name": vol["name"], "dc": vol.get("dataCenter"),
                             "type": vol.get("type"), "size": vol.get("size"), "ready": built}
        if new != old or (not new and model in self.data["models"]):
            if new:
                self.data["models"][model] = new
            else:
                self.data["models"].pop(model, None)
            self._save()

    def _not_yours(self, model: str, theirs: list[tuple[str, dict]]) -> str:
        what = "; ".join(f"{k} {x['name']} ({x['id']}, {self._whose(x)})" for k, x in theirs)
        pick = " ID" if len(theirs) > 1 else ""
        return (f"{model}: this Runpod account has {what}, not created from this state ({self.state_path}). `pgs up` "
                f"does not change it. `pgs up {model} --adopt{pick}` takes it over (its worker settings may be updated "
                f"and its worker restarted); `pgs up {model} --new` creates a separate one for you.")

    def _adopt(self, model: str, theirs: list[tuple[str, dict]], adopt, volumes: list[dict], vol: dict | None):
        pick = [(k, x) for k, x in theirs if x["id"] == adopt] if isinstance(adopt, str) else theirs
        if not pick:
            raise LauncherError(f"{adopt} is not one of: {', '.join(x['id'] for _, x in theirs)}")
        kind, x = max(pick, key=lambda kx: kx[1].get("createdAt") or "")
        self._ensure_owner()
        ep = x if kind == "endpoint" else None
        vol = next((v for v in volumes if v["id"] in mounted(ep)), vol) if ep else x
        self._record(model, ep, vol)
        self.out(f"Adopted {kind} {x['name']} ({x['id']}, {self._whose(x)}): this state now manages it. Its "
                 "creator's state still lists it.")
        return ep, vol

    # ----- up
    def up(self, model: str, *, warm: bool = False, idle_min: int = IDLE_MIN, adopt=None, new: bool = False,
           stock_path: bool = False, dc: str | None = None, gpu: str | None = None, direct: bool = False) -> dict:
        """`adopt`: True or an id, to take over another owner's resource of this model; `new`: create your own anyway;
        `stock_path`: serve openpi's own policy, not the fast path; `dc`: the data center of a new endpoint and volume
        (default: the nearest with stock, measured); `gpu`: a Runpod pool to use instead of the model's own pools;
        `direct`: the worker also serves on a port of its own public address (an endpoint made with it keeps it)."""
        spec = self._spec(model)
        if getattr(spec, "cpu", ""):  # an LLM agent: a CPU worker, no volume
            return self._up_agent(spec, workers(warm, idle_min))
        if gpu:
            spec = dataclasses.replace(spec, gpu_pools=[gpu])
        model = spec.name
        if getattr(spec, "experimental", False):
            self.out(f"{model}: experimental: slow or not fully verified, see MODELS.md")
        want = workers(warm, idle_min)
        env = {**self._env(spec), **secret_env(spec)}
        if stock_path and "RS_FAST" in env:
            env = {**env, "RS_FAST": "off"}
        with concurrent.futures.ThreadPoolExecutor(4) as ex:  # four independent reads, about 1.7 s one after another
            reads = [ex.submit(f) for f in (self.api.list_endpoints, self.api.list_volumes,
                                            lambda: self.api.serverless_gpus(spec_cuda(spec)),
                                            self.api.datacenters)]
            endpoints, volumes, gpus, dcs = (r.result() for r in reads)
        ep, vol, (their_eps, their_vols) = self.reconcile(model, endpoints, volumes)
        if direct or ((ep or {}).get("env") or {}).get(DIRECT_KEY):  # an endpoint made with --direct keeps it
            env = {**env, DIRECT_KEY: str(DIRECT_PORT)}
        env = {**env, RELEASE_KEY: release_id(env)}
        release = env[RELEASE_KEY]  # the worker (worker/serve.py) sends it in its metadata
        self._check_size(model, spec, env, gpus, dcs, want)  # before anything is created or changed
        kept = ((ep or {}).get("gpu") or {}).get("pools") or []
        if not gpu and kept and kept[0] not in spec.gpu_pools:  # an endpoint made with --gpu keeps its pool
            spec = dataclasses.replace(spec, gpu_pools=[kept[0], *spec.gpu_pools])
        if ep is None:
            theirs = [("endpoint", e) for e in their_eps] or ([] if vol else [("volume", v) for v in their_vols])
            if theirs and adopt:
                ep, vol = self._adopt(model, theirs, adopt, volumes, vol)
            elif theirs and not new:
                raise LauncherError(self._not_yours(model, theirs))
        t0 = self.clock()
        where = (ep.get("dataCenterIds") or [None])[0] if ep else (vol or {}).get("dataCenter")
        if dc and where and dc != where:
            raise LauncherError(f"{model} is already in {where} (endpoint or volume); `--region` applies when they are "
                                f"created. `pgs delete {model} --volume` first to move it to {dc}.")
        probing = None  # the region probe that runs while a new endpoint's worker boots
        if ep is None:
            ep, vol, price, todo = self._create(model, spec, env, vol, gpus, dcs, want, endpoints, dc)
            probing = self._probe_start(todo, [*endpoints, ep], gpus)
        else:
            ep, price = self._bring_up(spec, env, ep, gpus, dcs, want)
        first = not self._rec(model).get("volume", {}).get("ready")
        family = getattr(spec, "family", "openpi")
        if first:
            self.out(FIRST_BOOT_NOTE if family == "openpi" else FAMILY_FIRST_BOOT_NOTE)
        try:
            self._wait_ready(ep, FIRST_BOOT_TIMEOUT_S if first else BOOT_TIMEOUT_S, t0, want, release)
        except KeyboardInterrupt:
            if probing:
                probing(stop=True)
            kept = (ep.get("workers") or {}).get("min", 0) >= 1
            if kept and want["min"] < 1:  # the wait started a worker with min 1 (a key without gateway access)
                self._update(ep, {"workers": want})
                kept = False
            self.out(f"\nStopped waiting. The worker keeps starting and bills {price:.2f} USD/hr until "
                     f"`pgs down {model}`; `pgs status` shows it." if kept else
                     f"\nStopped waiting. No worker is kept (min 0): one that started stops by itself "
                     f"{idle_text(ep['workers'].get('idleTimeout'))} after the last request.")
            raise
        except BaseException:
            if probing:
                probing(stop=True)
            raise
        if not same(ep.get("workers"), want):  # sent once the server answers (see _bring_up)
            self._update(ep, {"workers": want})
        self.data["models"][model]["volume"]["ready"] = True
        self._save()
        url = ws_url(ep)
        self.out(f"\nReady after {fmt_elapsed(self.clock() - t0)}.")
        for line in probing(model=model, ep=ep) if probing else []:
            self.out(line)
        if first:
            self.out(FILL_NOTE)
        self.out(f"Websocket: {url}")
        if env.get(DIRECT_KEY):
            self.out("Direct port: on. Clients send requests straight to the worker's public address (TLS with the "
                     "worker's own certificate, a token per boot) and keep the gateway socket for start, keep-alive "
                     "and fallback; connect(..., direct=False) uses the gateway only.")
        self.out(billing(model, want, price))
        self.out(f"Volume: about {volume_month_usd(vol):.2f} USD/month until `pgs delete {model} --volume`.")
        from . import state

        robot = getattr(spec, "robot", "")
        if family == "openpi":
            demo = robot == "libero"  # demo.example_obs is openpi's LIBERO observation
        else:
            from . import models

            demo = models.example_observation(model) is not None  # the family's spec.py has an example
        self.out("\nNext: " + (f"`pgs demo {model}` sends a dummy observation and prints the actions' shape and the "
                               "round trip. Or in Python" if demo else "in Python")
                 + f" (the key comes from RUNPOD_API_KEY or {state.key_path()}):\n" + snippet(model, robot, family)
                 + "\nWhen done: `pgs down`. " + ("Until then 1 worker keeps running and billing (--warm)."
                                                  if want["min"] >= 1 else "Without it the worker stops by itself "
                                                  f"{idle_text(want['idleTimeout'])} after the last request."))
        return {"endpoint": ep, "volume": vol, "url": url}

    def _agent_body(self, spec, env: dict, want: dict) -> dict:
        flavor, vcpu = spec.cpu.split("-")[:2]
        entrypoint, cmd = self._start()
        return {"name": make_name(spec.name, self.owner), "type": "LOAD_BALANCER", "image": spec.image, "entrypoint": entrypoint,
                "cmd": [c.replace("/rs/boot.sh", "/rs/start.sh") for c in cmd], "ports": list(PORTS), "disk": AGENT_DISK_GB,
                "env": env, "cpu": [{"id": flavor, "vcpuCount": int(vcpu)}], "workers": dict(want), "scaling": dict(SCALING),
                "flashboot": FLASHBOOT, "timeout": TIMEOUT_MS}

    def _up_agent(self, spec, want: dict) -> dict:
        """An LLM agent (families/llm_agent): a load-balancing endpoint on a CPU worker of spec.cpu, no volume, the image
        spec.image, the family's start.sh in place of boot.sh, and the model's key (spec.secrets) from this machine."""
        import json

        from . import models

        model = spec.name
        env = {**self._env(spec), **secret_env(spec)}
        env = {**env, RELEASE_KEY: release_id(env)}
        release = env[RELEASE_KEY]
        ep, _, (their_eps, _) = self.reconcile(model, self.api.list_endpoints(), [])
        flavor, vcpu = spec.cpu.split("-")[:2]
        body = self._agent_body(spec, env, want)
        size = len(json.dumps(body).encode())
        if size > MAX_REQUEST_BYTES:
            raise LauncherError(f"{model}: the endpoint request would be {size:,} bytes, over the {MAX_REQUEST_BYTES:,} that Runpod takes")
        price = models.CPU_PRICES.get(flavor, 0.0) * int(vcpu)
        agent = getattr(spec, "agent", {}) or {}
        self.out(f"{model}: {agent.get('title', model)} through OpenRouter ({agent.get('slug', '')}), the agent loop on a CPU "
                 f"worker ({flavor}, {vcpu} vCPU): {price:.3f} USD/hr while it runs; the model's tokens bill to the "
                 "OPENROUTER_API_KEY it was given.")
        t0 = self.clock()
        if ep is None:
            if their_eps:
                raise LauncherError(self._not_yours(model, [("endpoint", e) for e in their_eps]))
            body["name"] = make_name(model, self._ensure_owner())
            ep = self._create_one(self.api.create_endpoint, body, self.api.list_endpoints)
            self.out(f"Created endpoint {ep['name']} ({ep['id']}).")
        else:
            ep = self._update(ep, {"env": env, "workers": dict(want)})
        self._record(model, ep, None)
        self._save()
        self._wait_ready(ep, FIRST_BOOT_TIMEOUT_S, t0, want, release)
        url = ws_url(ep)
        self.out(f"\nReady after {fmt_elapsed(self.clock() - t0)}.\nWebsocket: {url}\n" + billing(model, want, price)
                 + "\nEach episode sends its embodiment card (MODELS.md, LLM agents). When done: `pgs down`.")
        return {"endpoint": ep, "volume": None, "url": url}

    def _plan(self, c: Candidate, spec, vol: dict | None, how: str, want: dict) -> None:
        ms = region.fresh(self.data.get("probes"), c.dc, self.now())
        self.out(f"GPU:    {c.label()}: {c.price:.2f} USD/hr while a worker runs; {stops(want)}")
        self.out(f"Data center: {c.dc}" + (f", {ms:.1f} ms round trip from here (probe)" if ms is not None else
                                           f", {how}"))
        if vol is None:
            vtype = "STANDARD" if "STANDARD" in c.volume_types else c.volume_types[0]
            month = spec.volume_gb * VOLUME_USD_PER_GB_MONTH.get(vtype, VOLUME_USD_PER_GB_MONTH["HIGH_PERFORMANCE"])
            self.out(f"Volume: {spec.volume_gb} GB {vtype} in {c.dc}: about {month:.2f} USD/month until deleted")

    def _create(self, model, spec, env, vol, gpus, dcs, want, endpoints=(), dc=None):
        cands = candidates(gpus, dcs, spec, only_dc=vol["dataCenter"] if vol else dc)
        if not cands:
            where = (f" in {vol['dataCenter']}, where the volume {vol['name']} is" if vol else
                     f" in {dc} (--region; it needs serverless stock and network volumes)" if dc else "")
            raise LauncherError(f"No serverless stock for pools {', '.join(spec.gpu_pools)}{where} right now. Runpod's "
                                "stock changes by the minute: try again in a few minutes" + (
                                    f", or `pgs delete {model} --volume` to start over elsewhere." if vol else
                                    ", or leave out --region." if dc else "."))
        owner = self._ensure_owner()
        todo, how = [], "where the volume is" if vol else "chosen with --region"
        if vol is None and dc is None:
            cands, todo, how = self._order(cands, list(endpoints), owner)
        made = None  # a volume created by this call, removed again if the endpoint moves to another data center
        for c in cands:
            if vol is not None and vol["dataCenter"] != c.dc:
                if made is None:
                    continue
                self._drop_volume(model, made, "the endpoint moves to another data center")
                vol = made = None
            self._plan(c, spec, vol, how, want)
            if vol is None:
                vtype = "STANDARD" if "STANDARD" in c.volume_types else c.volume_types[0]
                body = {"name": make_name(model, owner), "size": spec.volume_gb, "dataCenter": c.dc, "type": vtype}
                try:
                    vol = made = self._create_one(self.api.create_volume, body, self.api.list_volumes)
                except Refused as e:
                    self.out(f"  volume create refused in {c.dc}: {e}; trying the next candidate")
                    continue
                self.out(f"Created volume {vol['name']} ({vol['id']}).")
                self.data["models"].setdefault(model, {})["volume"] = {
                    "id": vol["id"], "name": vol["name"], "dc": vol.get("dataCenter", c.dc), "type": vol.get("type"),
                    "size": vol.get("size"), "ready": False}
                self._save()
            try:
                ep = self._create_endpoint(model, owner, spec, env, vol, c, want)
            except Refused as e:
                self.out(f"  endpoint create refused for {c.pool} in {c.dc}: {e}; trying the next candidate")
                continue
            self.out(f"Created endpoint {ep['name']} ({ep['id']}), " + (
                "1 worker kept (--warm)." if want["min"] >= 1 else "no worker kept: the first request starts one."))
            self.data["models"][model]["endpoint"] = {"id": ep["id"], "name": ep["name"], "dc": c.dc, "pool": c.pool,
                                                      "url": ws_url(ep)}
            self._save()
            return ep, vol, c.price, todo
        if made is not None:
            self._drop_volume(model, made, "no candidate accepted the endpoint")
        raise LauncherError("Runpod refused every candidate (see above). Try again later.")

    def _endpoint_body(self, name: str, spec, env: dict, gpu: dict, dc: str, volume_id: str, want: dict) -> dict:
        return {"name": name, "type": "LOAD_BALANCER", **self._container(spec, env), "gpu": gpu, "dataCenterIds": [dc],
                "mounts": {"network": [{"volumeId": volume_id}]}, "workers": dict(want), "scaling": dict(SCALING),
                "flashboot": FLASHBOOT, "timeout": TIMEOUT_MS}

    def _create_endpoint(self, model, owner, spec, env, vol, c: Candidate, want) -> dict:
        last = ""
        for _ in range(4):
            body = self._endpoint_body(make_name(model, owner), spec, env, self._gpu(spec, c), c.dc, vol["id"], want)
            try:
                return self._create_one(self.api.create_endpoint, body, self.api.list_endpoints)
            except Refused as e:
                last = str(e)
                if "quota" in last.lower():  # the probes' slots may take a moment to come back
                    self.out("  the account's worker quota is full; retrying in 5 s")
                    self.sleep(5)
                elif "unique" in str(e).lower():
                    self.out("  that name is still held by an earlier endpoint's template; retrying with a new suffix")
                else:
                    raise
        if "quota" in last.lower():  # no other candidate can help
            raise LauncherError(f"Runpod refused the endpoint: {last}. {advice(Refused(last))}")
        raise Refused("endpoint create kept failing (name collisions)")

    def _largest_request(self, model: str, spec, env: dict, gpus: list[dict], dcs: list[dict], want: dict) -> int:
        """Bytes of the largest request `up` may send for this env: an endpoint create (an update sends a part of it),
        with its variable parts at their longest: the longest data center id, every GPU type of the model's pools that
        it may exclude, a long volume id."""
        excluded = sorted({i for p in spec.gpu_pools for i in pool_types(gpus, p, spec.min_container_ram_gb)[1]})
        gpu = {"pools": [max(spec.gpu_pools, key=len)], "count": 1, "minCudaVersion": spec_cuda(spec) or "",
               "excludedTypes": excluded}
        dc = max([d.get("id") or "" for d in dcs], key=len, default="")
        name = make_name(model, self.owner or new_owner())
        return len(encode(self._endpoint_body(name, spec, env, gpu, dc, "v" * 32, want)))

    def _check_size(self, model: str, spec, env: dict, gpus: list[dict], dcs: list[dict], want: dict) -> None:
        size = self._largest_request(model, spec, env, gpus, dcs, want)
        if size > MAX_REQUEST_BYTES:
            family = getattr(spec, "family", "openpi")
            raise LauncherError(
                f"{model}: the endpoint request would be {size:,} bytes, over the {MAX_REQUEST_BYTES:,} that Runpod "
                f"accepts (it answers HTTP 413); the worker files (RS_FILES) are {len(env.get('RS_FILES', '')):,} "
                f"bytes of it. Nothing was created. This is a size limit of this version of Playground Serve for "
                f"{model}, not of your account: please report it at {ISSUES}"
                + ("" if family == "openpi" else f" (the files of families/{family}/ are too large)") + ".")

    def _order(self, cands: list[Candidate], endpoints: list[dict], owner: str):
        """(cands, data centers to probe, how the first was chosen). Within each pool: data centers measured in the
        last day by round trip (region.py), then the rest by distance from this machine's time zone in BAND_KM steps
        (region.here and region.band; same band: more stock first), else stock order. Nothing is measured here: the
        probe runs while the new endpoint's worker boots (_probe_start), for the first pool's unmeasured data centers,
        nearest first."""
        for x in region.leftovers(endpoints, owner):
            self.out(f"Deleting probe endpoint {x['name']} ({x['id']}) left by an interrupted `pgs up`.")
            self.api.delete_endpoint(x["id"])
        cache, now, origin = self.data.setdefault("probes", {}), self.now(), self.here()
        known = {c.dc: region.fresh(cache, c.dc, now) for c in cands}
        pools = list(dict.fromkeys(c.pool for c in cands))
        cands = sorted(cands, key=lambda c: (pools.index(c.pool), known[c.dc] is None, known[c.dc] or 0,
                                             region.band(origin, c.dc), -c.stock))
        todo = [d for d in dict.fromkeys(c.dc for c in cands if c.pool == cands[0].pool) if known[d] is None]
        how = ("round trip from here not measured yet; nearest with stock to this machine's time zone"
               if origin else "round trip from here not measured yet; most stock")
        return cands, todo, how

    def _probe_start(self, todo: list[str], endpoints: list[dict], gpus):
        """Probe the data centers in `todo` (as many as the account's worker quota has free slots next to the new
        endpoint) on a thread while the worker boots. Returns finish(stop=False, model=None, ep=None) -> lines to print:
        it lets running probes end (at most PROBE_GRACE_S more unless `stop`), caches what was measured, and says how
        near the endpoint's data center is and whether a measured one is nearer."""
        owner = self.owner
        endpoints = [e for e in endpoints if not region.is_probe(e.get("name"), owner)]
        todo = todo[:max(0, region.free_slots(endpoints))]
        if not todo:
            return None
        lines, got, halt = [], {}, threading.Event()

        def run():
            try:
                try:
                    with_cpu = region.cpu_dcs(self.api.serverless_cpus())
                except ApiError:
                    with_cpu = set(todo)  # unknown: try CPU probes
                pools = {d: p for d in todo if d not in with_cpu and (p := region.gpu_pool(list(gpus), d))}
                got.update(region.probe(self.api, owner, todo, len(todo), out=lines.append, clock=self.clock,
                                        sleep=self.sleep, measure=self.measure, stop=halt,
                                        pools={d: p for d, (p, _) in pools.items()}))
            except Exception as e:  # noqa: BLE001  (a probe never stops `up`; its endpoints are deleted in region.probe)
                lines.append(f"  region probe failed: {e}")

        thread = threading.Thread(target=run, name="region-probe", daemon=True)
        thread.start()

        def finish(stop: bool = False, model: str | None = None, ep: dict | None = None) -> list[str]:
            thread.join(0 if stop else PROBE_GRACE_S)
            halt.set()
            thread.join()
            cache = self.data.setdefault("probes", {})
            for d, ms in got.items():
                if ms is not None:
                    cache[d] = {"ms": ms, "at": self.now()}
            self._save()
            if ep is None:
                return lines
            measured = sorted((ms, d) for d, ms in got.items() if ms is not None)
            out = [*lines, "Round trip from here through Runpod's gateway (median of 20 small messages), measured "
                   "while the worker started: " + (", ".join(f"{d} {ms:.1f} ms" for ms, d in measured) or "none")
                   + "".join(f"; {d} not measured" for d in todo if got.get(d) is None) + "."]
            mine = (ep.get("dataCenterIds") or [None])[0]
            if measured and got.get(mine) is not None and measured[0][1] != mine and \
                    measured[0][0] <= got[mine] - NEARER_MS:
                out.append(f"{measured[0][1]} is nearer than {mine} ({measured[0][0]:.1f} against {got[mine]:.1f} "
                           f"ms): `pgs delete {model} --volume`, then `pgs up {model}` moves it there (a first start "
                           "again).")
            return out

        return finish

    def _drop_volume(self, model, vol, why) -> None:
        self.out(f"  deleting the empty volume {vol['name']} ({vol['id']}) created a moment ago: {why}")
        self.api.delete_volume(vol["id"])
        self.data["models"].get(model, {}).pop("volume", None)
        if not self.data["models"].get(model):
            self.data["models"].pop(model, None)
        self._save()

    def _bring_up(self, spec, env, ep, gpus, dcs, want):
        patch = {}
        container = self._container(spec, env)
        for k in ("image", "ports", "disk", "env"):
            if ep.get(k) != container[k]:
                patch[k] = container[k]  # env: always the complete map (a partial env replaces the whole map)
        if ep.get("entrypoint") != container["entrypoint"] or ep.get("cmd") != container["cmd"]:
            patch["entrypoint"], patch["cmd"] = container["entrypoint"], container["cmd"]
        pool = ((ep.get("gpu") or {}).get("pools") or [None])[0]
        dc = (ep.get("dataCenterIds") or [None])[0]
        current = ep.get("workers") or {}
        ok, _ = pool_types(gpus, pool, spec.min_container_ram_gb)
        price = max([g.get("price", {}).get("serverless") or 0.0 for g in ok] or [0.0])
        if current.get("min", 0) < 1:  # no worker kept: check the pool still has stock where the volume is
            cands = candidates(gpus, dcs, spec, only_dc=dc)
            if cands and not any(c.pool == pool for c in cands):
                c = cands[0]
                self.out(f"No serverless stock for pool {pool} in {dc} now; switching to {c.label()}.")
                patch["gpu"], price = self._gpu(spec, c), c.price
            elif not cands:
                self.out(f"No serverless stock for pools {', '.join(spec.gpu_pools)} in {dc} now; the worker waits "
                         "for capacity.")
            # The readiness requests wake a worker through the gateway, and `want` (min 1 for --warm) is sent once it
            # serves. Setting min 1 first left the gateway routing to the restarted worker 21 to 26 s after it was
            # ready in 4 of 4 tries; this way 2 to 9 s in 4 of 5 and 22 s once.
            self.out(f"Endpoint {ep['name']} ({ep['id']}): a worker starts if none runs, {price:.2f} USD/hr while it "
                     f"runs; {stops(want)}.")
        elif not same(current, want):
            patch["workers"] = dict(want)  # a worker is kept now: change at once (min 0 lets it stop once idle)
        if any(k in patch for k in ("env", "image", "cmd", "ports", "disk")):
            self.out("Updating the endpoint's worker settings (new release; the worker restarts).")
        if patch:
            ep = self._update(ep, patch)
        return ep, price

    def _update(self, ep: dict, patch: dict) -> dict:
        try:
            self.api.update_endpoint(ep["id"], patch)
        except ApiError as e:
            raise (self._permission(e) if e.status in (401, 403) else LauncherError(str(e))) from None
        ep.update(patch)
        return ep

    # ----- waiting for the server
    def _busy_worker(self, ep: dict) -> dict | None:
        workers = (self.api.endpoint_workers(ep["id"]) or {}).get("workers") or []
        # The platform lists extra workers beyond max (IDLE, INITIALIZING) that never start; the busy one matters.
        order = {"RUNNING": 0, "UNHEALTHY": 1, "INITIALIZING": 2, "IDLE": 3, "THROTTLED": 4}
        return min(workers, key=lambda w: (order.get(w.get("status"), 9), bool(w.get("isStale")))) if workers else None

    def _boot_log(self, ep: dict, w: dict) -> list[dict]:
        """The phase lines the worker's current container logged. The recent log, not only lines since startedAt: a
        FlashBoot-resumed container keeps its process and prints nothing new. The last "boot" starts the sequence."""
        lines = [d for d in map(rs_line, self.api.worker_log_lines(ep["id"], w["id"], tail=1000, wait=LOG_READ_S)) if d]
        boots = [i for i, d in enumerate(lines) if d["rs"] == "boot"]
        return lines[boots[-1]:] if boots else lines

    def _boot_phases(self, ep: dict, w: dict) -> list[str]:
        return [d["rs"] for d in self._boot_log(ep, w)]

    def _worker_phase(self, ep: dict) -> tuple[str, bool, str | None]:
        """(what the worker is doing, whether its log says ready, the release id its boot line names)."""
        w = self._busy_worker(ep)
        if w is None:
            return "waiting for Runpod to assign a GPU worker", False, None
        status = w.get("status")
        if status == "THROTTLED":
            return "waiting for GPU capacity", False, None
        if status == "INITIALIZING" or not w.get("startedAt"):
            return "starting the container (the first time on a host it pulls the image, about 2 minutes)", False, None
        log = self._boot_log(ep, w)
        if not log:  # nothing yet, or a long build's output pushed the phase lines out of the tail
            return None, False, None
        current = [d["rs"] for d in log]
        last = current[-1]
        if last.startswith(("ws_", "fill_")) or last == "selftest_ws":
            last = "ready"
        boot = log[0] if current[0] == "boot" else {}
        return PHASES.get(last, last), "ready" in current, boot.get("release_id")

    def _served_release(self, ep: dict) -> str | None:
        """The release id in the metadata of the worker the gateway connects a client to ("" if it sends none), or
        None if no metadata came."""
        import msgpack

        try:
            ws = self.api.websocket(ws_url(ep), timeout=10)
        except Exception:  # noqa: BLE001  (no worker assigned yet, a timeout, a reset)
            return None
        try:
            meta = msgpack.unpackb(ws.recv(timeout=10), strict_map_key=False)
            return str(((meta if isinstance(meta, dict) else {}).get("playground_serve") or {}).get("release") or "")
        except Exception:  # noqa: BLE001
            return None
        finally:
            ws.close()

    def _recycle(self, ep: dict, after: dict, served: str | None, release: str) -> None:
        """A worker of another release serves (a client kept it after the release change, or Runpod started it
        again): stop every worker of the endpoint (workers max 0; listed gone 5 s later when measured), then allow
        `after` again, so that the next worker starts on this release. A connected client reconnects to it."""
        self.out(f"  The worker that answers runs another release ({served or 'one without a release id'}), not this "
                 f"one ({release}): stopping it so that a worker of this release starts.")
        self._update(ep, {"workers": {**after, "min": 0, "max": 0}})
        end = self.clock() + RECYCLE_WAIT_S
        while self.clock() < end and any(
                w.get("status") in ("RUNNING", "INITIALIZING", "UNHEALTHY", "THROTTLED")
                for w in (self.api.endpoint_workers(ep["id"]) or {}).get("workers") or []):
            self.sleep(3)
        self._update(ep, {"workers": dict(after)})

    def _wait_ready(self, ep: dict, timeout: float, t0: float, want: dict, release: str | None = None) -> None:
        """Wait until the server answers through the gateway (or, for a key without gateway access, until its log says
        ready; then no request can wake a worker, so min 1 is set at once and `up` sends `want` afterwards). With a
        `release`, only a worker that reports it counts; one of another release is stopped (_recycle)."""
        # Through the gateway every path reaches the worker's PORT (the websocket server), not PORT_HEALTH: /ping gets
        # 426 from websockets there, while openpi's server answers /healthz with 200. Either means it is serving. One
        # short request is kept in flight (the gateway holds it while the worker starts but may not hand it over when
        # the worker turns ready), and the worker log, a blocking read, only every PROGRESS_EVERY_S for the display.
        health = base_url(ep) + "/healthz"
        use_gateway, shown, shown_at, read_at = True, None, self.clock(), None
        while True:
            if use_gateway:
                asked = self.clock()
                code = self.api.gateway_status(health, timeout=GATEWAY_HOLD_S)
                if code in (200, 426):
                    served = self._served_release(ep) if release else None
                    if served == release:
                        return
                    if served is None:  # serving, but the metadata did not come through: ask again in a second
                        self.sleep(1)
                    else:
                        self._recycle(ep, {**want, "min": 0}, served, release)
                elif code in (401, 403):
                    use_gateway = False
                    self.out(f"  This key cannot call the endpoint (gateway HTTP {code}): the robot client needs a key "
                             "with endpoint access. Reading readiness from the worker log instead.")
                    if (ep.get("workers") or {}).get("min", 0) < 1:  # no request can wake a worker: min 1 now
                        self._update(ep, {"workers": {**want, "min": 1}})
                elif self.clock() - asked < 1:  # answered at once (no worker yet), not held: ask again in a second
                    self.sleep(1)
            elapsed = self.clock() - t0
            if use_gateway and read_at is not None and self.clock() - read_at < PROGRESS_EVERY_S and elapsed <= timeout:
                continue
            phase, ready, logged = self._worker_phase(ep)
            read_at = self.clock()
            if ready and not use_gateway:
                if release is None or logged == release:
                    return
                self._recycle(ep, {**want, "min": 1}, logged, release)
                continue
            phase = phase or shown or "container running, waiting for the server's first log line"
            if elapsed > timeout:
                self._update(ep, {"workers": dict(DOWN)})
                raise LauncherError(f"Not ready after {fmt_elapsed(elapsed)}, the limit of `pgs up` for this start "
                                    f"(last phase: {phase}). Scaled the endpoint to zero to stop billing. A slow host "
                                    f"or volume is the usual cause: `pgs up {self._model_of(ep)}` starts a new worker "
                                    "(often on another host). Worker logs: "
                                    f"https://console.runpod.io/serverless/user/endpoint/{ep['id']}")
            if phase != shown or self.clock() - shown_at >= 60:
                self.out(f"  [{fmt_elapsed(elapsed)}] {phase}")
                shown, shown_at = phase, self.clock()
            if not use_gateway:
                self.sleep(self.poll)

    def _model_of(self, ep: dict) -> str:
        """The model name this state file records for an endpoint (its slug from the name if not recorded)."""
        for model, rec in self.data["models"].items():
            if (rec.get("endpoint") or {}).get("id") == ep.get("id"):
                return model
        return (parse_name(ep.get("name")) or (None, "MODEL"))[1]

    def _wait_fill(self, ep: dict) -> None:
        """A first start saves the environment and checkpoint to the volume after the server is ready (worker
        stage.py fill). Scaling to zero before that ends throws the copy away and the next start builds them again,
        so wait for it; Ctrl-C scales down at once."""
        t0, shown_at = self.clock(), None
        while True:
            w = self._busy_worker(ep)
            if w is None or w.get("status") != "RUNNING":
                return
            phases = self._boot_phases(ep, w)
            if "fill_start" not in phases or "fill_done" in phases:
                return
            elapsed = self.clock() - t0
            if elapsed > FILL_TIMEOUT_S:
                self.out(f"  Still saving after {fmt_elapsed(elapsed)}; scaling down anyway.")
                return
            if shown_at is None:
                self.out("  The worker is still saving the environment and checkpoint to the volume; waiting so that "
                         "the next start is fast (Ctrl-C scales down now; the next start then builds them again).")
                shown_at = self.clock()
            elif self.clock() - shown_at >= 60:
                self.out(f"  [{fmt_elapsed(elapsed)}] still saving to the volume")
                shown_at = self.clock()
            self.sleep(self.poll * 2)

    # ----- down, status, delete
    def _models(self, endpoints: list[dict], volumes: list[dict], anyone: bool = False) -> list[str]:
        """Models in this state file plus those named on the account: by this state's owner id, or (anyone) by any."""
        by_slug = {slug(m): m for m in [*self._names(), *self.data["models"]]}
        out = set(self.data["models"])
        for x in endpoints + volumes:
            p = parse_name(x.get("name"))
            if p and (anyone or (self.owner and p[0] == self.owner)):
                out.add(by_slug.get(p[1], p[1]))
        return sorted(out)

    def down(self, model: str | None = None) -> None:
        endpoints, volumes = self.api.list_endpoints(), self.api.list_volumes()
        models = [self._key(model)] if model else self._models(endpoints, volumes)
        if not models:
            self.out("No Playground Serve endpoints of yours (`pgs status` also lists other owners').")
        for m in models:
            ep, vol, (their_eps, _) = self.reconcile(m, endpoints, volumes)
            if ep is None:
                self.out(f"{m}: no endpoint of yours." + "".join(
                    f" Not changed: endpoint {e['name']} ({e['id']}), {self._whose(e)}." for e in their_eps))
                continue
            if same(ep.get("workers"), DOWN):
                self.out(f"{m}: already down ({ep['name']}).")
            else:
                try:
                    self._wait_fill(ep)
                except KeyboardInterrupt:
                    self.out("  Not waiting.")
                self._update(ep, {"workers": dict(DOWN)})
                self.out(f"{m}: no worker kept on {ep['name']} ({ep['id']}), idle timeout 5 s: a running worker stops "
                         "within about a minute once no client is connected, and GPU billing stops then.")
            if vol:
                self.out(f"  volume {vol['name']} kept: about {volume_month_usd(vol):.2f} USD/month "
                         f"(`pgs delete {m} --volume` removes it). `pgs up {m}` starts again.")

    def status(self) -> None:
        """Read only: changes no resource and never writes the state file."""
        endpoints, volumes = self.api.list_endpoints(), self.api.list_volumes()
        models = self._models(endpoints, volumes, anyone=True)
        if not models:
            self.out("No Playground Serve endpoints. Start one with `pgs up <model>`.")
            return
        gpus = self.api.serverless_gpus()
        total = 0.0
        for m in models:
            ep, vol, (their_eps, their_vols) = self._find(m, endpoints, volumes)
            self.out(m)
            for kind, r in self._missing(m, endpoints, volumes):
                self.out(f"  {kind} {r.get('name')} ({r['id']}) from this state file is not in the account listing "
                         "(the next `pgs up`/`down`/`delete` checks it and forgets it if Runpod answers 404)")
            if ep:
                pool = ((ep.get("gpu") or {}).get("pools") or ["?"])[0]
                price = max([g.get("price", {}).get("serverless") or 0.0 for g in gpus if g.get("pool") == pool] or [0])
                info = self.api.endpoint_workers(ep["id"]) or {}
                states = [w.get("status") for w in info.get("workers") or []]
                w = ep.get("workers") or {}
                billed = sum(s in ("RUNNING", "INITIALIZING", "UNHEALTHY") for s in states)
                if w.get("min", 0) > 0:
                    billed += states.count("IDLE")
                billed = min(billed, w.get("max") or billed)  # extra listed workers beyond max do not bill
                now = billed * price
                total += now
                self.out(f"  endpoint {ep['name']} ({ep['id']}), {pool} in {(ep.get('dataCenterIds') or ['?'])[0]}, "
                         f"workers min {w.get('min')} max {w.get('max')}, idle timeout {idle_text(w.get('idleTimeout'))}")
                self.out(f"  worker state: {', '.join(states) or 'none'}; now about {now:.2f} USD/hr")
                self.out("  " + billing(m, w, price, short=True))
                self.out(f"  url {ws_url(ep)}")
            if vol:
                hourly = volume_month_usd(vol) / HOURS_PER_MONTH
                total += hourly
                self.out(f"  volume {vol['name']} ({vol['id']}), {vol.get('size')} GB {vol.get('type')} in "
                         f"{vol.get('dataCenter')}: about {volume_month_usd(vol):.2f} USD/month ({hourly:.3f} USD/hr)")
            for e in their_eps:
                w = e.get("workers") or {}
                self.out(f"  not yours: endpoint {e['name']} ({e['id']}), workers min {w.get('min')} max "
                         f"{w.get('max')}, {self._whose(e)}; `pgs` here does not change it")
            for v in their_vols:
                self.out(f"  not yours: volume {v['name']} ({v['id']}), {self._whose(v)}")
        self.out(f"Total now for yours: about {total:.3f} USD/hr")

    def delete(self, model: str, volume: bool = False, force: bool = False, yes: bool = False) -> None:
        """Deletes yours; another owner's endpoint (and with --volume its volume) only with `force`; asks first unless
        `yes`."""
        model = self._key(model)
        endpoints, volumes = self.api.list_endpoints(), self.api.list_volumes()
        ep, vol, (their_eps, their_vols) = self.reconcile(model, endpoints, volumes)
        if force and ep is None and their_eps:
            ep = newest(their_eps)
            vol = vol or next((v for v in volumes if v["id"] in mounted(ep)), None)
        if force and volume and vol is None and their_vols:
            vol = newest(their_vols)
        theirs = [("endpoint", e) for e in their_eps if not ep or e["id"] != ep["id"]]
        theirs += [("volume", v) for v in their_vols if volume and (not vol or v["id"] != vol["id"])]
        refused = "; ".join(f"{k} {x['name']} ({x['id']}, {self._whose(x)})" for k, x in theirs)
        what = []
        if ep:
            what.append(f"endpoint {ep['name']} ({ep['id']}" + ("" if self._is_mine(model, ep) else
                                                              f", not yours: {self._whose(ep)}") + ")")
        if volume and vol:
            what.append(f"volume {vol['name']} ({vol['id']}, {vol.get('size')} GB: the built environment, checkpoint "
                        "and caches; the next `up` rebuilds them" + ("" if self._is_mine(model, vol) else
                                                                    f"; not yours: {self._whose(vol)}") + ")")
        if not what:
            if refused:
                raise LauncherError(f"{model}: nothing of yours to delete. Not deleted: {refused}. "
                                    f"`pgs delete {model} --force{' --volume' if volume else ''}` deletes "
                                    f"{'them' if len(theirs) > 1 else 'it'} (asks first).")
            self.out(f"{model}: nothing to delete." + ("" if volume or not vol else
                                                       f" The volume {vol['name']} remains; add --volume to delete it."))
            return
        if refused:
            self.out(f"Not yours, not deleted: {refused}.")
        if not yes and not self.confirm(f"Delete {' and '.join(what)}? This cannot be undone. [y/N] "):
            self.out("Nothing deleted.")
            return
        rec = self.data["models"].get(model, {})
        if ep:
            self.api.delete_endpoint(ep["id"])
            if rec.get("endpoint", {}).get("id") == ep["id"]:
                rec.pop("endpoint")
            self._save()
            self.out(f"Deleted endpoint {ep['name']}.")
        if volume and vol:
            for attempt in range(12):  # the volume stays attached for a few seconds after the endpoint delete
                try:
                    self.api.delete_volume(vol["id"])
                    break
                except ApiError as e:
                    if e.status not in (400, 409) or attempt == 11:
                        raise
                    self.sleep(5)
            if rec.get("volume", {}).get("id") == vol["id"]:
                rec.pop("volume")
            self.out(f"Deleted volume {vol['name']}.")
        elif vol:
            self.out(f"Volume {vol['name']} kept: about {volume_month_usd(vol):.2f} USD/month. "
                     f"`pgs delete {model} --volume` removes it.")
        if not self.data["models"].get(model):
            self.data["models"].pop(model, None)
        self._save()

    def _is_mine(self, model: str, x: dict) -> bool:
        p = parse_name(x.get("name"))
        return any(r.get("id") == x["id"] for r in self._rec(model).values()) or bool(
            p and self.owner and p[0] == self.owner)


def billing(model: str, w: dict, price: float, short: bool = False) -> str:
    """In plain words: what bills and when it stops, for an endpoint's workers settings `w`."""
    if (w.get("min") or 0) >= 1:
        return (f"billing: 1 worker kept running (--warm): {price:.2f} USD/hr with or without clients, until "
                f"`pgs down {model}`" if short else
                f"Billing: 1 worker is kept running (--warm): {price:.2f} USD/hr with or without clients, until "
                f"`pgs down {model}`.")
    idle = idle_text(w.get("idleTimeout"))
    if short:
        return (f"billing: {price:.2f} USD/hr only while a worker runs; a worker stops by itself {idle} after its last "
                "client disconnects (no worker is kept)")
    return (f"Billing: the GPU bills {price:.2f} USD/hr only while a worker runs. The worker started now stops by itself "
            f"{idle} after the last request if no client connects. A connected client keeps it running; it stops {idle} "
            f"after the last client disconnects, and a later connect starts one again (about 45 to 75 s to the first "
            f"action). `pgs up {model} --warm` keeps one running until `pgs down {model}`.")
