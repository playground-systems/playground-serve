"""GPU bring-up of a model family before `pgs up <entry>` serves it (families abc_vla, gr00t_n17, xiaomi_r1).
Run from a laptop; the key comes from --key-file, $BRINGUP_KEY_FILE, $RUNPOD_API_KEY or ~/.config/playground-serve/key
(resolve.resolve_key) and is never printed.

    python -m playground_serve.families.abc_vla.bringup up FAMILY ENTRY [--pool POOL] [--dc DC] [--env K=V ...] [--dev]
    python -m playground_serve.families.abc_vla.bringup logs ENTRY [--grep TEXT] [--all]
    python -m playground_serve.families.abc_vla.bringup workers ENTRY
    python -m playground_serve.families.abc_vla.bringup exec ENTRY FILE.py     (endpoints made with --dev only)
    python -m playground_serve.families.abc_vla.bringup down ENTRY             (workers max 0: a running worker stops)
    python -m playground_serve.families.abc_vla.bringup delete ENTRY [--volume]
    python -m playground_serve.families.abc_vla.bringup ls                     (this state's resources on the account)

It does what `pgs up` does for an openpi model (launcher.py), in a few lines: a network volume and a load-balancing
endpoint named playground-serve-<owner>-<entry>-<hex> on the stock image, the worker files in RS_FILES (START_COMMAND
unpacks them and runs boot.sh), scale to zero (workers min 0, max 1, idle 5 min), FlashBoot off. The files are
families/common/{boot.sh, build_lib.sh, pgs_server.py, pgs_msgpack_numpy.py}, worker/{health.py, server_decode.py,
stage.py} and every worker file of the family folder (not spec.py, NOTES.md, bringup.py). The owner id and resource
ids live in a local state file (BRINGUP_STATE, default /tmp/pgsb/bringup_state.json); every mutation acts on those
ids only.

--dev bundles devhook.py as serve.py and the family's serve.py as family_serve.py: the worker then also answers
{"rs_dev_exec": "<python>"} with the code's printed output, run on the serving thread with the loaded policy in scope.
For bring-up measurements only; `pgs up` never ships it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import secrets
import sys
import time

from playground_serve import bundle
from playground_serve.resolve import resolve_key
from playground_serve.runpod_api import RunpodApi

HERE = pathlib.Path(__file__).resolve().parent
FAMILIES = HERE.parent
WORKER = FAMILIES.parent / "worker"
COMMON_FILES = ("boot.sh", "build_lib.sh", "pgs_server.py", "pgs_msgpack_numpy.py")
WORKER_FILES = ("health.py", "server_decode.py", "stage.py")
NOT_SHIPPED = {"spec.py", "NOTES.md", "bringup.py", "devhook.py", "__init__.py"}
STATE = pathlib.Path(os.environ.get("BRINGUP_STATE", "/tmp/pgsb/bringup_state.json"))
PORTS = ["8000/http", "8001/http"]
MAX_USD_PER_HOUR = 2.0


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"owner": os.environ.get("BRINGUP_OWNER") or "o" + secrets.token_hex(4), "entries": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def family_files(family: str, dev: bool = False) -> dict[str, bytes]:
    files = {n: (FAMILIES / "common" / n).read_bytes() for n in COMMON_FILES}
    files.update({n: (WORKER / n).read_bytes() for n in WORKER_FILES})
    for p in sorted((FAMILIES / family).iterdir()):
        if p.is_file() and p.name not in NOT_SHIPPED and not p.name.endswith(".pyc"):
            files[p.name] = p.read_bytes()
    if dev:
        files["family_serve.py"] = files.pop("serve.py")
        files["serve.py"] = (HERE / "devhook.py").read_bytes()
    return files


def spec_of(family: str, entry: str):
    return importlib.import_module(f"playground_serve.families.{family}.spec").SPECS[entry]


def endpoint_env(family: str, entry: str, dev: bool, extra: dict) -> dict:
    spec = spec_of(family, entry)
    env = {**spec.endpoint_env(), "RS_FAMILY": family, "RS_WEIGHTS": "/runpod-volume/pgs/weights",
           "RS_RELEASE": f"bringup-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}",
           "RS_FILES": bundle.pack(family_files(family, dev)), **extra}
    return bundle._checked(env)


def gpu_price(api: RunpodApi, pool: str, only: str | None = None) -> tuple[float, dict, list[str]]:
    """(hourly price, {dc: stock}, type ids to exclude) of a serverless pool, read now; `only` keeps one GPU type."""
    types = [g for g in api.serverless_gpus() if g.get("pool") == pool]
    excluded = [g["id"] for g in types if only and g["id"] != only]
    types = [g for g in types if g["id"] not in excluded]
    if not types:
        raise SystemExit(f"no serverless GPU types in pool {pool}" + (f" named {only}" if only else ""))
    price = max((g.get("price") or {}).get("serverless") or 0.0 for g in types)
    stock: dict[str, str] = {}
    for g in types:
        for d in g.get("dataCenters") or []:
            if d.get("availability") not in (None, "NONE"):
                stock[d["id"]] = d["availability"]
    return price, stock, excluded


def cmd_up(api: RunpodApi, st: dict, a) -> None:
    family, entry = a.family, a.entry
    spec = spec_of(family, entry)
    pool = a.pool or spec.gpu_pools[0]
    price, stock, excluded = gpu_price(api, pool, a.only)
    if price > a.max_usd:
        raise SystemExit(f"{pool} costs {price} USD/hr, over the {a.max_usd} USD/hr cap (--max-usd)")
    rec = st["entries"].setdefault(entry, {"family": family})
    extra = dict(kv.split("=", 1) for kv in a.env or [])
    for kv in a.secret_env or []:  # NAME=FILE: the value is read from FILE now and never printed
        name, path = kv.split("=", 1)
        extra[name] = pathlib.Path(path).expanduser().read_text().strip()
    env = endpoint_env(family, entry, a.dev, extra)
    print(f"RS_FILES {len(env['RS_FILES'])} bytes; pool {pool} at {price:.2f} USD/hr (read now); stock {stock}")
    vol = rec.get("volume")
    if vol is None and a.no_volume:  # a one-off measurement: no network volume; --dc any (comma-separated) or "all"
        dcs_wanted = list(stock) if a.dc == "all" else [d for d in (a.dc or "").split(",") if d in stock]
        if not dcs_wanted:
            raise SystemExit(f"--no-volume needs --dc with {pool} stock: {stock}")
        vol = {"id": None, "dc": ",".join(dcs_wanted)}
    if vol is None:
        dcs = {d["id"]: d.get("networkVolumeTypes") or [] for d in api.datacenters()}
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        cands = sorted((dc for dc in stock if dcs.get(dc) and (a.dc is None or dc == a.dc)),
                       key=lambda dc: order.get(stock[dc], 3))
        if not cands:
            raise SystemExit(f"no data center with {pool} stock and network volumes")
        dc = cands[0]
        vtype = "STANDARD" if "STANDARD" in dcs[dc] else dcs[dc][0]
        body = {"name": f"playground-serve-{st['owner']}-{entry.replace('_', '-')}-{secrets.token_hex(3)}",
                "size": spec.volume_gb, "dataCenter": dc, "type": vtype}
        v = api.create_volume(body)
        vol = rec["volume"] = {"id": v["id"], "name": v["name"], "dc": dc, "size": spec.volume_gb, "type": vtype}
        save_state(st)
        print(f"created volume {v['name']} ({v['id']}) {spec.volume_gb} GB {vtype} in {dc}")
    start = list(bundle.START_COMMAND)
    gpu = {"pools": [pool], "count": 1, "minCudaVersion": a.min_cuda}
    if excluded:
        gpu["excludedTypes"] = excluded
    workers = {"min": 0, "max": 1, "idleTimeout": a.idle}
    body = {"type": "LOAD_BALANCER", "image": spec.image, "entrypoint": start[:2], "cmd": start[2:], "ports": PORTS,
            "disk": 50, "env": env, "gpu": gpu, "dataCenterIds": vol["dc"].split(","),
            "mounts": {"network": [{"volumeId": vol["id"]}]}, "workers": workers,
            "scaling": {"type": "REQUEST_COUNT", "requestCount": 4}, "flashboot": "OFF", "timeout": 3_600_000}
    if not vol["id"]:
        del body["mounts"]
    ep = rec.get("endpoint")
    if ep is None:
        body["name"] = f"playground-serve-{st['owner']}-{entry.replace('_', '-')}-{secrets.token_hex(3)}"
        e = api.create_endpoint(body)
        ep = rec["endpoint"] = {"id": e["id"], "name": e["name"], "dc": vol["dc"], "pool": pool}
        print(f"created endpoint {e['name']} ({e['id']}) in {vol['dc']}, {pool} at {price:.2f} USD/hr")
    else:
        api.update_endpoint(ep["id"], {k: body[k] for k in ("image", "entrypoint", "cmd", "ports", "disk", "env", "gpu",
                                                             "workers")})
        ep["pool"] = pool
        print(f"updated endpoint {ep['name']} ({ep['id']}): new release {env['RS_RELEASE']}")
    rec.update(family=family, dev=a.dev, release=env["RS_RELEASE"], price=price)
    save_state(st)
    print(f"wss://{ep['id']}.api.runpod.ai/ws")


def worker_list(api: RunpodApi, ep_id: str) -> list[dict]:
    return (api.endpoint_workers(ep_id) or {}).get("workers") or []


def cmd_logs(api: RunpodApi, st: dict, a) -> None:
    ep = st["entries"][a.entry]["endpoint"]
    for w in worker_list(api, ep["id"]):
        print(f"# worker {w.get('id')} {w.get('status')} started {w.get('startedAt')}")
        for line in api.worker_log_lines(ep["id"], w["id"], tail=a.tail, wait=a.wait):
            if a.all or (a.grep and a.grep in line) or (not a.grep and line.lstrip().startswith(('{"rs"', "=="))):
                print(line[:a.width])


def cmd_workers(api: RunpodApi, st: dict, a) -> None:
    ep = st["entries"][a.entry]["endpoint"]
    print(json.dumps(api.get_endpoint(ep["id"]).get("workers"), default=str))
    for w in worker_list(api, ep["id"]):
        print(json.dumps({k: w.get(k) for k in ("id", "status", "startedAt", "gpuTypeId", "dataCenterId")}))


def ws_exec(url: str, key: str, code: str, timeout: float) -> str:
    """Send {"rs_dev_exec": code} to a --dev worker; return the printed output."""
    sys.path.insert(0, str(FAMILIES / "common"))
    import pgs_msgpack_numpy as msgpack_numpy
    from websockets.sync.client import connect

    deadline = time.monotonic() + 600
    while True:
        try:
            ws = connect(url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=10,
                         compression=None, max_size=None)
            break
        except Exception as e:  # noqa: BLE001  (cold start: the gateway refuses or times out until ready)
            if time.monotonic() > deadline:
                raise
            print(f"connect: {e!r}; retrying", file=sys.stderr)
            time.sleep(2)
    with ws:
        meta = msgpack_numpy.unpackb(ws.recv(timeout=60))
        print(f"# metadata path={meta.get('path')} reason={meta.get('path_reason')}", file=sys.stderr)
        ws.send(msgpack_numpy.packb({"rs_dev_exec": code}))
        reply = ws.recv(timeout=timeout)
    if isinstance(reply, str):
        return reply
    return msgpack_numpy.unpackb(reply).get("out", "")


def cmd_down(api: RunpodApi, st: dict, a) -> None:
    ep = st["entries"][a.entry]["endpoint"]
    api.update_endpoint(ep["id"], {"workers": {"min": 0, "max": 0, "idleTimeout": 5}})
    print(f"{ep['id']}: workers max 0")


def cmd_delete(api: RunpodApi, st: dict, a) -> None:
    rec = st["entries"][a.entry]
    if rec.get("endpoint"):
        api.delete_endpoint(rec["endpoint"]["id"])
        print(f"deleted endpoint {rec['endpoint']['id']}")
        rec.pop("endpoint")
    if a.volume and (rec.get("volume") or {}).get("id"):
        for _ in range(12):  # the volume stays attached for a moment after the endpoint is gone
            try:
                api.delete_volume(rec["volume"]["id"])
                break
            except Exception as e:  # noqa: BLE001
                print(f"volume delete: {e}; retrying in 10 s")
                time.sleep(10)
        print(f"deleted volume {rec['volume']['id']}")
        rec.pop("volume")
    save_state(st)


def cmd_ls(api: RunpodApi, st: dict) -> None:
    owner = st["owner"]
    eps = [e for e in api.list_endpoints() if owner in (e.get("name") or "")]
    vols = [v for v in api.list_volumes() if owner in (v.get("name") or "")]
    print(f"owner {owner}: {len(eps)} endpoints, {len(vols)} volumes")
    for e in eps:
        print(" endpoint", e["id"], e["name"], e.get("workers"))
    for v in vols:
        print(" volume", v["id"], v["name"], v.get("dataCenter"), v.get("size"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("up", "logs", "workers", "exec", "down", "delete", "ls"))
    p.add_argument("args", nargs="*")
    p.add_argument("--pool")
    p.add_argument("--only", help="one GPU type id of the pool (the others are excluded)")
    p.add_argument("--max-usd", type=float, default=MAX_USD_PER_HOUR, help="refuse pools above this hourly price")
    p.add_argument("--dc")
    p.add_argument("--env", action="append")
    p.add_argument("--secret-env", action="append", help="NAME=FILE: endpoint env NAME from FILE (for example a token)")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--idle", type=int, default=300)
    p.add_argument("--min-cuda", default="12.8")
    p.add_argument("--grep")
    p.add_argument("--all", action="store_true")
    p.add_argument("--tail", type=int, default=3000)
    p.add_argument("--wait", type=float, default=6)
    p.add_argument("--width", type=int, default=1500)
    p.add_argument("--volume", action="store_true")
    p.add_argument("--no-volume", action="store_true", help="up: no network volume (one-off measurements)")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--key-file", default=os.environ.get("BRINGUP_KEY_FILE"))
    a = p.parse_args(argv)
    key = resolve_key(None, os.path.expanduser(a.key_file) if a.key_file else None)
    if not key:
        raise SystemExit("no Runpod key: pass --key-file or set BRINGUP_KEY_FILE or RUNPOD_API_KEY")
    api, st = RunpodApi(key), load_state()
    save_state(st)
    if a.command == "up":
        a.family, a.entry = a.args
        cmd_up(api, st, a)
    elif a.command == "ls":
        cmd_ls(api, st)
    else:
        a.entry = a.args[0]
        if a.command == "logs":
            cmd_logs(api, st, a)
        elif a.command == "workers":
            cmd_workers(api, st, a)
        elif a.command == "exec":
            ep = st["entries"][a.entry]["endpoint"]
            code = pathlib.Path(a.args[1]).read_text()
            print(ws_exec(f"wss://{ep['id']}.api.runpod.ai/ws", key, code, a.timeout))
        elif a.command == "down":
            cmd_down(api, st, a)
        elif a.command == "delete":
            cmd_delete(api, st, a)


if __name__ == "__main__":
    main()
