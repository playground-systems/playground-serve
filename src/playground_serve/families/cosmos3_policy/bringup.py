"""GPU bring-up and measurement runs for this folder's family entries (and flex_pi, flux3_action); users serve them
with `pgs up ENTRY`. It creates the endpoint the way playground_serve.launcher does (one Runpod Serverless
load-balancing endpoint on the stock image, the env map of bundle.family_env, a network volume, scale to zero,
FlashBoot off) with this script's own owner id in every resource name, and adds what measurement runs need: verify runs
(--env RS_VERIFY=1), one endpoint per GPU pool (--slot), a worker kept until `down`, requests that alternate the
model path (rs_path). Run from a checkout:

    uv run python -m playground_serve.families.cosmos3_policy.bringup up ENTRY --pool AMPERE_48 [--dc DC]
    ... update ENTRY [--env K=V]   new release: the complete env map again (new worker files, settings)
    ... logs ENTRY [--grep TEXT]   the worker's container log (rs phase lines by default)
    ... act ENTRY [--n 5]          one connect through the gateway, requests with the entry's test observations
    ... down ENTRY                 workers max 0: a running worker stops
    ... delete ENTRY               deletes this entry's endpoint and volume (ids from this script's state file)
    ... mine                       every endpoint and volume on the account whose name carries this owner id

The Runpod key comes from RUNPOD_API_KEY or --key-file; it goes only into the Authorization header. State (owner id,
ids of what this script created): $PGS_BRINGUP_HOME/state.json (default ~/.config/playground-serve-bringup).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import pathlib
import sys
import time

import numpy as np

from playground_serve import bundle, launcher
from playground_serve.runpod_api import RunpodApi

# "Expanded upstream request exceeds the 102400 byte limit" (HTTP 413): measured 2026-11-16, a create request of
# 100,300 bytes was accepted and one of 100,400 refused; the rest of the request is about 0.8 kB.
MAX_ENV_TOTAL = 99_000
MAX_USD_HR = 2.0  # this project's price cap per GPU hour; --max-usd raises it for the faster-GPU measurements
HOME = pathlib.Path(os.environ.get("PGS_BRINGUP_HOME", pathlib.Path.home() / ".config" / "playground-serve-bringup"))


def spec_module(entry: str):
    for family in ("cosmos3_policy", "flex_pi", "flux3_action"):
        mod = importlib.import_module(f"playground_serve.families.{family}.spec")
        if entry in mod.SPECS:
            return mod
    raise SystemExit(f"unknown entry {entry}")


SECRETS: list[str] = []  # values never printed: show() replaces them


def show(line: str) -> str:
    for value in SECRETS:
        line = line.replace(value, "<redacted>")
    return line


def secret_env(args) -> dict[str, str]:
    """HF_TOKEN for a gated entry, read from --hf-token-file at run time; it goes only into the endpoint env."""
    if not args.hf_token_file:
        return {}
    token = pathlib.Path(os.path.expanduser(args.hf_token_file)).read_text().strip()
    SECRETS.append(token)
    return {"HF_TOKEN": token}


def endpoint_env(entry: str, release: str, overrides: dict[str, str]) -> dict[str, str]:
    from playground_serve import models

    spec = models.get(entry)  # the overrides go into the spec's env: RS_VERIFY=1 ships the verify* files
    env = bundle.family_env(dataclasses.replace(spec, env={**spec.env, **overrides}), release)
    env[launcher.RELEASE_KEY] = launcher.release_id(env)
    total = sum(len(k) + len(v) + 2 for k, v in env.items())
    if total > MAX_ENV_TOTAL:  # Runpod answered HTTP 413 for an endpoint update above this
        raise SystemExit(f"endpoint env is {total} bytes, over the {MAX_ENV_TOTAL} Runpod accepts")
    return bundle._checked(env)


def load_state() -> dict:
    try:
        return json.loads((HOME / "state.json").read_text())
    except OSError:
        return {"entries": {}}


def save_state(data: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "state.json").write_text(json.dumps(data, indent=1))


def owner(data: dict) -> str:
    if not data.get("owner"):
        data["owner"] = launcher.new_owner()
        save_state(data)
    return data["owner"]


def api_from(args) -> RunpodApi:
    key = os.environ.get("RUNPOD_API_KEY") or pathlib.Path(os.path.expanduser(args.key_file)).read_text()
    return RunpodApi(key.strip())


def pick(api: RunpodApi, spec, pool: str, dc: str | None, vol: dict | None, max_usd: float = MAX_USD_HR):
    gpus = api.serverless_gpus(launcher.spec_cuda(spec))
    pool_spec = dataclasses.replace(spec, gpu_pools=(pool,))
    cands = launcher.candidates(gpus, api.datacenters(), pool_spec, only_dc=vol["dataCenter"] if vol else dc)
    if not cands:
        raise SystemExit(f"no serverless stock for {pool}" + (f" in {vol['dataCenter']}" if vol else ""))
    c = cands[0]
    if c.price > max_usd:
        raise SystemExit(f"{pool} costs {c.price} USD/hr, over the {max_usd} cap")
    return c


def key(args) -> str:
    """State key and resource-name model part: the entry, plus --slot when one entry has several endpoints (one per
    GPU pool measured)."""
    return args.entry + (f"-{args.slot}" if args.slot else "")


def cmd_up(api, args, data) -> None:
    mod = spec_module(args.entry)
    spec = mod.SPECS[args.entry]
    rec = data["entries"].setdefault(key(args), {})
    own = owner(data)
    if not rec.get("volume") and args.volume_from:  # one volume for two endpoints in its data center
        rec["volume"] = {**data["entries"][args.volume_from]["volume"], "shared_with": args.volume_from}
    vol = api.get_volume(rec["volume"]["id"]) if rec.get("volume") else None
    c = pick(api, spec, args.pool, args.dc, vol, args.max_usd)
    if args.gpu:  # only this GPU type of the pool
        gpus = api.serverless_gpus(launcher.spec_cuda(spec))
        c.excluded = sorted({*c.excluded, *(g["id"] for g in gpus if g.get("pool") == c.pool and g["id"] != args.gpu)})
        c.gpus = [args.gpu]
    print(f"GPU {c.label()}: {c.price:.2f} USD/hr while a worker runs (read {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})")
    if vol is None:
        vtype = "STANDARD" if "STANDARD" in c.volume_types else c.volume_types[0]
        body = {"name": launcher.make_name(key(args), own), "size": spec.volume_gb, "dataCenter": c.dc, "type": vtype}
        vol = api.create_volume(body)
        rec["volume"] = {"id": vol["id"], "name": vol["name"], "dc": c.dc, "size": spec.volume_gb, "type": vtype}
        save_state(data)
        print(f"created volume {vol['name']} {spec.volume_gb} GB {vtype} in {c.dc}")
    env = endpoint_env(args.entry, args.release, {**dict(kv.split("=", 1) for kv in args.env), **secret_env(args)})
    # A new endpoint with workers min 0 max 1 got a worker without any request (listed IDLE within a minute), so
    # without --warm it is created with max 0; `update --warm` starts one.
    want = {"min": 1 if args.warm else 0, "max": 1 if args.warm else 0, "idleTimeout": args.idle}
    if rec.get("endpoint"):
        ep = api.update_endpoint(rec["endpoint"]["id"], {"env": env, "workers": want})
        print(f"updated endpoint {rec['endpoint']['name']}: release {env[launcher.RELEASE_KEY]}")
        return
    gpu = {"pools": [c.pool], "count": 1, "minCudaVersion": launcher.spec_cuda(spec)}
    if c.excluded:
        gpu["excludedTypes"] = c.excluded
    body = {"name": launcher.make_name(key(args), own), "type": "LOAD_BALANCER", "image": spec.image,
            "entrypoint": bundle.START_COMMAND[:2], "cmd": bundle.START_COMMAND[2:], "ports": list(launcher.PORTS),
            "disk": launcher.DISK_GB, "env": env, "gpu": gpu, "dataCenterIds": [c.dc],
            "mounts": {"network": [{"volumeId": vol["id"]}]}, "workers": want, "scaling": dict(launcher.SCALING),
            "flashboot": launcher.FLASHBOOT, "timeout": launcher.TIMEOUT_MS}
    ep = api.create_endpoint(body)
    rec["endpoint"] = {"id": ep["id"], "name": ep["name"], "dc": c.dc, "pool": c.pool, "url": launcher.ws_url(ep)}
    save_state(data)
    print(f"created endpoint {ep['name']} in {c.dc}, release {env[launcher.RELEASE_KEY]}; no worker kept")


def cmd_update(api, args, data) -> None:
    """New env map, then a worker of the old release is stopped (workers max 0 until none is listed), as the
    launcher's _recycle does: Runpod otherwise keeps routing to a running worker of the old release."""
    rec = data["entries"][key(args)]
    ep = rec["endpoint"]["id"]
    env = endpoint_env(args.entry, args.release, {**dict(kv.split("=", 1) for kv in args.env), **secret_env(args)})
    api.update_endpoint(ep, {"env": env, "workers": {"min": 0, "max": 0, "idleTimeout": args.idle}})
    for _ in range(60):
        if not (api.endpoint_workers(ep) or {}).get("workers"):
            break
        time.sleep(3)
    api.update_endpoint(ep, {"workers": {"min": 1 if args.warm else 0, "max": 1, "idleTimeout": args.idle}})
    print(f"release {env[launcher.RELEASE_KEY]}" + (" (one worker kept: --warm)" if args.warm else ""))


def cmd_wait(api, args, data) -> None:
    """Until the worker logs ready (or fails): its rs phase lines as they come, without touching the gateway (a
    request through the gateway while no worker is ready made Runpod start up to three workers on a max-1 endpoint)."""
    ep = data["entries"][key(args)]["endpoint"]["id"]
    seen, t0 = set(), time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        workers = (api.endpoint_workers(ep) or {}).get("workers") or []
        for w in workers:
            for line in api.worker_log_lines(ep, w["id"], tail=3000, wait=4):
                seen_key = (w["id"], line)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                if line.lstrip().startswith('{"rs"') or "Traceback" in line or "Error" in line[:120]:
                    print(show(line[:args.width]), flush=True)
                if '"rs": "ready"' in line or '"rs": "selftest_ws"' in line:
                    return
        if len(workers) > 1:
            print(f"warning: {len(workers)} workers listed", flush=True)
        time.sleep(10)
    raise SystemExit("timeout")


def cmd_down(api, args, data) -> None:
    rec = data["entries"][key(args)]
    api.update_endpoint(rec["endpoint"]["id"], {"workers": {"min": 0, "max": 0, "idleTimeout": 5}})
    for _ in range(60):
        if not (api.endpoint_workers(rec["endpoint"]["id"]) or {}).get("workers"):
            break
        time.sleep(5)
    print("workers max 0, none listed")


def cmd_logs(api, args, data) -> None:
    ep = data["entries"][key(args)]["endpoint"]
    for w in (api.endpoint_workers(ep["id"]) or {}).get("workers") or []:
        print(f"# worker {w.get('status')} started {w.get('startedAt')} gpu {w.get('gpuTypeId') or w.get('gpu')}")
        for line in api.worker_log_lines(ep["id"], w["id"], tail=args.tail, wait=args.wait):
            if args.all or (args.grep and args.grep in line) or (not args.grep and line.lstrip().startswith('{"rs"')):
                print(show(line[:args.width]))


def cmd_workers(api, args, data) -> None:
    ep = data["entries"][key(args)]["endpoint"]
    print(json.dumps(api.endpoint_workers(ep["id"]), indent=1)[:4000])


def observations(entry: str) -> list[dict]:
    """The entry's test observations: real recorded frames when the family has them (NOTES.md), else the spec's."""
    mod = spec_module(entry)
    if hasattr(mod, "test_observations"):
        return mod.test_observations(entry)
    return [mod.example_observation(entry)]


def cmd_act(api, args, data) -> None:
    from playground_serve.client import client

    url = data["entries"][key(args)]["endpoint"]["url"]
    t0 = time.monotonic()
    policy = client.connect(url, api_key=api._key, mode=args.mode, wait=1800, keepalive=20, split=False)
    meta = {k: v for k, v in policy.metadata.items() if k not in ("playground_serve",)}
    print(json.dumps({"connect_s": round(time.monotonic() - t0, 1), "mode": policy.mode, "size": policy.size,
                      "metadata": meta}, default=str))
    obs_list = observations(args.entry)
    paths = args.paths.split(",") if args.paths else [None]  # verify runs: rs_path alternates per request
    out = []
    for i in range(args.n):
        for j, obs in enumerate(obs_list):
            for path in paths:
                seed = args.seed if args.seed is not None else None
                req = {**obs, **({"rs_seed": seed} if seed is not None else {}), **({"rs_path": path} if path else {})}
                reply = policy.infer(req)
                a = np.asarray(reply["actions"], np.float64)
                row = {"obs": j, "seed": seed, "path": path, "rtt_ms": round(policy.timing["rtt_ms"], 1),
                       "infer_ms": round(policy.timing["infer_ms"], 1), "shape": list(a.shape),
                       "finite": bool(np.isfinite(a).all())}
                if args.save:
                    out.append({**row, "actions": a.tolist(), "served_path": path or policy.metadata.get("path")})
                print(json.dumps(row))
    if args.save:
        pathlib.Path(args.save).write_text(json.dumps(out))
    policy.close()


def cmd_delete(api, args, data) -> None:
    rec = data["entries"].get(key(args), {})
    if rec.get("endpoint"):
        api.delete_endpoint(rec["endpoint"]["id"])
        print(f"deleted endpoint {rec['endpoint']['name']}")
        rec.pop("endpoint")
        save_state(data)
    if rec.get("volume") and rec["volume"].get("shared_with"):
        rec.pop("volume")  # deleted with the endpoint record that created it
        save_state(data)
    if rec.get("volume") and args.volume:
        for _ in range(12):  # a volume stays attached for a moment after its endpoint is deleted
            try:
                api.delete_volume(rec["volume"]["id"])
                break
            except Exception as e:  # noqa: BLE001
                print(f"volume delete: {e}; retrying")
                time.sleep(10)
        print(f"deleted volume {rec['volume']['name']}")
        rec.pop("volume")
        save_state(data)


def cmd_mine(api, args, data) -> None:
    own = data.get("owner") or "-"
    eps = [e for e in api.list_endpoints() if own in (e.get("name") or "")]
    vols = [v for v in api.list_volumes() if own in (v.get("name") or "")]
    print(json.dumps({"owner_in_names": own, "endpoints": [e.get("name") for e in eps],
                      "volumes": [v.get("name") for v in vols]}))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("up", "update", "wait", "down", "logs", "workers", "act", "delete", "mine"))
    p.add_argument("entry", nargs="?", default="")
    p.add_argument("--pool", default="AMPERE_48")
    p.add_argument("--volume-from", default="", help="state key of an endpoint whose volume to mount")
    p.add_argument("--gpu", default="", help="one GPU type id of the pool, e.g. 'NVIDIA L40S'")
    p.add_argument("--max-usd", type=float, default=MAX_USD_HR)
    p.add_argument("--hf-token-file", default="", help="gated entries: HF_TOKEN from this file, never printed")
    p.add_argument("--slot", default="", help="a second endpoint of the same entry, for example on another GPU pool")
    p.add_argument("--dc")
    p.add_argument("--idle", type=int, default=120)
    p.add_argument("--release", default="bringup-c")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--grep")
    p.add_argument("--all", action="store_true")
    p.add_argument("--tail", type=int, default=400)
    p.add_argument("--wait", type=float, default=6)
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", default=None)
    p.add_argument("--save")
    p.add_argument("--paths", help="verify runs: comma-separated rs_path values sent in turn, e.g. stock,fast")
    p.add_argument("--volume", action="store_true")
    p.add_argument("--warm", action="store_true")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--key-file", default=os.environ.get("RUNPOD_API_KEY_FILE", ""))
    args = p.parse_args(argv)
    api, data = api_from(args), load_state()
    secret_env(args)  # registers the token for redaction in every command
    {"up": cmd_up, "update": cmd_update, "wait": cmd_wait, "down": cmd_down, "logs": cmd_logs, "workers": cmd_workers, "act": cmd_act,
     "delete": cmd_delete, "mine": cmd_mine}[args.command](api, args, data)


if __name__ == "__main__":
    sys.exit(main())
