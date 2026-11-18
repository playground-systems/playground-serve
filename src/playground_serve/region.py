"""Nearest data center for `pgs up`: a guess from this machine's time zone, then round trips measured through Runpod's
gateway.

A robot's round trip depends on where its worker runs: through the gateway an empty message took 19 ms to US-MD-1,
25 ms to US-NC-1 and about 55 ms to US-IL-1 from a client on the US east coast, and moving the model from US-IL-1 to
US-NC-1 took 24 ms off every request (docs/BENCHMARKS.md). The gateway's TCP and TLS end at a Cloudflare
edge near the client, and Runpod's per-data-center hosts (S3 API) answered 2x slower or faster than a worker there, so
only a worker in the data center measures it. So `probe` gives each candidate data center a short-lived CPU endpoint
whose worker echoes websocket messages (PROBE_SERVER, standard library only, on a slim Python image), times ECHOES
small messages through the gateway, and deletes the endpoint at once. Results are cached in the state file for a day.
Where the catalog lists no serverless CPU worker in a data center (US-TX-3 and CA-MTL-3 on 2026-11-14; US-NC-1 and
US-IL-1 came and went that day, and CPU probes there got a worker in some runs and none in 60 s in others), the probe
runs on the cheapest GPU pool with stock there instead: the same echo, no GPU used, at that pool's price. A probe
answered 18 to 51 s after its create, most of it the gateway's lag before it routes to a new worker; all probes run at
once and get PROBE_WAIT_S. Done before the model's endpoint, that was about a minute of the first `pgs up` (2:58 to
3:49 in all), so `pgs up` no longer waits for it: a new endpoint goes to the data
center nearest to this machine's time zone (`here`, `band`: read locally, nothing sent anywhere) unless a probe of the
last day measured otherwise, and the probe runs while the model's worker boots, for the next endpoint and a hint.

Limits kept: creates go one at a time; the account's serverless quota caps the sum of max workers over all its
endpoints (5 on the account this was built on, QUOTA), so at most as many probes as there are free slots run (each with
max 1 worker) next to the model's endpoint. A CPU worker bills about 0.07 USD/hr, a GPU one 0.6 to 1.4 USD/hr, and a
probe runs under a minute: a few cents at most. Every probe endpoint is named playground-serve-<owner>-probe-<dc>-<6
hex> and deleted by id when its measurement ends, also on errors and Ctrl-C; one that an interrupted run left behind is
deleted by the next run.
"""

from __future__ import annotations

import base64
import concurrent.futures
import math
import os
import re
import secrets
import socket
import statistics
import threading
import time

QUOTA = 5  # max workers summed over all endpoints of the account (HTTP 400 beyond it)
CACHE_S = 24 * 3600
ECHOES = 20
PROBE_WAIT_S = 50  # all probes: from the first create to the first echo (18 to 47 s measured, mostly the gateway's lag;
# 50 s ends them before a first start's worker is ready, about 60 s in, so `pgs up` does not wait for them)
PROBE_IMAGE = "python:3.12-slim"
PROBE_CPU = "cpu3c"  # compute-optimized, 2 vCPU: the cheapest serverless CPU type (0.036 USD/hr per vCPU)
# A websocket echo for the load balancer: GET on any path answers 200 (the health check is /ping on PORT_HEALTH, the
# same port); each data frame up to 125 bytes comes back as it was sent, a ping gets a pong.
PROBE_SERVER = r'''
import asyncio, base64, hashlib, os, struct
async def handle(r, w):
    try:
        head = (await r.readuntil(b"\r\n\r\n")).decode("latin1").split("\r\n")
        h = {k.strip().lower(): v.strip() for k, _, v in (x.partition(":") for x in head[1:] if x)}
        if "sec-websocket-key" not in h:
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            return await w.drain()
        key = h["sec-websocket-key"].encode() + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        w.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                b"Sec-WebSocket-Accept: " + base64.b64encode(hashlib.sha1(key).digest()) + b"\r\n\r\n")
        while True:
            b0, b1 = await r.readexactly(2)
            n = b1 & 127
            if n >= 126:
                n = int.from_bytes(await r.readexactly(2 if n == 126 else 8), "big")
            mask = await r.readexactly(4) if b1 & 128 else bytes(4)
            data = bytes(c ^ mask[i % 4] for i, c in enumerate(await r.readexactly(n)))
            op = b0 & 15
            if op == 8:
                break
            if op in (1, 2, 9) and n < 126:
                w.write(bytes([128 | (10 if op == 9 else op), n]) + data)
                await w.drain()
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        pass
    finally:
        w.close()
async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", int(os.environ.get("PORT", "8000")))
    async with server:
        await server.serve_forever()
asyncio.run(main())
'''
PROBE_NAME = re.compile(r"playground-serve-(o[0-9a-f]{8})-probe-[a-z0-9-]+-[0-9a-f]{6}")
# Where Runpod's data centers are, roughly: the state or country in the id (the catalog gives no location), its largest
# city. Good to a few hundred km, which is what `band` needs.
PLACES = {"US-NC": (35.8, -78.6), "US-IL": (41.9, -87.6), "US-TX": (32.8, -96.8), "US-CA": (37.4, -121.9),
          "US-GA": (33.7, -84.4), "US-KS": (39.1, -94.6), "US-MO": (38.6, -90.2), "US-WA": (47.6, -122.3),
          "US-CO": (39.7, -105.0), "US-NE": (41.3, -96.0), "US-PA": (40.0, -75.2), "US-MD": (39.3, -76.6),
          "US-DE": (39.7, -75.5), "CA-MTL": (45.5, -73.6), "EU-RO": (44.4, 26.1), "EU-CZ": (50.1, 14.4),
          "EU-FR": (48.9, 2.4), "EU-NL": (52.4, 4.9), "EU-SE": (59.3, 18.1), "EUR-IS": (64.1, -21.9),
          "EUR-NO": (59.9, 10.8), "AP-JP": (35.7, 139.7), "AP-IN": (19.1, 72.9), "OC-AU": (-33.9, 151.2)}
BAND_KM = 1000  # data centers this close to each other in distance from here count as equally near: stock decides
ZONE_TABS = ("/usr/share/zoneinfo/zone1970.tab", "/usr/share/zoneinfo/zone.tab")


def here(tz: str | None = None, tabs=ZONE_TABS) -> tuple[float, float] | None:
    """(lat, lon) of this machine's time zone (its city in the tz database's zone table: New York for
    America/New_York), read from local files only; None where that is unknown (Windows, UTC)."""
    name = tz or os.environ.get("TZ", "").lstrip(":") or os.path.realpath("/etc/localtime").partition("zoneinfo/")[2]
    for path in tabs:
        try:
            with open(path) as f:
                rows = [line.split("\t") for line in f if not line.startswith("#")]
        except OSError:
            continue
        for row in rows:
            m = len(row) > 2 and row[2].strip() == name and re.fullmatch(
                r"([+-])(\d\d)(\d\d)(\d\d)?([+-])(\d\d\d)(\d\d)(\d\d)?", row[1])
            if m:
                lat = (int(m[2]) + int(m[3]) / 60 + int(m[4] or 0) / 3600) * (-1 if m[1] == "-" else 1)
                return lat, (int(m[6]) + int(m[7]) / 60 + int(m[8] or 0) / 3600) * (-1 if m[5] == "-" else 1)
    return None


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance."""
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    return 6371 * math.acos(min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))


def band(origin: tuple[float, float] | None, dc: str) -> int:
    """How far `dc` is from `origin` in BAND_KM steps (0 when origin is unknown; far when the data center is)."""
    place = PLACES.get(dc.rsplit("-", 1)[0])
    if origin is None:
        return 0
    return int(km(origin, place) // BAND_KM) if place else 99


def probe_name(owner: str, dc: str) -> str:
    return f"playground-serve-{owner}-probe-{re.sub(r'[^a-z0-9]+', '-', dc.lower()).strip('-')}-{secrets.token_hex(3)}"


def is_probe(name: str | None, owner: str | None) -> bool:
    m = PROBE_NAME.fullmatch(name or "")
    return bool(m and owner and m[1] == owner)


def probe_body(name: str, dc: str, pool: str | None = None) -> dict:
    """A load-balancing endpoint in `dc` running PROBE_SERVER: on a CPU worker, or on GPU `pool` if one is given."""
    where = {"gpu": {"pools": [pool], "count": 1}} if pool else {"cpu": [{"id": PROBE_CPU, "vcpuCount": 2}]}
    return {"name": name, "type": "LOAD_BALANCER", "image": PROBE_IMAGE, "disk": 5, **where,
            "entrypoint": ["python3", "-u", "-c"], "cmd": ["import os,base64;exec(base64.b64decode(os.environ['PROBE_PY']))"],
            "env": {"PROBE_PY": base64.b64encode(PROBE_SERVER.encode()).decode(), "PORT": "8000", "PORT_HEALTH": "8000"},
            "ports": ["8000/http"], "dataCenterIds": [dc],
            "workers": {"min": 0, "max": 1, "idleTimeout": 5}, "scaling": {"type": "REQUEST_COUNT", "requestCount": 4},
            "flashboot": "OFF", "timeout": 60_000}


def cpu_dcs(cpus: list[dict]) -> set[str]:
    """Data centers where the catalog offers PROBE_CPU serverless workers."""
    return {d["id"] for c in cpus if c.get("id") == PROBE_CPU for d in c.get("dataCenters") or []
            if d.get("availability") not in (None, "NONE")}


def gpu_pool(gpus: list[dict], dc: str) -> tuple[str, float] | None:
    """(pool, USD/hr) of the cheapest serverless GPU with stock in `dc`, for a probe where no CPU worker is offered."""
    offers = [((g.get("price") or {}).get("serverless") or 99.0, g["pool"]) for g in gpus if g.get("pool")
              for d in g.get("dataCenters") or [] if d.get("id") == dc and d.get("availability") not in (None, "NONE")]
    return (min(offers)[1], min(offers)[0]) if offers else None


def free_slots(endpoints: list[dict]) -> int:
    """Worker slots of the account's quota that no endpoint holds."""
    return QUOTA - sum(int((e.get("workers") or {}).get("max") or 0) for e in endpoints)


def fresh(cache: dict, dc: str, now: float) -> float | None:
    """The cached round trip of `dc` if it was measured less than CACHE_S ago."""
    entry = (cache or {}).get(dc) or {}
    return entry.get("ms") if entry.get("ms") is not None and now - entry.get("at", 0) < CACHE_S else None


def echo_ms(api, url: str, until, n: int = ECHOES, clock=time.monotonic, sleep=time.sleep) -> float | None:
    """Median round trip (ms) of n small websocket messages to the probe at `url` (after 2 untimed ones), or None if
    no worker answered before until() (a clock() time). Each connect attempt is capped at 2.5 s: the gateway holds an
    upgrade while a worker starts but does not always hand it over once the worker is up."""
    while clock() < until():
        t = clock()
        try:
            ws = api.websocket(url, timeout=max(1.0, min(2.5, until() - t)))
        except Exception:  # noqa: BLE001  (no worker yet: 502, 400, timeouts, resets)
            if clock() - t < 1:
                sleep(1)
            continue
        try:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            msg, ms = b"playground-serve probe".ljust(64, b"."), []
            for i in range(n + 2):
                t0 = time.perf_counter()
                ws.send(msg)
                if ws.recv(timeout=10) != msg:
                    return None
                if i >= 2:
                    ms.append((time.perf_counter() - t0) * 1e3)
            return round(statistics.median(ms), 1)
        except Exception:  # noqa: BLE001
            return None
        finally:
            ws.close()
    return None


def probe(api, owner: str, dcs: list[str], slots: int, out=print, wait_s: float = PROBE_WAIT_S,
          clock=time.monotonic, sleep=time.sleep, measure=echo_ms, pools: dict | None = None,
          stop: threading.Event | None = None) -> dict[str, float | None]:
    """Round trip (ms, or None) to each of `dcs`, at most `slots` probes at a time; creates are one at a time and
    every probe endpoint is deleted when its measurement ends. `pools`: data center -> GPU pool for a GPU probe.
    `stop` set: no new probe starts and running measurements end at their next check."""
    results: dict[str, float | None] = {}
    if slots < 1 or not dcs:
        return results
    create_lock, made, made_lock = threading.Lock(), {}, threading.Lock()
    deadline = [clock() + wait_s]
    until = lambda: 0 if stop is not None and stop.is_set() else deadline[0]  # noqa: E731

    def one(dc: str) -> None:
        name = probe_name(owner, dc)
        if stop is not None and stop.is_set():
            return
        with create_lock:
            try:
                ep = api.create_endpoint(probe_body(name, dc, (pools or {}).get(dc)))
            except Exception as e:  # noqa: BLE001  (refused: quota, capacity, data center; or lost in transit)
                out(f"  {dc}: probe not created ({e})")
                results[dc] = None
                if getattr(e, "status", None) in (0, None) or getattr(e, "status", 0) >= 500:  # it may exist anyway
                    for x in [x for x in api.list_endpoints() if x.get("name") == name]:
                        delete(x)
                return
        with made_lock:
            made[ep["id"]] = ep
        try:
            base = ((ep.get("requestUrls") or {}).get("base") or f"https://{ep['id']}.api.runpod.ai").rstrip("/")
            try:
                ms = measure(api, base.replace("https://", "wss://", 1) + "/ws", until, clock=clock, sleep=sleep)
            except Exception as e:  # noqa: BLE001  (no measurement; the endpoint is still deleted below)
                out(f"  {dc}: probe failed ({e})")
                ms = None
            results[dc] = ms
            if ms is None and until():  # say what happened: no worker, or a worker the gateway did not route to in time
                try:
                    states = [w.get("status") for w in (api.endpoint_workers(ep["id"]) or {}).get("workers") or []]
                except Exception:  # noqa: BLE001
                    states = ["unknown"]
                out(f"  {dc}: no answer in {wait_s:.0f} s (probe worker: {', '.join(states) or 'none assigned'})")
        finally:
            delete(ep)

    def delete(ep: dict) -> None:
        for attempt in range(3):
            try:
                api.delete_endpoint(ep["id"])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    out(f"  could not delete probe endpoint {ep['name']} ({ep['id']}): {e}; the next `pgs up` "
                        "deletes it")
                    return
                sleep(2)
        with made_lock:
            made.pop(ep["id"], None)

    ex = concurrent.futures.ThreadPoolExecutor(slots)
    try:
        list(ex.map(one, dcs))
    except BaseException:  # Ctrl-C: running measurements stop at their next check
        deadline[0] = 0
        raise
    finally:  # nothing may stay behind
        ex.shutdown(wait=True)
        for ep in list(made.values()):
            delete(ep)
    return results


def leftovers(endpoints: list[dict], owner: str | None) -> list[dict]:
    """This owner's probe endpoints on the account (an interrupted probe left them)."""
    return [e for e in endpoints if is_probe(e.get("name"), owner)]
