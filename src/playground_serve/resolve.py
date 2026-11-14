"""`connect(target)`: find the endpoint for a model (or take a URL), find the key, open the client. `status(target)`: is a
worker running there, read without starting one.

    from playground_serve import connect, status
    policy = connect("pi05_libero")
    actions = policy.act(obs)
    status("pi05_libero").state    # "warm", "cold" or "missing"

Target: a ws:// or wss:// URL is used as given. A model name, or no target, is looked up in this order: the launcher's
state file ($PLAYGROUND_SERVE_HOME/state.json, default ~/.config/playground-serve/state.json), the PLAYGROUND_SERVE_URL
environment variable, then the Runpod account by the endpoint name `pgs up` gives it, the caller's own (the owner id in
that state file) before other owners'. No target: the one endpoint there is, if there is exactly one. Only the account
lookup needs a key that may read the account; a key that may only call the endpoint works with the others.
Key: the `api_key` argument, the `key_file` argument, RUNPOD_API_KEY, then ~/.config/playground-serve/key. The key is
sent only in the Authorization header and never printed.

This module imports on Python 3.8 like the client; the launcher's modules are imported only for the account lookup.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

from . import __version__, state

URL_ENV = "PLAYGROUND_SERVE_URL"
WS_URL = re.compile(r"wss?://")
PASS_URL = (f'pass the endpoint URL that `pgs up` printed, connect("wss://<endpoint-id>.api.runpod.ai/ws"), or set '
            f"{URL_ENV} to it")
# Runpod's worker counts for one endpoint, answered by Runpod's API, not by a worker. A key without access to the
# endpoint got 403 here and one with access 200. It did not start a worker: polled every 15 s, it read "idle" 31 times
# in a row (7.5 min) and a worker ran only once a client connected.
HEALTH_URL = "https://api.runpod.ai/v2/{}/health"
# Seconds from a connect to its first action. Cold, a fresh container on the filled volume (RTX 4090, FlashBoot off):
# 42.7 to 119.1 s over ten starts, median 63 s; 50 to 75 s for a start after `pgs down` (docs/BENCHMARKS.md). Warm, a
# worker already serving: the socket opened in 0.15 to 0.54 s and an inference takes 0.2 to 0.5 s.
COLD_WAIT_S = (40.0, 120.0)
WARM_WAIT_S = (0.0, 1.0)


class EndpointNotFound(LookupError):
    """The target cannot be resolved to an endpoint; the message says what to do."""


class EndpointMissing(EndpointNotFound):
    """The account was read and holds no endpoint for the model."""


def resolve_key(api_key: str | None = None, key_file: str | os.PathLike | None = None) -> str | None:
    if api_key and api_key.strip():
        return api_key.strip()
    if key_file:
        key = pathlib.Path(key_file).expanduser().read_text().strip()
        if not key:
            raise ValueError(f"{key_file} is empty")
        return key
    if os.environ.get("RUNPOD_API_KEY", "").strip():
        return os.environ["RUNPOD_API_KEY"].strip()
    path = state.key_path()
    if path.is_file() and path.read_text().strip():
        return path.read_text().strip()
    return None


def _model_key(name: str) -> str:
    """The name `pgs up` stores and names resources by: the registry name, or custom-<config>-<hash> for CONFIG@CKPT."""
    try:
        from . import models

        return models.get(name).name
    except (KeyError, ValueError):
        return name


def _state_url(endpoint: dict) -> str:
    return endpoint.get("url") or f"wss://{endpoint['id']}.api.runpod.ai/ws"


def resolve_url(target: str | None, key: str | None, *, state_path=None, api_factory=None) -> str:
    if target and WS_URL.match(target):
        return target
    model = _model_key(target) if target else None
    data = state.load(state_path or state.default_path())
    managed = {m: r["endpoint"] for m, r in data["models"].items() if r.get("endpoint")}
    if model in managed:
        return _state_url(managed[model])
    if model is None and len(managed) == 1:
        return _state_url(next(iter(managed.values())))
    env = os.environ.get(URL_ENV, "").strip()
    if env:
        if not WS_URL.match(env):
            raise ValueError(f"{URL_ENV} must be a ws:// or wss:// URL, such as the one `pgs up` prints")
        return env
    if model is None and len(managed) > 1:
        raise EndpointNotFound(f"several endpoints are up ({', '.join(sorted(managed))}); name one, for example "
                               f'connect("{sorted(managed)[0]}")')
    return _account_url(model, key, api_factory, data.get("owner"))


def _account_url(model: str | None, key: str | None, api_factory, owner: str | None = None) -> str:
    if not key:
        raise EndpointNotFound(f"{model or 'no endpoint'} is not in this machine's state file and there is no Runpod "
                               f"API key to look it up: {PASS_URL}, or pass api_key= or key_file=, set RUNPOD_API_KEY, "
                               f"or write the key to {state.key_path()}")
    from .launcher import parse_name, slug, ws_url
    from .runpod_api import ApiError, RunpodApi

    try:
        endpoints = (api_factory or RunpodApi)(key).list_endpoints()
    except ApiError as e:
        if e.status not in (401, 403):
            raise
        raise EndpointNotFound(f"this Runpod key may not list the account's endpoints (HTTP {e.status}; a key that "
                               f"may only call the endpoint cannot): {PASS_URL}") from None
    named = [(e, parse_name(e.get("name"))) for e in endpoints]
    found = [(e, p) for e, p in named if p and (not model or p[1] == slug(model))]
    found = [(e, p) for e, p in found if owner and p[0] == owner] or found  # the caller's own first
    if not found:
        raise EndpointMissing(f"no Playground Serve endpoint for {model or 'any model'} on this Runpod account: run "
                              f"`pgs up {model or '<model>'}` first")
    mine = [e for e, _ in found]
    models = sorted({p[1] for _, p in found})
    if model is None and len(models) > 1:
        raise EndpointNotFound(f"several endpoints are up ({', '.join(models)}); name one, for example "
                               f'connect("{models[0]}")')
    return ws_url(max(mine, key=lambda e: e.get("createdAt") or ""))


def connect(target: str | None = None, *, api_key: str | None = None, key_file: str | os.PathLike | None = None,
            **options):
    """Open a policy client for a model name or ws/wss URL. `options` go to `playground_serve.client.Policy`
    (mode, wait, keepalive, ...). The returned policy has `act(obs)` (the action array) and `infer(obs)` (openpi's
    whole reply)."""
    from .client import connect as open_client

    key = resolve_key(api_key, key_file)
    return open_client(resolve_url(target, key), api_key=key, **options)


@dataclasses.dataclass(frozen=True)
class Status:
    """state: "warm" (a worker is running, so a connect opens at once), "cold" (none is running: a connect starts one)
    or "missing" (no such endpoint). wait_s: (low, high) seconds from a connect to its first action, measured
    (WARM_WAIT_S, COLD_WAIT_S); None when missing. workers: Runpod's raw counts, for example {"idle": 1,
    "initializing": 0, "ready": 1, "running": 0, "throttled": 0, "unhealthy": 0}."""

    state: str
    wait_s: tuple | None
    workers: dict
    endpoint_id: str | None = None
    url: str | None = None


def status(target: str | None = None, *, api_key: str | None = None,
           key_file: str | os.PathLike | None = None) -> Status:
    """Whether the endpoint for `target` (as for `connect`) has a running worker, from one GET of Runpod's worker
    counts. It starts no worker and works with a key that may only call the endpoint (a model name that is neither in
    the state file nor in PLAYGROUND_SERVE_URL is looked up on the account, which needs more). "running" counts a
    worker from its container start, so a worker that a connect woke within the last two minutes or so is "warm" while
    it may still be starting."""
    key = resolve_key(api_key, key_file)
    if not key:
        raise ValueError("no Runpod API key: pass api_key= or key_file=, set RUNPOD_API_KEY, or write it to "
                         f"{state.key_path()}")
    try:
        url = resolve_url(target, key)
    except EndpointMissing:
        return Status("missing", None, {})
    m = re.match(r"wss?://([A-Za-z0-9]+)\.api\.runpod\.ai(?:[:/]|$)", url)
    if not m:
        raise ValueError(f"{url} is not a Runpod endpoint URL (wss://<endpoint-id>.api.runpod.ai/ws)")
    endpoint_id = m.group(1)
    code, body = _get(HEALTH_URL.format(endpoint_id), key)
    if code == 404:
        return Status("missing", None, {}, endpoint_id, url)
    if code in (401, 403):
        raise PermissionError(f"Runpod refused the worker counts of endpoint {endpoint_id} (HTTP {code}): the key "
                              "may not call this endpoint, or the endpoint was deleted")
    try:
        workers = json.loads(body)["workers"] if code == 200 else None
    except (ValueError, KeyError, TypeError):
        workers = None
    if not isinstance(workers, dict):
        raise ConnectionError(f"unexpected answer from {HEALTH_URL.format(endpoint_id)}: HTTP {code} "
                              f"{body[:200].decode(errors='replace')}")
    if (workers.get("running") or 0) > 0:
        return Status("warm", WARM_WAIT_S, workers, endpoint_id, url)
    return Status("cold", COLD_WAIT_S, workers, endpoint_id, url)


def _get(url: str, key: str, timeout: float = 10.0) -> tuple:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                                   "User-Agent": f"playground-serve/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
