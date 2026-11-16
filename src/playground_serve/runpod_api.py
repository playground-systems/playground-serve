"""Small client for Runpod's public REST API v2 (https://api.runpod.io/v2) and the load-balancer gateway.

REST v2 is the only Runpod API that has both the serverless stock per pool and data center
(`/catalog/gpus?include=AVAILABILITY&product=SERVERLESS`) and the endpoint worker list and logs. Standard library
only. The key goes only into the Authorization header; it is never logged and never part of an error message.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import __version__

API = "https://api.runpod.io/v2"
# An explicit, generic User-Agent: Cloudflare in front of Runpod blocks urllib's default one ("error code: 1010"
# on rest.runpod.io). It carries the tool name and version only, nothing about the user.
USER_AGENT = f"playground-serve/{__version__}"


class ApiError(Exception):
    def __init__(self, status: int, method: str, path: str, detail: str, errors: list[str] | None = None):
        self.status, self.method, self.path, self.detail, self.errors = status, method, path, detail, errors or []
        text = f"{method} {path}: HTTP {status}: {detail}" if status else f"{method} {path}: {detail}"
        super().__init__(text + ("" if not self.errors else " (" + "; ".join(self.errors) + ")"))


def encode(body) -> bytes:
    """A request body as this client sends it (Runpod's size limit counts these bytes: launcher.MAX_REQUEST_BYTES)."""
    return json.dumps(body).encode()


def _error(status: int, method: str, path: str, raw: bytes) -> ApiError:
    try:
        body = json.loads(raw)
        return ApiError(status, method, path, str(body.get("detail") or body.get("title") or body), body.get("errors"))
    except (ValueError, AttributeError):
        return ApiError(status, method, path, raw.decode(errors="replace").strip()[:300])


class RunpodApi:
    def __init__(self, key: str, base: str = API, timeout: float = 30.0, retries: int = 4, sleep=time.sleep):
        if not key or not key.strip():
            raise ValueError("empty Runpod API key")
        self._key = key.strip()
        self.base, self.timeout, self.retries, self._sleep = base.rstrip("/"), timeout, retries, sleep

    def __repr__(self) -> str:
        return f"RunpodApi(base={self.base!r}, key=<hidden>)"

    def _open(self, method: str, url: str, body=None, timeout: float | None = None, accept="application/json"):
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT, "Accept": accept}
        data = None
        if body is not None:
            data, headers["Content-Type"] = encode(body), "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def request(self, method: str, path: str, body=None, params: dict | None = None):
        """One JSON call. 429 is retried for every method, 5xx and network errors only for GET."""
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        attempt = 0
        while True:
            try:
                with self._open(method, url, body) as r:
                    raw = r.read()
                return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                raw = e.read()
                if attempt < self.retries and (e.code == 429 or (e.code >= 500 and method == "GET")):
                    attempt += 1
                    self._sleep(2**attempt)
                    continue
                raise _error(e.code, method, path, raw) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.retries and method == "GET":
                    attempt += 1
                    self._sleep(2**attempt)
                    continue
                raise ApiError(0, method, path, f"network error: {getattr(e, 'reason', e)}") from None

    def _list(self, path: str, key: str) -> list[dict]:
        out, cursor = [], None
        while True:
            page = self.request("GET", path, params={"cursor": cursor} if cursor else None) or {}
            out += page.get(key) or []
            more = page.get("pagination") or {}
            if not more.get("hasNextPage") or not more.get("nextCursor"):
                return out
            cursor = more["nextCursor"]

    # Serverless endpoints
    def list_endpoints(self) -> list[dict]:
        return self._list("/serverless", "endpoints")

    def get_endpoint(self, endpoint_id: str) -> dict:
        return self.request("GET", f"/serverless/{endpoint_id}")

    def create_endpoint(self, body: dict) -> dict:
        return self.request("POST", "/serverless", body)

    def update_endpoint(self, endpoint_id: str, body: dict) -> dict:
        return self.request("PATCH", f"/serverless/{endpoint_id}", body)

    def delete_endpoint(self, endpoint_id: str) -> None:
        self.request("DELETE", f"/serverless/{endpoint_id}")

    def endpoint_workers(self, endpoint_id: str) -> dict:
        return self.request("GET", f"/serverless/{endpoint_id}/workers")

    # Network volumes
    def list_volumes(self) -> list[dict]:
        return self._list("/network-volumes", "networkVolumes")

    def get_volume(self, volume_id: str) -> dict:
        return self.request("GET", f"/network-volumes/{volume_id}")

    def create_volume(self, body: dict) -> dict:
        return self.request("POST", "/network-volumes", body)

    def delete_volume(self, volume_id: str) -> None:
        self.request("DELETE", f"/network-volumes/{volume_id}")

    # Catalog
    def serverless_gpus(self, min_cuda: str | None = None) -> list[dict]:
        params = {"include": "AVAILABILITY", "product": "SERVERLESS"}
        if min_cuda:
            params["minCudaVersion"] = min_cuda
        return (self.request("GET", "/catalog/gpus", params=params) or {}).get("gpus") or []

    def serverless_cpus(self) -> list[dict]:
        return (self.request("GET", "/catalog/cpus", params={"include": "AVAILABILITY", "product": "SERVERLESS"})
                or {}).get("cpus") or []

    def datacenters(self) -> list[dict]:
        return (self.request("GET", "/catalog/datacenters") or {}).get("dataCenters") or []

    # Worker logs (server-sent events) and the gateway
    def worker_log_lines(self, endpoint_id: str, worker_id: str, since: str | None = None, wait: float = 3.0,
                         tail: int = 300) -> list[str]:
        """Container log lines of one worker: the last `tail`, or those since `since` (RFC 3339); read for at most
        `wait` seconds."""
        params = {"source": "container", **({"since": since} if since else {"tail": tail})}
        url = f"{self.base}/serverless/{endpoint_id}/workers/{worker_id}/logs?" + urllib.parse.urlencode(params)
        lines, deadline = [], time.monotonic() + wait
        try:
            with self._open("GET", url, timeout=wait, accept="text/event-stream") as r:
                while time.monotonic() < deadline:
                    raw = r.readline()
                    if not raw:
                        break
                    line = parse_sse_data(raw.decode(errors="replace"))
                    if line is not None:
                        lines.append(line)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass  # a timeout ends the read; logs are only for progress display
        return lines

    def websocket(self, url: str, timeout: float = 10.0):
        """An open websocket (websockets' sync client) to a load-balancing endpoint, the key in the upgrade header."""
        from websockets.sync.client import connect

        return connect(url, additional_headers={"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT},
                       open_timeout=timeout, compression=None, max_size=None)

    def gateway_status(self, url: str, timeout: float = 10.0) -> int:
        """HTTP status of a GET on the endpoint's gateway (for example its /ping); 0 on a timeout or network error."""
        try:
            with self._open("GET", url, timeout=timeout, accept="*/*") as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return 0


def parse_sse_data(raw: str) -> str | None:
    """The `line` field of one SSE `data:` line ({"source", "line", "ts"}), else None."""
    if not raw.startswith("data:"):
        return None
    try:
        return json.loads(raw[5:].strip()).get("line")
    except (ValueError, AttributeError):
        return None
