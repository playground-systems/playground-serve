"""RunpodApi against a local HTTP server: headers, pagination, errors, retries, SSE logs, gateway status."""

from __future__ import annotations

import http.server
import json
import re
import threading

import pytest

from playground_serve import __version__
from playground_serve.runpod_api import USER_AGENT, ApiError, RunpodApi, parse_sse_data

KEY = "test-key-0123456789"


class Handler(http.server.BaseHTTPRequestHandler):
    seen: list = []
    replies: dict = {}

    def _reply(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Handler.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                             "body": json.loads(body) if body else None})
        queue = Handler.replies.get((self.command, self.path.split("?")[0]))
        status, payload, ctype = queue.pop(0) if queue and len(queue) > 1 else queue[0] if queue else (404, {}, None)
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PATCH = do_DELETE = _reply

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.seen, Handler.replies = [], {}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v2", Handler
    srv.shutdown()


def test_user_agent_is_generic_and_key_only_in_authorization(server):
    base, h = server
    h.replies[("GET", "/v2/catalog/datacenters")] = [(200, {"dataCenters": [{"id": "US-IL-1"}]}, None)]
    assert RunpodApi(KEY, base).datacenters() == [{"id": "US-IL-1"}]
    headers = h.seen[0]["headers"]
    # urllib's default "Python-urllib/x" is blocked by Runpod's Cloudflare (error 1010): send our own, with no user data
    assert headers["User-Agent"] == USER_AGENT == f"playground-serve/{__version__}"
    assert re.fullmatch(r"playground-serve/[0-9.]+", headers["User-Agent"])
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert [k for k, v in headers.items() if KEY in v] == ["Authorization"]
    assert KEY not in repr(RunpodApi(KEY, base))


def test_list_follows_pagination(server):
    base, h = server
    h.replies[("GET", "/v2/serverless")] = [
        (200, {"endpoints": [{"id": "a"}], "pagination": {"hasNextPage": True, "nextCursor": "c1"}}, None),
        (200, {"endpoints": [{"id": "b"}], "pagination": {"hasNextPage": False, "nextCursor": None}}, None)]
    assert [e["id"] for e in RunpodApi(KEY, base).list_endpoints()] == ["a", "b"]
    assert h.seen[1]["path"] == "/v2/serverless?cursor=c1"


def test_errors_carry_status_and_detail_but_not_the_key(server):
    base, h = server
    h.replies[("POST", "/v2/network-volumes")] = [
        (403, {"detail": "this API key does not grant volumes:write", "status": 403, "title": "Forbidden"}, None)]
    with pytest.raises(ApiError) as e:
        RunpodApi(KEY, base).create_volume({"name": "playground-serve-x-abcdef", "size": 40, "dataCenter": "US-IL-1"})
    assert e.value.status == 403 and "volumes:write" in str(e.value) and KEY not in str(e.value)
    assert h.seen[0]["body"]["dataCenter"] == "US-IL-1" and h.seen[0]["headers"]["Content-Type"] == "application/json"


def test_validation_errors_are_listed(server):
    base, h = server
    h.replies[("POST", "/v2/serverless")] = [
        (422, {"detail": "Request validation failed.", "errors": ["$: missing property 'image'"]}, None)]
    with pytest.raises(ApiError, match="missing property 'image'"):
        RunpodApi(KEY, base).create_endpoint({})


def test_429_is_retried_and_post_5xx_is_not(server):
    base, h = server
    sleeps = []
    h.replies[("GET", "/v2/serverless/e1/workers")] = [(429, {"detail": "slow down"}, None), (200, {"workers": []}, None)]
    assert RunpodApi(KEY, base, sleep=sleeps.append).endpoint_workers("e1") == {"workers": []}
    assert sleeps == [2]
    h.replies[("POST", "/v2/serverless")] = [(502, "bad gateway", "text/plain"), (201, {"id": "x"}, None)]
    with pytest.raises(ApiError) as e:
        RunpodApi(KEY, base, sleep=sleeps.append).create_endpoint({"name": "n"})
    assert e.value.status == 502
    assert len([s for s in h.seen if s["method"] == "POST"]) == 1


def test_delete_returns_on_204(server):
    base, h = server
    h.replies[("DELETE", "/v2/serverless/e1")] = [(204, "", None)]
    RunpodApi(KEY, base).delete_endpoint("e1")
    assert h.seen[0]["method"] == "DELETE"


def test_catalog_asks_for_serverless_availability(server):
    base, h = server
    h.replies[("GET", "/v2/catalog/gpus")] = [(200, {"gpus": [{"id": "g", "pool": "ADA_24"}]}, None)]
    RunpodApi(KEY, base).serverless_gpus("12.8")
    assert h.seen[0]["path"] == "/v2/catalog/gpus?include=AVAILABILITY&product=SERVERLESS&minCudaVersion=12.8"


def test_worker_log_lines_parse_server_sent_events(server):
    base, h = server
    events = "".join(f"id: 2026-11-13T00:00:0{i}Z\ndata: {json.dumps({'source': 'container', 'line': line})}\n\n"
                     for i, line in enumerate(['{"rs":"boot"}', "plain text", '{"rs":"ready"}']))
    h.replies[("GET", "/v2/serverless/e1/workers/w1/logs")] = [(200, events, "text/event-stream")]
    lines = RunpodApi(KEY, base).worker_log_lines("e1", "w1", since="2026-11-13T00:00:00Z", wait=2)
    assert lines == ['{"rs":"boot"}', "plain text", '{"rs":"ready"}']
    assert "since=2026-11-13T00%3A00%3A00Z" in h.seen[0]["path"] and "source=container" in h.seen[0]["path"]
    assert h.seen[0]["headers"]["Accept"] == "text/event-stream"
    assert parse_sse_data("id: x") is None and parse_sse_data("data: not json") is None


def test_gateway_status_returns_codes_and_zero_on_network_error(server):
    base, h = server
    h.replies[("GET", "/ping")] = [(204, "", None), (200, "", None)]
    api = RunpodApi(KEY, base)
    root = base.rsplit("/v2", 1)[0]
    assert api.gateway_status(root + "/ping") == 204
    assert api.gateway_status(root + "/ping") == 200
    assert api.gateway_status("http://127.0.0.1:9/ping", timeout=1) == 0
    assert h.seen[0]["headers"]["User-Agent"] == USER_AGENT
