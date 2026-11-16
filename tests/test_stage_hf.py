"""worker/stage.py fetch of hf://REPO@REVISION: the revision's file list with sizes from the Hugging Face API, small
files from huggingface.co, large ones from the address their first request redirects to (Hugging Face's CDN), in
parallel ranges, marked complete last."""

import http.server
import json
import random
import threading
import urllib.parse

import pytest

from playground_serve.worker import stage

SHA = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
FILES = {"config.json": b'{"a": 1}', "model-00001-of-00002.safetensors": random.Random(2).randbytes(9000),
         "assets/sample rgb.png": b"png" * 10}


class Hub(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        self.seen.append((url.path, self.headers.get("Range")))
        if url.path == "/api/models/allenai/MolmoAct2-LIBERO/revision/v1" and url.query == "blobs=true":
            body = {"sha": SHA, "siblings": [{"rfilename": n, "size": len(b)} for n, b in FILES.items()]}
            return self._send(200, json.dumps(body).encode())
        prefix = f"/allenai/MolmoAct2-LIBERO/resolve/{SHA}/"
        if url.path.startswith(prefix):
            name = urllib.parse.unquote(url.path[len(prefix):])
            if len(FILES[name]) > stage.CHUNK:  # large files live on the CDN
                self.send_response(302)
                self.send_header("Location", f"/cdn/{urllib.parse.quote(name)}?signed=1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._range(FILES[name])
        if url.path.startswith("/cdn/") and url.query == "signed=1":
            return self._range(FILES[urllib.parse.unquote(url.path[5:])])
        self._send(404, b"")

    def _range(self, data):
        lo, hi = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        self._send(206, data[lo:hi + 1])

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


ORIGINAL = dict(FILES)


@pytest.fixture
def hub(monkeypatch):
    FILES.clear()
    FILES.update(ORIGINAL)
    Hub.seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(stage, "HF", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(stage, "CHUNK", 4096)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    yield Hub
    server.shutdown()


def test_fetch_hf_repo_at_a_revision(hub, tmp_path):
    files = stage.fetch("hf://allenai/MolmoAct2-LIBERO@v1", tmp_path / "d")
    assert files == {n: len(b) for n, b in FILES.items()} and stage.complete(tmp_path / "d")
    assert {n: (tmp_path / "d" / n).read_bytes() for n in FILES} == FILES
    big = [r for p, r in hub.seen if p.startswith("/cdn/")]
    assert sorted(big) == sorted(["bytes=0-0"] + [f"bytes={o}-{min(o + 4095, 8999)}" for o in range(0, 9000, 4096)])
    # the large file's resolve URL is asked once (to find the CDN address), not once per range
    assert sum(p.endswith(".safetensors") and p.startswith("/allenai") for p, _ in hub.seen) == 1


def test_subfolder_filter_token_and_a_complete_copy_kept(hub, tmp_path, monkeypatch):
    names = {"suite/config.json": b'{"b": 2}', "suite/model-1.safetensors": random.Random(3).randbytes(5000),
             "suite/optimizer.pt": b"o" * 100, "other/config.json": b"x"}
    FILES.clear()
    FILES.update(names)
    monkeypatch.setenv("RS_CKPT_FILES", "config.json, model-*.safetensors")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token-value")
    seen_auth = []
    real = Hub.do_GET

    def do_get(self):
        seen_auth.append((urllib.parse.urlparse(self.path).path, self.headers.get("Authorization"),
                          self.headers.get("Range")))
        return real(self)

    monkeypatch.setattr(Hub, "do_GET", do_get)
    dest = tmp_path / "d"
    files = stage.fetch("hf://allenai/MolmoAct2-LIBERO@v1/suite", dest)
    assert files == {"config.json": 8, "model-1.safetensors": 5000} and stage.complete(dest)
    assert (dest / "model-1.safetensors").read_bytes() == names["suite/model-1.safetensors"]
    assert all(a == "Bearer hf-test-token-value" for p, a, _ in seen_auth if p.startswith(("/api/", "/allenai/")))
    # the redirect to the CDN drops it (here the CDN shares the test server's host; on Hugging Face it is another)
    assert [a for p, a, r in seen_auth if p.startswith("/cdn/") and r == "bytes=0-0"] == [None]
    # a restarted container: the same complete copy is kept, nothing is downloaded or unmarked
    seen_auth.clear()
    assert stage.fetch("hf://allenai/MolmoAct2-LIBERO@v1/suite", dest) == files and stage.complete(dest)
    assert [p for p, _, _ in seen_auth] == ["/api/models/allenai/MolmoAct2-LIBERO/revision/v1"]


def test_a_uri_that_cannot_be_listed_leaves_a_complete_copy(tmp_path):
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "w.bin").write_bytes(b"w")
    stage.mark(dest, stage.sizes(dest))
    assert stage.main(["fetch", "ftp://nowhere/x", str(dest)]) == 1
    assert stage.complete(dest)
