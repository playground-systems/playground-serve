"""worker/stage.py: the parallel GCS download, complete-copy markers, and the volume fill after the server is ready,
including fills cut short (the worker scaled down mid-copy), which must leave nothing a later boot trusts."""

import http.server
import json
import os
import pathlib
import random
import shutil
import subprocess
import threading
import urllib.parse

import pytest

from playground_serve.worker import stage

FILES = {"params/ocdbt.process_0/d/big": random.Random(1).randbytes(10_000), "params/manifest.ocdbt": b"m" * 70,
         "params/empty": b"", "assets/libero/norm_stats.json": b'{"norm_stats": {}}'}


class Bucket(http.server.BaseHTTPRequestHandler):
    """Public GCS as stage.fetch uses it: JSON listing (two pages, a folder placeholder) and ranged media GETs."""
    objects: dict[str, bytes] = {}
    fail: set[str] = set()
    ranges: list[str] = []

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/storage/v1/b/bkt/o":
            names = ["ckpt/"] + sorted(n for n in self.objects if n.startswith(q["prefix"][0]))
            page = int(q.get("pageToken", ["0"])[0])
            body = {"items": [{"name": n, "size": str(len(self.objects.get(n, b"")))} for n in names[page:page + 3]]}
            if page + 3 < len(names):
                body["nextPageToken"] = str(page + 3)
            return self._send(200, json.dumps(body).encode())
        name = urllib.parse.unquote(url.path.removeprefix("/bkt/"))
        if name in self.fail or name not in self.objects:
            return self._send(500 if name in self.fail else 404, b"")
        lo, hi = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append(self.headers["Range"])
        self._send(206, self.objects[name][lo:hi + 1])

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def bucket(monkeypatch):
    Bucket.objects = {f"ckpt/pi05/{k}": v for k, v in FILES.items()}
    Bucket.fail, Bucket.ranges = set(), []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Bucket)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(stage, "GCS", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(stage, "CHUNK", 4096)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    yield Bucket
    server.shutdown()


def read_all(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and not p.name.startswith(".rs_")}


def test_fetch_downloads_every_file_in_ranges_and_marks_it_last(bucket, tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "THREADS", 1)  # one range at a time: the order is the plan's
    files = stage.fetch("gs://bkt/ckpt/pi05", tmp_path / "d")
    assert read_all(tmp_path / "d") == FILES and files == {k: len(v) for k, v in FILES.items()}
    assert stage.complete(tmp_path / "d") and (tmp_path / "d" / stage.SMALL).exists()
    assert len(bucket.ranges) == 3 + 1 + 1  # the 10,000-byte file in three 4 KiB ranges; the empty file needs none
    # small files first (norm stats, manifest), then the big ones by offset
    assert bucket.ranges == ["bytes=0-17", "bytes=0-69", "bytes=0-4095", "bytes=4096-8191", "bytes=8192-9999"]


def test_fetch_yields_to_the_environment_build_until_it_is_done(bucket, tmp_path, monkeypatch):
    env_ok = tmp_path / "env_ok"
    monkeypatch.setattr(stage, "SLOW_THREADS", 1)
    monkeypatch.setattr(stage, "THREADS", 4)
    seen, peak, lock = [0], [0], __import__("threading").Lock()
    real = stage._get_range

    def counting(*a):
        with lock:
            seen[0] += 1
            peak[0] = max(peak[0], seen[0])
        try:
            if a[1].name == "big" and a[2] == 4096:  # half way: the environment is built
                env_ok.touch()
                __import__("time").sleep(0.8)
            return real(*a)
        finally:
            with lock:
                seen[0] -= 1

    monkeypatch.setattr(stage, "_get_range", counting)
    stage.fetch("gs://bkt/ckpt/pi05", tmp_path / "d", wait_for=str(env_ok))
    assert stage.complete(tmp_path / "d") and peak[0] == 1  # one range at a time while the build ran


def test_fetch_that_fails_leaves_no_marker_and_says_so(bucket, tmp_path, capsys):
    bucket.fail = {"ckpt/pi05/params/manifest.ocdbt"}
    assert stage.main(["fetch", "gs://bkt/ckpt/pi05", str(tmp_path / "d")]) == 1
    assert not stage.complete(tmp_path / "d") and (tmp_path / "d" / ".rs_failed").exists()
    assert '"rs": "ckpt_fetch_failed"' in capsys.readouterr().out
    bucket.fail = set()  # a later boot starts over and clears the failure flag
    assert stage.main(["fetch", "gs://bkt/ckpt/pi05", str(tmp_path / "d")]) == 0
    assert stage.complete(tmp_path / "d") and not (tmp_path / "d" / ".rs_failed").exists()


def test_complete_needs_the_marker_and_every_size(tmp_path):
    d = tmp_path / "c"
    for name, data in FILES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    assert not stage.complete(d)
    stage.mark(d, stage.sizes(d))
    assert stage.complete(d) and stage.main(["check", str(d)]) == 0
    (d / "params/ocdbt.process_0/d/big").write_bytes(b"short")
    assert not stage.complete(d) and stage.main(["check", str(d)]) == 1
    assert not stage.complete(tmp_path / "missing")


def make_env(tmp_path):
    env_dir = tmp_path / "opt" / "rs"
    (env_dir / "openpi" / ".venv" / "bin").mkdir(parents=True)
    (env_dir / "openpi" / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (env_dir / "openpi" / ".git").mkdir()
    (env_dir / "openpi" / ".git" / "HEAD").write_text("ref")
    (env_dir / "openpi" / "pyproject.toml").write_text("[project]\n")
    return env_dir


def test_write_tar_is_atomic_and_its_size_marker_is_checked(tmp_path):
    env_dir = make_env(tmp_path)
    tar = tmp_path / "vol" / "rs" / "env.tar"
    (tmp_path / "vol" / "rs").mkdir(parents=True)
    (tmp_path / "vol" / "rs" / "env.tar.part-old-1").write_bytes(b"x" * 100)  # left by an interrupted fill
    size = stage.write_tar(tar, str(env_dir))
    assert stage.tar_ok(tar) and size == tar.stat().st_size and stage.main(["check-tar", str(tar)]) == 0
    assert sorted(p.name for p in tar.parent.iterdir()) == ["env.tar", "env.tar.size"]
    out = tmp_path / "unpacked"
    out.mkdir()
    subprocess.run(["tar", "-C", str(out), "-xf", str(tar)], check=True)
    root = out / str(env_dir).strip("/")
    assert (root / "openpi" / "pyproject.toml").exists() and not (root / "openpi" / ".git").exists()
    with open(tar, "r+b") as f:  # cut short after the rename: a later boot must not trust it
        f.truncate(size - 512)
    assert not stage.tar_ok(tar) and stage.main(["check-tar", str(tar)]) == 1
    assert not stage.tar_ok(tmp_path / "vol" / "nothing.tar")


def test_fill_waits_for_ready_then_saves_env_and_checkpoint(bucket, tmp_path, monkeypatch, capsys):
    env_dir = make_env(tmp_path)
    local, vol = tmp_path / "local" / "ckpt", tmp_path / "vol" / "openpi_data" / "ckpt"
    stage.fetch("gs://bkt/ckpt/pi05", local)
    vol.parent.mkdir(parents=True)
    shutil.copytree(local, str(vol) + ".part-old-1")  # an earlier fill cut short
    (vol.parent / "ckpt.part-old-1" / stage.MARKER).unlink()
    ready, ok = tmp_path / "rs_ready", tmp_path / "rs_env_ok"
    ok.touch()
    monkeypatch.setattr(stage, "READY", str(ready))
    monkeypatch.setattr(stage, "ENV_OK", str(ok))
    monkeypatch.setattr(stage, "ENV_DIR", str(env_dir))
    monkeypatch.setattr(stage.os, "nice", lambda n: 0)
    monkeypatch.setattr(stage.time, "sleep", lambda s: ready.touch())  # the server turns ready while fill waits
    tar = tmp_path / "vol" / "rs" / "env.tar"
    monkeypatch.setenv("RS_ENV_TAR", str(tar))
    monkeypatch.setenv("RS_CKPT_DIR", str(local))
    monkeypatch.setenv("RS_CKPT_VOLUME", str(vol))
    assert stage.main(["fill"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [x["rs"] for x in lines][0] == "fill_start" and lines[-1]["rs"] == "fill_done" and lines[-1]["failed"] == []
    assert stage.tar_ok(tar) and stage.complete(vol) and read_all(vol) == FILES
    assert sorted(p.name for p in vol.parent.iterdir()) == ["ckpt"]  # the leftover temporary copy is gone
    assert stage.main(["fill"]) == 0 and capsys.readouterr().out == ""  # nothing left to do


def test_fill_cut_short_leaves_nothing_a_later_boot_trusts(bucket, tmp_path, monkeypatch):
    local, vol = tmp_path / "local" / "ckpt", tmp_path / "vol" / "ckpt"
    stage.fetch("gs://bkt/ckpt/pi05", local)
    vol.parent.mkdir(parents=True)
    calls = []

    def copy_then_die(src, dst):  # the container is stopped after the first file
        if calls:
            raise KeyboardInterrupt
        calls.append(dst)
        return shutil.copyfile(src, dst)

    monkeypatch.setattr(stage.shutil, "copyfile", copy_then_die)
    with pytest.raises(KeyboardInterrupt):
        stage.copy_dir(local, vol)
    assert not vol.exists() and not stage.complete(vol)
    assert [p.name.startswith("ckpt.part-") for p in vol.parent.iterdir()] == [True]
    monkeypatch.undo()
    stage.copy_dir(local, vol)
    assert stage.complete(vol) and sorted(p.name for p in vol.parent.iterdir()) == ["ckpt"]


def test_fill_decides_after_ready_and_skips_what_the_volume_has(bucket, tmp_path, monkeypatch, capsys):
    """boot.sh starts fill while the checkpoint is still downloading: what to save is decided once the server is
    ready. A later boot that used the volume's copies has nothing to save."""
    local, vol = tmp_path / "local" / "ckpt", tmp_path / "vol" / "ckpt"
    ready = tmp_path / "rs_ready"
    monkeypatch.setattr(stage, "READY", str(ready))
    monkeypatch.setattr(stage, "ENV_OK", str(tmp_path / "no_env"))  # the environment came from the volume's tar
    monkeypatch.setattr(stage.os, "nice", lambda n: 0)
    monkeypatch.setenv("RS_ENV_TAR", str(tmp_path / "vol" / "env.tar"))
    monkeypatch.setenv("RS_CKPT_DIR", str(local))
    monkeypatch.setenv("RS_CKPT_VOLUME", str(vol))

    def download_then_ready(s):  # the download finishes, serve.py loads it and turns ready
        stage.fetch("gs://bkt/ckpt/pi05", local)
        ready.touch()

    monkeypatch.setattr(stage.time, "sleep", download_then_ready)
    assert stage.fill() == 0
    assert stage.complete(vol) and '"what": ["checkpoint"]' in capsys.readouterr().out
    monkeypatch.setenv("RS_CKPT_DIR", str(vol))  # the next boot reads the volume's copy
    assert stage.fill() == 0 and capsys.readouterr().out == ""


def test_fill_waits_for_the_fast_copy_that_serve_py_writes_after_ready(tmp_path, monkeypatch, capsys):
    """A first start's fast path is ready before its weights are on the container disk: serve.py flags the directory
    .rs_pending, writes model.pkl, marks it complete, then drops the flag. fill copies it once complete; a flag that
    goes away without a marker (the save failed) copies nothing."""
    fast, vol = tmp_path / "fast", tmp_path / "vol" / "fast"
    (fast / "assets").mkdir(parents=True)
    (fast / "assets" / "norm_stats.json").write_text("{}")
    (fast / stage.PENDING).touch()
    ready = tmp_path / "rs_ready"
    ready.touch()
    monkeypatch.setattr(stage, "READY", str(ready))
    monkeypatch.setattr(stage, "ENV_OK", str(tmp_path / "no_env"))
    monkeypatch.setattr(stage.os, "nice", lambda n: 0)
    monkeypatch.setenv("RS_ENV_TAR", str(tmp_path / "vol" / "env.tar"))
    monkeypatch.setenv("RS_FAST_DIR", str(fast))
    monkeypatch.setenv("RS_FAST_VOLUME", str(vol))

    def saved(s):  # serve.py finishes writing while fill waits
        (fast / "model.pkl").write_bytes(b"weights")
        stage.mark(fast, stage.sizes(fast))
        (fast / stage.PENDING).unlink()

    monkeypatch.setattr(stage.time, "sleep", saved)
    assert stage.fill() == 0
    out = capsys.readouterr().out
    assert '"what": ["fast"]' in out and '"failed": []' in out
    assert stage.complete(vol) and (vol / "model.pkl").read_bytes() == b"weights" and not (vol / stage.PENDING).exists()
    vol2 = tmp_path / "vol2"
    monkeypatch.setenv("RS_FAST_VOLUME", str(vol2))
    (fast / stage.MARKER).unlink()
    (fast / stage.PENDING).touch()
    monkeypatch.setattr(stage.time, "sleep", lambda s: (fast / stage.PENDING).unlink())  # the save failed
    assert stage.fill() == 1 and not vol2.exists() and '"failed": ["fast"]' in capsys.readouterr().out


def test_stage_is_standard_library_only():
    """It runs on the base image's python3 before the openpi environment exists."""
    source = pathlib.Path(stage.__file__).read_text()
    imported = {line.split()[1].split(".")[0] for line in source.splitlines() if line.startswith(("import ", "from "))}
    assert imported <= {"__future__", "concurrent", "json", "os", "pathlib", "shutil", "socket", "subprocess", "sys",
                        "threading", "time", "urllib"}
    env = {"PATH": os.environ["PATH"]}
    r = subprocess.run(["python3", "-I", stage.__file__], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "stage.py fetch" in r.stderr
