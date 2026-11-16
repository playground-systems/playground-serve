"""Worker staging, standard library only: it runs on the base image's python3 before the openpi environment exists.

    stage.py fetch URI DEST [WAIT_FOR]      download a checkpoint to DEST (container disk) in parallel HTTP ranges: a
                                            public GCS directory gs://BUCKET/PATH, a Hugging Face repo
                                            hf://REPO@REVISION[/SUBFOLDER] (every file at that revision, or under
                                            SUBFOLDER with paths relative to it; HF_TOKEN, if set, goes to
                                            huggingface.co only) or one https:// file. RS_CKPT_FILES (comma-separated
                                            fnmatch patterns of relative paths) keeps only matching files. A DEST
                                            that is already a complete copy of the same file list is kept as it is.
                                            With WAIT_FOR, only 4 streams until that file exists
                                            (the environment build gets the bandwidth first). DEST/.rs_small marks the
                                            small files done (norm stats, metadata), DEST/.rs_complete everything
    stage.py check DIR                      exit 0 if DIR is a complete copy (its .rs_complete lists every file's size)
    stage.py check-tar TAR                  exit 0 if TAR's size equals the size recorded in TAR.size
    stage.py fill                           after the server is ready, save what this boot made locally to the volume

The first boot serves from the container disk and only then copies to the network volume: the environment tar
(RS_ENV_TAR, made from RS_ENV_DIR) when the volume has no complete one, the checkpoint when it was downloaded to the container disk
(RS_CKPT_DIR, saved to RS_CKPT_VOLUME), and the fast path's checked copy (RS_FAST_DIR, saved to RS_FAST_VOLUME; on a
first start serve.py writes it after ready, flagged .rs_pending until its marker is there, and fill waits for that).
Every copy is written under a temporary name, checked against the expected
sizes, then renamed; its marker (TAR.size, DIR/.rs_complete) comes last. A fill cut short (the worker scaled down
mid-copy) leaves only temporary names or no marker, which a later boot ignores: it rebuilds or re-downloads.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import threading
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

GCS = "https://storage.googleapis.com"  # public buckets need no credentials
HF = "https://huggingface.co"  # public repos need no token
HEADERS = {"User-Agent": "playground-serve-worker"}  # Cloudflare-fronted hosts refuse urllib's default one
MARKER = ".rs_complete"
SMALL = ".rs_small"
PENDING = ".rs_pending"  # serve.py: the fast path's checked copy is still being written; its marker comes after
PENDING_TIMEOUT_S = 900
FETCHING = ".rs_fetching"  # `stage.py fetch` is writing this directory; holds its process id
CHUNK = 32 << 20
THREADS, SLOW_THREADS = 48, 4
READY = "/tmp/rs_ready"
ENV_DIR = os.environ.get("RS_ENV_DIR", "/opt/rs")  # openpi: checkout + .venv; other families: /opt/pgs (boot.sh)
ENV_OK = "/opt/rs_env_ok"  # boot.sh: ENV_DIR holds a complete environment


def log(phase: str, **kw) -> None:
    now = time.time()
    boot = float(os.environ.get("RS_BOOT_T") or now)
    print(json.dumps({"rs": phase, "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z",
                      "s_since_boot": round(now - boot, 3), **kw}), flush=True)


def tmp_name(path: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(f"{path}.part-{socket.gethostname()}-{os.getpid()}")


# ----- complete copies
def sizes(root: pathlib.Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*"))
            if p.is_file() and not p.name.startswith(".rs_")}  # not this module's own flags


def complete(root: str | pathlib.Path) -> bool:
    """True if root/.rs_complete exists and every file it lists is there with that size."""
    root = pathlib.Path(root)
    try:
        want = json.loads((root / MARKER).read_text())["files"]
        return all((root / name).stat().st_size == size for name, size in want.items())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def fetching(root: str | pathlib.Path) -> bool:
    """True while a `stage.py fetch` process writes root (its .rs_fetching names a live process)."""
    try:
        os.kill(int((pathlib.Path(root) / FETCHING).read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def mark(root: pathlib.Path, files: dict[str, int]) -> None:
    tmp = root / (MARKER + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps({"files": files}))
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(root / MARKER)


def copy_synced(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy one file and flush it to the volume, so that the marker written after it never describes unwritten data."""
    shutil.copyfile(src, dst)
    with open(dst, "rb") as f:
        os.fsync(f.fileno())


def tar_ok(tar: str | pathlib.Path) -> bool:
    try:
        return int(pathlib.Path(f"{tar}.size").read_text()) == os.path.getsize(tar)
    except (OSError, ValueError):
        return False


# ----- download
def _request(url: str, headers: dict | None = None, method: str | None = None) -> urllib.request.Request:
    """A request with HEADERS; to huggingface.co also HF_TOKEN, as a header that a redirect to another host drops."""
    req = urllib.request.Request(url, headers={**HEADERS, **(headers or {})}, method=method)
    if url.startswith(HF + "/") and os.environ.get("HF_TOKEN", "").strip():
        req.add_unredirected_header("Authorization", f"Bearer {os.environ['HF_TOKEN'].strip()}")
    return req


def _json(url: str):
    with urllib.request.urlopen(_request(url), timeout=30) as r:
        return json.load(r)


def gcs_objects(bucket: str, prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", **({"pageToken": token} if token else {})}
        page = _json(f"{GCS}/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q))
        items += page.get("items") or []
        token = page.get("nextPageToken")
        if not token:
            return items


def listing(uri: str) -> list[tuple[str, int, str]]:
    """(path relative to the destination, size, download URL) of every file of a checkpoint URI that matches
    RS_CKPT_FILES (all if unset)."""
    patterns = [p.strip() for p in os.environ.get("RS_CKPT_FILES", "").split(",") if p.strip()]
    return [o for o in _listing(uri) if not patterns or any(fnmatch.fnmatch(o[0], p) for p in patterns)]


def _listing(uri: str) -> list[tuple[str, int, str]]:
    if uri.startswith("gs://"):
        bucket, _, path = uri.removeprefix("gs://").partition("/")
        prefix = path.strip("/") + "/"
        return [(o["name"][len(prefix):], int(o["size"]), f"{GCS}/{bucket}/{urllib.parse.quote(o['name'])}")
                for o in gcs_objects(bucket, prefix) if not o["name"].endswith("/")]
    if uri.startswith("hf://"):  # every file of the repo at that revision (under the subfolder), with its size
        repo, _, rest = uri.removeprefix("hf://").partition("@")
        rev, _, sub = rest.partition("/")
        prefix = sub.strip("/") + "/" if sub.strip("/") else ""
        info = _json(f"{HF}/api/models/{repo}/revision/{rev or 'main'}?blobs=true")
        return [(s["rfilename"][len(prefix):], int(s["size"]),
                 f"{HF}/{repo}/resolve/{info['sha']}/{urllib.parse.quote(s['rfilename'])}")
                for s in info["siblings"] if s["rfilename"].startswith(prefix)]
    if uri.startswith("https://"):  # one file
        with urllib.request.urlopen(_request(uri, method="HEAD"), timeout=30) as r:
            return [(pathlib.PurePosixPath(urllib.parse.urlparse(uri).path).name, int(r.headers["Content-Length"]), uri)]
    raise ValueError(f"cannot download {uri}: not gs://, hf:// or https://")


def final_url(url: str) -> str:
    """Where `url` redirects to (Hugging Face sends large files to its CDN with a signed URL), asked once per file so
    that each range request goes straight there instead of through another redirect."""
    with urllib.request.urlopen(_request(url, {"Range": "bytes=0-0"}), timeout=30) as r:
        return r.geturl()


def _get_range(url: str, target: pathlib.Path, offset: int, n: int) -> int:
    for attempt in range(6):
        try:
            req = _request(url, {"Range": f"bytes={offset}-{offset + n - 1}"})
            got = 0
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "r+b") as f:
                if r.status != 206 and offset:  # the whole object instead of the range: not what was asked
                    raise OSError(f"HTTP {r.status} for a range request")
                f.seek(offset)
                while got < n and (chunk := r.read(min(1 << 20, n - got))):
                    f.write(chunk)
                    got += len(chunk)
            if got == n:
                return n
        except OSError:
            pass
        time.sleep(1 + 2 * attempt)
    raise OSError(f"could not download bytes {offset}-{offset + n - 1} of {url}")


def fetch(uri: str, dest: str | pathlib.Path, wait_for: str | None = None) -> dict[str, int]:
    """Download the checkpoint `uri` (see listing) to `dest` with parallel HTTP range requests; returns name -> size.
    Small files come first (dest/.rs_small once they are all there); dest/.rs_complete is written last. With
    `wait_for`, at most SLOW_THREADS ranges are in flight until that path exists."""
    dest = pathlib.Path(dest)
    objects = listing(uri)  # before any flag is removed: a URI that cannot be listed leaves a complete copy alone
    if not objects:
        raise FileNotFoundError(f"no objects under {uri}")
    if complete(dest) and json.loads((dest / MARKER).read_text())["files"] == {rel: size for rel, size, _ in objects}:
        return {rel: size for rel, size, _ in objects}  # a restarted container: the same files are already here
    for flag in (".rs_failed", SMALL, MARKER):
        (dest / flag).unlink(missing_ok=True)
    files, jobs = {}, []
    for rel, size, url in objects:
        files[rel] = size
        if url.startswith(HF) and size > CHUNK:
            url = final_url(url)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.truncate(size)
        jobs += [(url, target, off, min(CHUNK, size - off)) for off in range(0, size, CHUNK)]
    jobs.sort(key=lambda j: (files[j[1].relative_to(dest).as_posix()] > CHUNK, j[2]))  # small files, then by offset
    gate = threading.Semaphore(SLOW_THREADS if wait_for else THREADS)
    done = threading.Event()

    def speed_up():
        while not done.wait(0.5):
            if os.path.exists(wait_for):
                for _ in range(THREADS - SLOW_THREADS):
                    gate.release()
                return

    def get(job):
        with gate:
            return _get_range(*job)

    if wait_for:
        threading.Thread(target=speed_up, daemon=True).start()
    try:
        with concurrent.futures.ThreadPoolExecutor(THREADS) as ex:  # each range either arrives whole or raises
            futures = [(files[j[1].relative_to(dest).as_posix()] <= CHUNK, ex.submit(get, j)) for j in jobs]
            for small, fut in futures:
                fut.result()
                if not small and not (dest / SMALL).exists():
                    (dest / SMALL).write_text("")
    finally:
        done.set()
    (dest / SMALL).touch()
    mark(dest, files)
    return files


# ----- fill the volume after the server is ready
def drop_leftovers(path: pathlib.Path) -> None:
    """Remove temporary copies of `path` that an earlier, interrupted fill left on the volume (they take space)."""
    for old in path.parent.glob(path.name + ".part-*"):
        shutil.rmtree(old) if old.is_dir() else old.unlink(missing_ok=True)


def copy_dir(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy a complete directory to dst atomically: temporary name, size check, marker, rename."""
    files = json.loads((src / MARKER).read_text())["files"]
    drop_leftovers(dst)
    tmp = tmp_name(dst)
    for name in files:
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(8) as ex:  # parallel streams: one stream to the volume is slow
        list(ex.map(lambda name: copy_synced(src / name, tmp / name), sorted(files, key=lambda n: -files[n])))
    if sizes(tmp) != files:
        raise OSError(f"copy to {tmp} is incomplete")
    mark(tmp, files)
    if dst.exists():
        shutil.rmtree(dst)  # an incomplete or unmarked copy: a complete one is never replaced (checked by the caller)
    os.rename(tmp, dst)


def write_tar(tar: pathlib.Path, env_dir: str) -> int:
    """tar of env_dir (members keep its absolute path, unpacked with tar -C /) to `tar` atomically: temporary name,
    fsync, rename; TAR.size (the byte count) is written last."""
    tar.parent.mkdir(parents=True, exist_ok=True)
    drop_leftovers(tar)
    tmp = tmp_name(tar)
    member = env_dir.strip("/")
    r = subprocess.run(["tar", "-C", "/", f"--exclude={member}/openpi/.git", f"--exclude={member}/repo/.git", "-cf",
                        str(tmp), member],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):  # GNU tar's 1: a file changed while read (a .pyc written); the archive is whole
        tmp.unlink(missing_ok=True)
        raise OSError(f"tar to {tmp} failed (rc {r.returncode}): {r.stderr[-300:]}")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
        size = os.fstat(f.fileno()).st_size
    os.rename(tmp, tar)
    size_tmp = tmp_name(f"{tar}.size")
    size_tmp.write_text(str(size))
    os.rename(size_tmp, f"{tar}.size")
    return size


def wait_for(path: str | None, timeout: float) -> None:
    """Wait until `path` exists (if one is given), for at most `timeout` seconds."""
    end = time.time() + timeout
    while path and not os.path.exists(path) and time.time() < end:
        time.sleep(1)


def copy_pending(src: str, dst: str, timeout: float = PENDING_TIMEOUT_S) -> None:
    """copy_dir once `src` is complete: wait while serve.py still writes it (its PENDING flag), at most `timeout`."""
    end = time.time() + timeout
    while os.path.exists(os.path.join(src, PENDING)) and not complete(src) and time.time() < end:
        time.sleep(1)
    if not complete(src):
        raise OSError(f"{src} was not completed")
    copy_dir(pathlib.Path(src), pathlib.Path(dst))


def fill() -> int:
    """Wait until the server is ready (serve.py has the environment and the whole checkpoint by then), then save to
    the volume whatever it lacks."""
    env_tar = pathlib.Path(os.environ.get("RS_ENV_TAR", "/runpod-volume/rs/env.tar"))
    ckpt, vol_ckpt = os.environ.get("RS_CKPT_DIR"), os.environ.get("RS_CKPT_VOLUME")
    while not os.path.exists(READY):
        time.sleep(1)
    jobs = {}
    if not tar_ok(env_tar) and os.path.exists(ENV_OK):
        jobs["env"] = lambda: wait_for(os.environ.get("RS_PYC_DONE"), 300) or write_tar(env_tar, ENV_DIR)
    if ckpt and vol_ckpt and ckpt != vol_ckpt and complete(ckpt) and not complete(vol_ckpt):
        jobs["checkpoint"] = lambda: copy_dir(pathlib.Path(ckpt), pathlib.Path(vol_ckpt))
    fast, vol_fast = os.environ.get("RS_FAST_DIR"), os.environ.get("RS_FAST_VOLUME")  # marked only once checked
    if fast and vol_fast and not complete(vol_fast) and (complete(fast) or os.path.exists(os.path.join(fast, PENDING))):
        jobs["fast"] = lambda: copy_pending(fast, vol_fast)
    if not jobs:
        return 0
    os.nice(19)
    t0 = time.time()
    log("fill_start", what=sorted(jobs))
    failed = []
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as ex:
        futures = {name: ex.submit(job) for name, job in jobs.items()}
        for name, fut in futures.items():
            try:
                fut.result()
                log(f"fill_{name}_done", s=round(time.time() - t0, 1))
            except Exception as e:  # noqa: BLE001  (the server keeps serving; the next boot tries again)
                failed.append(name)
                log("fill_failed", what=name, error=str(e)[:300])
    log("fill_done", s=round(time.time() - t0, 1), failed=failed)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "check":
        return 0 if complete(argv[1]) else 1
    if cmd == "check-tar":
        return 0 if tar_ok(argv[1]) else 1
    if cmd == "fill":
        return fill()
    if cmd == "fetch":
        t0 = time.time()
        log("ckpt_fetch_start", uri=argv[1], dest=argv[2])
        running = pathlib.Path(argv[2]) / FETCHING
        running.parent.mkdir(parents=True, exist_ok=True)
        running.write_text(str(os.getpid()))
        try:
            files = fetch(argv[1], argv[2], argv[3] if len(argv) > 3 else None)
        except Exception as e:
            log("ckpt_fetch_failed", error=str(e)[:300])
            (pathlib.Path(argv[2]) / ".rs_failed").write_text(str(e)[:300])
            return 1
        finally:
            running.unlink(missing_ok=True)
        n, s = sum(files.values()), time.time() - t0
        log("ckpt_fetch_done", bytes=n, files=len(files), s=round(s, 1), mb_s=round(n / s / 1e6))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
