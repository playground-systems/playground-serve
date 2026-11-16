"""Download the Wan2.2 VAE, the umT5 text encoder and its tokenizer from ModelScope before the model loads them
(standard library; /rs/prefetch.py on the worker, next to serve.py).

The model's own loader (flexpi/models/helpers/io.py, ModelConfig.download_if_necessary) fetches these files with
modelscope.snapshot_download, one stream per file, into DIFFSYNTH_MODEL_BASE_PATH/<model_id>/<path>, and skips any file
already there. From Runpod's CA-MTL-1 that stream ran at 1.3 to 1.6 MB/s (about two hours for the 11.4 GB encoder), and
ModelScope serves byte ranges (measured: 16 parallel ranges gave 33 MB/s from a home connection). So this module
writes the same files to the same paths first, with the openpi worker's parallel range downloader (stage.py), checks
each file's size and SHA-256 against the listing below (read from ModelScope's API on 2026-11-16, which pins the
content: the loader itself takes the latest revision), and only then renames it into place. The loader then finds
every file and downloads nothing.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import socket
import time
import urllib.parse

MS = "https://www.modelscope.cn/models/{repo}/resolve/{revision}/{path}"
# (repo, revision, path, size, sha256)
MANIFEST = [
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "master", "Wan2.2_VAE.safetensors", 1409401152,
     "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"),
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "master", "models_t5_umt5-xxl-enc-bf16.safetensors",
     11361845432, "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "master", "google/umt5-xxl/special_tokens_map.json", 6623,
     "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "master", "google/umt5-xxl/spiece.model", 4548313,
     "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "master", "google/umt5-xxl/tokenizer.json", 16837417,
     "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "master", "google/umt5-xxl/tokenizer_config.json", 61728,
     "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"),
]
CHUNK = 32 << 20
THREADS = 32


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def fetch_one(base: pathlib.Path, repo: str, revision: str, path: str, size: int, digest: str, log) -> str:
    """'present', or 'fetched' after a checked download; raises on a size or hash mismatch."""
    import stage  # /rs/stage.py: _get_range retries each range

    final = base / repo / path
    if final.exists() and final.stat().st_size == size:
        return "present"
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.part-{socket.gethostname()}-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.truncate(size)
    url = MS.format(repo=repo, revision=revision, path=urllib.parse.quote(path))
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(THREADS) as ex:
        jobs = [ex.submit(stage._get_range, url, tmp, off, min(CHUNK, size - off)) for off in range(0, size, CHUNK)]
        for job in jobs:
            job.result()
    s = time.time() - t
    got = sha256(tmp)
    if tmp.stat().st_size != size or got != digest:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{repo}/{path}: size or SHA-256 differs from the pinned listing ({got})")
    os.rename(tmp, final)
    log("prefetch_file", file=f"{repo}/{path}", bytes=size, download_s=round(s, 1), mb_s=round(size / max(s, 1e-3) / 1e6),
        check_s=round(time.time() - t - s, 1))
    return "fetched"


def run(base: str | None = None, log=lambda *a, **k: None) -> dict:
    base_path = pathlib.Path(base or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints"))
    t = time.time()
    status = {f"{repo}/{path}": fetch_one(base_path, repo, rev, path, size, digest, log)
              for repo, rev, path, size, digest in MANIFEST}
    log("prefetch_done", s=round(time.time() - t, 1), status=status)
    return status


if __name__ == "__main__":
    print(json.dumps(run()))
