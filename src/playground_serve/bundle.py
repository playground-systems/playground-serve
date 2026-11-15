"""RS_FILES and the endpoint's env map.

The worker scripts ship inside this package (playground_serve/worker). RS_FILES packs them as one tar, gzip, base64
string, built deterministically: fixed file order, mode 0644, owner 0:0, mtime 0, ustar headers only (no extended
attributes, no macOS metadata), gzip -9 with a fixed header (no file name, mtime 0, OS byte 255). START_COMMAND
unpacks it to /rs and runs boot.sh.

Runpod replaces an endpoint's whole env map on every update (a partial map drops the other keys), so endpoint_env
returns the complete map.
"""

from __future__ import annotations

import base64
import gzip
import io
import struct
import tarfile
import zlib
from importlib import resources

from playground_serve.models import ModelSpec

WORKER_FILES = ("boot.sh", "build_env.sh", "health.py", "serve.py", "server_decode.py", "stage.py", "warmup.py",
                "fastpath.py", "fast_policy.py", "selftest.py", "selftest_pi05_libero.json", "rtvla_opt/pi05_infer.py",
                "direct.py")
VOLUME = "/runpod-volume"
PORT = 8000  # openpi websocket server; the load balancer routes client traffic here
PORT_HEALTH = 8001  # health.py: GET /ping is 204 while starting, 200 when ready
START_COMMAND = ["bash", "-c", 'mkdir -p /rs && printf %s "$RS_FILES" | base64 -d | tar -xzf - -C /rs && exec bash /rs/boot.sh']
# Linux caps each environment string "NAME=value" at MAX_ARG_STRLEN = 131072 bytes including the trailing NUL
# (execve(2)); a longer one stops the container from starting. Runpod's docs state no lower limit.
MAX_ENV_STRING_BYTES = 131072 - 1


def worker_files() -> dict[str, bytes]:
    """The packaged worker scripts, by file name (all unpacked side by side into /rs)."""
    root = resources.files("playground_serve")
    return {name: root.joinpath("worker", *name.split("/")).read_bytes() for name in WORKER_FILES}


def pack(files: dict[str, bytes]) -> str:
    """tar, gzip -9, base64 of `files` (name -> bytes); the same bytes for the same files."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(files[name]), 0o644, 0
            tar.addfile(info, io.BytesIO(files[name]))
    return base64.b64encode(_gzip(raw.getvalue())).decode("ascii")


def _gzip(data: bytes) -> bytes:
    """gzip -9n with a fixed header. gzip.compress(mtime=0) writes the OS byte of the platform's zlib (Python 3.11
    and 3.12), so the same files gave different bytes on macOS and Linux; here mtime is 0 and the OS byte 255."""
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS, 8, zlib.Z_DEFAULT_STRATEGY)
    body = deflate.compress(data) + deflate.flush()
    return (b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x02\xff" + body
            + struct.pack("<II", zlib.crc32(data), len(data) & 0xFFFFFFFF))


def unpack(rs_files: str) -> dict[str, bytes]:
    """Inverse of pack."""
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(rs_files)))) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def endpoint_env(spec: ModelSpec, release: str) -> dict[str, str]:
    """The complete env map of an endpoint serving `spec`. `release` is a free label; changing it starts a new
    Runpod release (new containers) even when nothing else changed."""
    env = {
        "RS_FILES": pack(worker_files()),
        "RS_RELEASE": release,
        "RS_MODEL": spec.name,
        "RS_OPENPI_CONFIG": spec.openpi_config,
        "RS_CHECKPOINT": spec.checkpoint,
        "RS_FAST": "auto",  # the fast model path where it was measured (worker/fastpath.py); "off" = stock openpi
        "PORT": str(PORT),
        "PORT_HEALTH": str(PORT_HEALTH),
        # On the network volume: the environment tar is shared by all models. Checkpoints land per model under
        # OPENPI_DATA_HOME (openpi caches gs://bucket/path at OPENPI_DATA_HOME/bucket/path). The worker sets
        # JAX_COMPILATION_CACHE_DIR to RS_JAX_CACHE_ROOT/<RS_MODEL>/<GPU name>.
        "RS_ENV_TAR": f"{VOLUME}/rs/env.tar",
        "OPENPI_DATA_HOME": f"{VOLUME}/openpi_data",
        "RS_JAX_CACHE_ROOT": f"{VOLUME}/jax_cache",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }
    return _checked(env)


def _checked(env: dict[str, str]) -> dict[str, str]:
    for key, value in env.items():
        size = len(f"{key}={value}".encode())
        if size > MAX_ENV_STRING_BYTES:
            raise ValueError(f"env {key} is {size} bytes, over the {MAX_ENV_STRING_BYTES} bytes Linux allows")
    return env
