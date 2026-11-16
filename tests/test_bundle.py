"""RS_FILES determinism and round trip, the env map, and pi05_libero's map against the first release that served."""

import base64
import io
import json
import lzma
import os
import pathlib
import py_compile
import random
import subprocess
import sys
import tarfile

import pytest

from playground_serve import bundle, models

WORKER = pathlib.Path(bundle.__file__).parent / "worker"

# The endpoint env of the first release that served pi05_libero on a GPU (before the launcher existed), kept as a
# regression check of the env map.
TODAY = {
    "RS_RELEASE": "h-final",
    "PORT": "8000",
    "PORT_HEALTH": "8001",
    "OPENPI_DATA_HOME": "/runpod-volume/openpi_data",
    "JAX_COMPILATION_CACHE_DIR": "/runpod-volume/jax_cache/v1",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
}
# Intended renames: the fixed JAX cache dir becomes a root (the worker adds <model>/<GPU name>); the env tar path that
# boot.sh hard-coded and the model that serve.py hard-coded (serve_policy --env LIBERO) become variables.
REMOVED = {"JAX_COMPILATION_CACHE_DIR"}
ADDED = {
    "RS_JAX_CACHE_ROOT": "/runpod-volume/jax_cache",
    "RS_ENV_TAR": "/runpod-volume/rs/env.tar",
    "RS_MODEL": "pi05_libero",
    "RS_OPENPI_CONFIG": "pi05_libero",
    "RS_CHECKPOINT": "gs://openpi-assets/checkpoints/pi05_libero",
    "RS_FAST": "auto",  # the fast model path where it was measured (worker/fastpath.py)
}


def test_worker_files_are_the_packaged_files():
    files = bundle.worker_files()
    assert tuple(files) == bundle.WORKER_FILES
    on_disk = {p.relative_to(WORKER).as_posix() for p in WORKER.rglob("*")
               if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts}
    assert on_disk == set(bundle.WORKER_FILES) | {"__init__.py"}


def test_pack_is_deterministic():
    files = bundle.worker_files()
    first = bundle.pack(files)
    assert bundle.pack(dict(reversed(list(files.items())))) == first
    assert bundle.endpoint_env(models.get("pi05_droid"), "r1") == bundle.endpoint_env(models.get("pi05_droid"), "r1")
    xz = base64.b64decode(first)
    assert xz[:6] == b"\xfd7zXZ\x00"  # an xz stream: no file name, no time
    raw = lzma.decompress(xz)
    for marker in (b"PaxHeader", b"LIBARCHIVE", b"com.apple"):
        assert marker not in raw
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == sorted(bundle.WORKER_FILES)
    for m in members:
        assert (m.type, m.mode, m.mtime, m.uid, m.gid, m.uname, m.gname, m.pax_headers) == (
            tarfile.REGTYPE, 0o644, 0, 0, 0, "", "", {})


def test_round_trip_python():
    files = bundle.worker_files()
    assert bundle.unpack(bundle.pack(files)) == files


def test_round_trip_start_command(tmp_path):
    """The start command's unpack step with python3 (only its standard library) and the system's tar gives back the
    same bytes."""
    script = bundle.START_COMMAND[2]
    unpack_step, run_step = script.split(" && exec ")
    assert run_step == "bash /rs/boot.sh"
    path = f"{os.path.dirname(sys.executable)}:/usr/bin:/bin"  # a python3 there, as the image has one on its PATH
    env = {"RS_FILES": bundle.pack(bundle.worker_files()), "PATH": path}
    subprocess.run(["bash", "-c", unpack_step.replace("/rs", str(tmp_path / "rs"))], env=env, check=True)
    out = {p.relative_to(tmp_path / "rs").as_posix(): p.read_bytes() for p in (tmp_path / "rs").rglob("*") if p.is_file()}
    assert out == bundle.worker_files()
    env["RS_FILES"] = "not a bundle"  # a failed unpack stops the start (pipefail): boot.sh does not run
    assert subprocess.run(["bash", "-c", unpack_step.replace("/rs", str(tmp_path / "rs2"))], env=env,
                          capture_output=True).returncode != 0


def test_env_size_limit(monkeypatch):
    env = bundle.endpoint_env(models.get("pi05_libero"), "r1")
    assert len(env["RS_FILES"]) < 80_000  # about 72 kB (the fast path's files), under the 128 KiB per-string limit
    big = {name: random.Random(name).randbytes(20_000) for name in bundle.WORKER_FILES}  # incompressible: 320 kB base64
    monkeypatch.setattr(bundle, "worker_files", lambda: big)
    with pytest.raises(ValueError, match="RS_FILES"):
        bundle.endpoint_env(models.get("pi05_libero"), "r1")


@pytest.mark.parametrize("name", models.names("openpi"))
def test_env_map_per_model(name):
    spec = models.get(name)
    env = bundle.endpoint_env(spec, "r1")
    assert all(isinstance(k, str) and isinstance(v, str) and v for k, v in env.items())
    assert (env["RS_MODEL"], env["RS_OPENPI_CONFIG"], env["RS_CHECKPOINT"]) == (name, spec.openpi_config, spec.checkpoint)
    shared = {k: v for k, v in env.items() if k not in ("RS_MODEL", "RS_OPENPI_CONFIG", "RS_CHECKPOINT")}
    base = bundle.endpoint_env(models.get("pi05_libero"), "r1")
    assert shared == {k: v for k, v in base.items() if k not in ("RS_MODEL", "RS_OPENPI_CONFIG", "RS_CHECKPOINT")}


def test_custom_env_map():
    env = bundle.endpoint_env(models.get("pi05_droid@/runpod-volume/ckpt/1000"), "r2")
    assert env["RS_CHECKPOINT"] == "/runpod-volume/ckpt/1000" and env["RS_MODEL"].startswith("custom-pi05_droid-")


def test_pi05_libero_env_equals_today_except_renames():
    env = bundle.endpoint_env(models.get("pi05_libero"), "h-final")
    assert set(env) == (set(TODAY) | {"RS_FILES"}) - REMOVED | set(ADDED)
    assert {k: env[k] for k in TODAY if k not in REMOVED} == {k: v for k, v in TODAY.items() if k not in REMOVED}
    assert {k: env[k] for k in ADDED} == ADDED
    assert TODAY["JAX_COMPILATION_CACHE_DIR"].startswith(env["RS_JAX_CACHE_ROOT"] + "/")


def test_worker_scripts_parse(tmp_path):
    for name in bundle.WORKER_FILES:
        path = WORKER / name
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(path)], check=True)
        elif name.endswith(".json"):
            json.loads(path.read_text())
        else:
            py_compile.compile(str(path), cfile=str(tmp_path / (name.replace("/", "_") + "c")), doraise=True)
    assert "RS_DIAG" not in (WORKER / "serve.py").read_text()
