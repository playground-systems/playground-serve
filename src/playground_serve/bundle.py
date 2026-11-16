"""RS_FILES and the endpoint's env map.

The worker scripts ship inside this package: playground_serve/worker for the openpi family; for any other family the
files of families/<family>/ (all but spec.py, NOTES.md, __init__.py and the dev tools in NOT_SHIPPED; files whose
name starts with "verify" only for a bring-up run, RS_VERIFY=1; a self-test reference selftest_<entry>.json only for
that entry) plus families/common/ and the openpi worker's family-neutral health.py, server_decode.py and stage.py. A
family's Python files ship without comments and docstrings, line for line (slim), so a worker's traceback names the
repository's line numbers. RS_FILES packs one entry's files as one tar, xz, base64 string, built deterministically:
fixed file order, mode 0644, owner 0:0, mtime 0, ustar headers only (no extended attributes, no macOS metadata), xz
preset 9e. START_COMMAND unpacks it to /rs with the image's python3 (lzma is in its standard library) and tar, and runs
boot.sh.

Size: Runpod refuses a create or update request it expands past 102400 bytes (HTTP 413; measured 2026-11-16: a
100,300-byte create request accepted, a 100,400-byte one refused; launcher.MAX_REQUEST_BYTES), and RS_FILES is most of
it. Against gzip with every file shipped, xz, the per-entry self-test file and slim took the largest entry
(cosmos3_edge_droid) from 97.6 to 79.6 kB of request; tests/test_families.py keeps every entry and bring-up run 15 %
under the limit, and `pgs up` measures its request before it creates anything. Not chosen: splitting RS_FILES over
several env values (the limit is on the whole request), and fetching the files at boot from the public repository
(every start would need GitHub and the code published at that commit; a local change could not run).

Volume layout of a family endpoint (family_env): its environment tar at /runpod-volume/pgs/<family>/env.tar, the
checkpoint's complete copy under /runpod-volume/pgs/weights/<repo@revision>, HF_HOME at /runpod-volume/hf, and
whatever the family's serve.py keeps under /runpod-volume/pgs/<family>/ (its fast path's caches).

Runpod replaces an endpoint's whole env map on every update (a partial map drops the other keys), so endpoint_env
returns the complete map.
"""

from __future__ import annotations

import ast
import base64
import io
import lzma
import tarfile
import tokenize
from importlib import resources

from playground_serve.models import ModelSpec

WORKER_FILES = ("boot.sh", "build_env.sh", "health.py", "serve.py", "server_decode.py", "stage.py", "warmup.py",
                "fastpath.py", "fast_policy.py", "selftest.py", "selftest_pi05_libero.json", "rtvla_opt/pi05_infer.py",
                "direct.py")
SHARED_FILES = ("health.py", "server_decode.py", "stage.py")  # from worker/, used by every family
COMMON_FILES = ("boot.sh", "build_lib.sh", "pgs_msgpack_numpy.py", "pgs_server.py")  # families/common/
NOT_SHIPPED = ("spec.py", "NOTES.md", "__init__.py", "bringup.py", "devhook.py")  # read by `pgs` or dev tools only
VERIFY_PREFIX = "verify"  # a family's bring-up checks (verify.py, verify_obs.json): shipped only when RS_VERIFY=1
SELFTEST_PREFIX = "selftest_"  # selftest_<entry>.json: an entry's stored stock reference, shipped only for that entry
VOLUME = "/runpod-volume"
PORT = 8000  # openpi websocket server; the load balancer routes client traffic here
PORT_HEALTH = 8001  # health.py: GET /ping is 204 while starting, 200 when ready
UNPACK = "import base64,lzma,sys; sys.stdout.buffer.write(lzma.decompress(base64.b64decode(sys.stdin.buffer.read())))"
START_COMMAND = ["bash", "-c", f'set -o pipefail; mkdir -p /rs && printf %s "$RS_FILES" | python3 -c "{UNPACK}" | '
                 'tar -xf - -C /rs && exec bash /rs/boot.sh']
# Linux caps each environment string "NAME=value" at MAX_ARG_STRLEN = 131072 bytes including the trailing NUL
# (execve(2)); a longer one stops the container from starting. Runpod's docs state no lower limit.
MAX_ENV_STRING_BYTES = 131072 - 1


def worker_files(family: str = "openpi", verify: bool = False, entry: str | None = None) -> dict[str, bytes]:
    """The packaged worker scripts of a family, by path (all unpacked side by side into /rs); with `verify`, also the
    family's bring-up files (VERIFY_PREFIX); with `entry`, only that entry's self-test reference (SELFTEST_PREFIX). A
    family's Python files are slim."""
    root = resources.files("playground_serve")
    if family == "openpi":
        return {name: root.joinpath("worker", *name.split("/")).read_bytes() for name in WORKER_FILES}
    folder = root.joinpath("families", family)
    if family == "common" or not folder.joinpath("serve.py").is_file():
        raise ValueError(f"no worker family {family!r} in playground_serve/families")
    files = {name: root.joinpath("worker", name).read_bytes() for name in SHARED_FILES}
    files.update({name: root.joinpath("families", "common", name).read_bytes() for name in COMMON_FILES})

    def shipped(name: str) -> bool:
        if name in NOT_SHIPPED or (name.startswith(VERIFY_PREFIX) and not verify):
            return False
        return not (entry and name.startswith(SELFTEST_PREFIX) and name.split(".")[0] != SELFTEST_PREFIX + entry)

    def walk(node, prefix=""):
        for item in sorted(node.iterdir(), key=lambda x: x.name):
            if item.name.startswith(".") or item.name == "__pycache__" or item.name.endswith(".pyc"):
                continue
            if item.is_dir():
                walk(item, f"{prefix}{item.name}/")
            elif prefix or shipped(item.name):
                files[prefix + item.name] = item.read_bytes()  # a family's own file may replace a common one

    walk(folder)
    return {name: slim(data) if name.endswith(".py") else data for name, data in files.items()}


def slim(source: bytes) -> bytes:
    """A Python file without its comments and docstrings, line for line: each comment is cut from its line, and each
    docstring becomes an empty string over the same lines, so the code (its AST but for the docstrings) and every line
    number stay the same. A file that reads __doc__ keeps its module docstring; one that does not parse here (newer
    syntax than this Python) ships as it is."""
    try:
        text = source.decode("utf-8")
        reads_doc = any(t.type == tokenize.NAME and t.string == "__doc__"
                        for t in tokenize.generate_tokens(io.StringIO(text).readline))
        lines = [line.encode() for line in text.splitlines(keepends=True)]
        docs = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or (
                    isinstance(node, ast.Module) and not reads_doc):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
                        first.value.value, str):
                    docs.append(first)
        for d in sorted(docs, key=lambda d: d.lineno, reverse=True):  # ast columns count UTF-8 bytes
            head, tail = lines[d.lineno - 1][:d.col_offset], lines[d.end_lineno - 1][d.end_col_offset:]
            n = d.end_lineno - d.lineno
            lines[d.lineno - 1:d.end_lineno] = ([head + b'""' + tail] if n == 0 else
                                                [head + b'"""\n', *[b"\n"] * (n - 1), b'"""' + tail])
        text = b"".join(lines).decode()
        cut = {t.start[0]: t.start[1] for t in tokenize.generate_tokens(io.StringIO(text).readline)
               if t.type == tokenize.COMMENT and not (t.start[0] == 1 and t.string.startswith("#!"))}
        out = [line[:cut[i]].rstrip() + ("\n" if line.endswith("\n") else "") if i in cut else line
               for i, line in enumerate(text.splitlines(keepends=True), 1)]
        return "".join(out).encode()
    except (SyntaxError, ValueError, tokenize.TokenError):
        return source


def pack(files: dict[str, bytes]) -> str:
    """tar, xz -9e, base64 of `files` (name -> bytes); the same bytes for the same files."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(files[name]), 0o644, 0
            tar.addfile(info, io.BytesIO(files[name]))
    xz = lzma.compress(raw.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32, preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(xz).decode("ascii")


def unpack(rs_files: str) -> dict[str, bytes]:
    """Inverse of pack."""
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode(rs_files)))) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def endpoint_env(spec: ModelSpec, release: str) -> dict[str, str]:
    """The complete env map of an endpoint serving `spec`. `release` is a free label; changing it starts a new
    Runpod release (new containers) even when nothing else changed."""
    if getattr(spec, "family", "openpi") != "openpi":
        return _checked(family_env(spec, release))
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


def family_env(spec: ModelSpec, release: str) -> dict[str, str]:
    """The env map of a family other than openpi (families/common/boot.sh). Each family has its own environment tar,
    the weights are kept per checkpoint (stage.py's complete-marker copies), both saved to the volume after the first
    start is ready. Secrets (spec.secrets) are not in this map: `pgs up` adds them from the user's machine
    (launcher.secret_env). The stock image turns on huggingface_hub's hf_transfer mode without its package; it is off
    here unless the family's env turns it on."""
    return {
        "RS_FILES": pack(worker_files(spec.family, verify=spec.env.get("RS_VERIFY") == "1", entry=spec.name)),
        "RS_RELEASE": release,
        "RS_MODEL": spec.name,
        "RS_FAMILY": spec.family,
        "RS_CHECKPOINT": spec.checkpoint,
        "RS_FAST": "auto",  # the family's fast path where it has one and it passes its check; "off" = stock
        "PORT": str(PORT),
        "PORT_HEALTH": str(PORT_HEALTH),
        "RS_ENV_TAR": f"{VOLUME}/pgs/{spec.family}/env.tar",
        "RS_WEIGHTS": f"{VOLUME}/pgs/weights",
        "HF_HOME": f"{VOLUME}/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        **spec.env,
    }


def _checked(env: dict[str, str]) -> dict[str, str]:
    for key, value in env.items():
        size = len(f"{key}={value}".encode())
        if size > MAX_ENV_STRING_BYTES:
            raise ValueError(f"env {key} is {size} bytes, over the {MAX_ENV_STRING_BYTES} bytes Linux allows")
    return env
