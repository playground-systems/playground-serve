"""Which model path the openpi worker serves. Standard library only: boot.sh runs it on the image's python3 before the
openpi environment exists.

fast   realtime-vla's Triton kernels captured as one CUDA graph (github.com/dexmal/realtime-vla, MIT, RTVLA_COMMIT),
       the same bf16 weights converted from the same checkpoint, the flow state in float32 as openpi keeps it, openpi's
       own tokenizer, normalization and output transforms (fast_policy.py). Only for a model and GPU in FAST, where it
       was measured served end to end, and only after its boot self-test reproduced stock openpi's stored actions
       (selftest.py, selftest_<model>.json); serve.py exits with FALLBACK_EXIT if it cannot build it or the check fails.
stock  openpi's own policy (serve.py): every other model and GPU, RS_FAST=off (`pgs up --stock-path`), and the
       fallback. The worker's log and the metadata frame say which path serves and why.

    python3 fastpath.py decide GPU_NAME    log {"rs": "path", "path", "reason"}, reason to PATH_REASON; exit 0 = fast
    python3 fastpath.py clone DIR          realtime-vla at RTVLA_COMMIT in DIR (kept in the environment tar)
    python3 fastpath.py fallback RC        after the fast server gave up: log {"rs": "fallback", "reason"}, reason to
                                           PATH_REASON (serve.py's REASON_FILE, else the exit code RC)
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

# Model -> GPU names (nvidia-smi) where the fast path was measured served and its actions checked against stock
# (docs/BENCHMARKS.md: RTX 4090, served infer_ms 31 to 34 ms against 75 to 79 for stock). Not here: the RTX 5090
# (needs torch's cu128 build swapped into the shared environment), the L40S and H100 (measured in-process only, and not
# in any model's GPU pools), the A40 (not measured).
FAST = {"pi05_libero": ("NVIDIA GeForce RTX 4090",)}
RTVLA_REPO = "https://github.com/dexmal/realtime-vla.git"
RTVLA_COMMIT = "eb46c8d"
FALLBACK_EXIT = 3  # serve.py in fast mode: the fast path is not usable here; boot.sh starts the stock path
REASON_FILE = "/tmp/rs_fast_failed"  # serve.py in fast mode writes why it gave up
PATH_REASON = "/tmp/rs_path_reason"  # why this path serves; boot.sh passes it to serve.py as RS_PATH_REASON
HERE = pathlib.Path(__file__).resolve().parent


def reference(model: str) -> pathlib.Path:
    return HERE / f"selftest_{model}.json"


def decide(mode: str, model: str, gpu: str, have_reference: bool) -> tuple[str, str]:
    """("fast" or "stock", why). mode: the endpoint's RS_FAST, "auto" (default) or "off"."""
    if mode != "auto":
        return "stock", f"fast path turned off (RS_FAST={mode}, `pgs up --stock-path`)"
    if model not in FAST:
        return "stock", f"no fast path for {model}"
    if gpu not in FAST[model]:
        return "stock", f"fast path not measured on {gpu or 'this GPU'}, only on {', '.join(FAST[model])}"
    if not have_reference:
        return "stock", f"no self-test reference for {model}"
    return "fast", f"realtime-vla graph on {gpu}, if its boot self-test passes"


def fallback_reason(rc: str, reason_file: str = REASON_FILE) -> str:
    """Why the stock path serves after the fast server exited with `rc` ("clone": realtime-vla was not fetched)."""
    try:
        return pathlib.Path(reason_file).read_text().strip() or f"fast path exited with code {rc}"
    except OSError:
        if rc == "clone":
            return f"fast path failed: could not fetch realtime-vla at {RTVLA_COMMIT}"
        return f"fast path exited with code {rc} before its self-test passed"


def log(phase: str, **kw) -> None:
    now = time.time()
    print(json.dumps({"rs": phase, "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z",
                      **kw}), flush=True)


def clone(target: str) -> int:
    t = time.time()
    rc = 0 if os.path.isdir(os.path.join(target, ".git")) else subprocess.call(["git", "clone", "-q", RTVLA_REPO, target])
    rc = rc or subprocess.call(["git", "-C", target, "checkout", "-q", RTVLA_COMMIT])
    log("rtvla_clone", rc=rc, s=round(time.time() - t, 1))
    return rc


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "decide":
        model = os.environ.get("RS_MODEL", "")
        path, reason = decide(os.environ.get("RS_FAST", "auto"), model, argv[1].strip(), reference(model).is_file())
        pathlib.Path(PATH_REASON).write_text(reason)
        log("path", path=path, reason=reason)
        return 0 if path == "fast" else 1
    if len(argv) == 2 and argv[0] == "clone":
        return clone(argv[1])
    if len(argv) == 2 and argv[0] == "fallback":
        reason = fallback_reason(argv[1])
        pathlib.Path(PATH_REASON).write_text(reason)
        log("fallback", reason=reason)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
