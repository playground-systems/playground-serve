#!/bin/bash
# Called by boot.sh when the volume holds no complete environment tar: build /opt/rs = openpi checkout + .venv on the
# container disk. stage.py fill saves it to the volume as one tar once the server is ready. The tar only works with the
# image it was built on (uv uses the image's Python 3.11) and holds no model.
set -euo pipefail
OPENPI_COMMIT=${OPENPI_COMMIT:-215abfb217dbac7d5f1273282331b9b1866c0479}
# The uv cache sits next to /opt/rs on the same disk, so uv hardlinks instead of copying (not in the tar).
export UV_PYTHON_INSTALL_DIR=/opt/rs/python UV_CACHE_DIR=/opt/uvcache GIT_LFS_SKIP_SMUDGE=1
ts() { echo "== $(date -u +%FT%T.%3NZ) $*"; }

ts start "$(nproc) cpus, cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
pip install -q uv
mkdir -p /opt/rs/openpi
cd /opt/rs/openpi
git init -q
git fetch -q --depth 1 https://github.com/Physical-Intelligence/openpi.git "$OPENPI_COMMIT"
git checkout -q FETCH_HEAD
ts fetched "$(git rev-parse HEAD)"
uv sync --frozen --no-dev
ts synced "$(du -sh /opt/rs/openpi/.venv | cut -f1) venv"
