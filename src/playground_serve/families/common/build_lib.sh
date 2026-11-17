# Sourced by each family's build_env.sh, which boot.sh runs on a first boot (no complete environment tar on the volume).
# The family builds /opt/pgs = repo/ (its model code at a pinned commit) + venv/ + python/ (uv-managed, so the tar does
# not depend on the image's own Python) on the container disk. Nothing here touches the volume: once the server is
# ready, stage.py fill saves /opt/pgs there as one tar (RS_ENV_TAR). Weights are not in the tar (stage.py fetch).
set -euo pipefail
ROOT=/opt/pgs
UV_VERSION=${UV_VERSION:-0.11.28}  # a family whose repo pins another uv sets UV_VERSION before sourcing this
SERVER_DEPS=("websockets==15.0.1" "msgpack==1.1.0" "pillow>=13.0.0")  # what pgs_server.py and server_decode.py import
# The uv cache sits next to /opt/pgs on the same disk, so uv hardlinks instead of copying (not in the tar).
export UV_PYTHON_INSTALL_DIR=$ROOT/python UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR=/opt/uvcache
export GIT_LFS_SKIP_SMUDGE=1
ts() { echo "== $(date -u +%FT%T.%3NZ) $*"; }

start_build() {  # start_build REPO_URL COMMIT
  ts start "$(nproc) cpus"
  pip install -q "uv==$UV_VERSION"
  rm -rf "$ROOT"
  mkdir -p "$ROOT"
  git clone -q --filter=blob:none "$1" "$ROOT/repo"
  git -C "$ROOT/repo" checkout -q "$2"
  ts cloned "$(git -C "$ROOT/repo" rev-parse HEAD)"
}

finish_build() {
  uv pip install --python "$ROOT/venv/bin/python" "${SERVER_DEPS[@]}"
  uv pip freeze --python "$ROOT/venv/bin/python" > "$ROOT/freeze.txt"
  ts synced "$(du -sh "$ROOT/venv" | cut -f1) venv, $(du -sh "$ROOT/python" | cut -f1) python"
}
