#!/bin/bash
# Playground Serve worker boot for a family other than openpi (stock runpod/pytorch image, scripts in /rs from RS_FILES:
# this file, build_lib.sh, pgs_server.py, pgs_msgpack_numpy.py, the openpi worker's health.py, server_decode.py and
# stage.py, and the family's serve.py, build_env.sh and other files; network volume at /runpod-volume). The openpi
# worker's first-start design: health responder first; the environment (/opt/pgs) comes from the volume's tar
# (RS_ENV_TAR, one per family) when it holds a complete one, else build_env.sh builds it on the container disk. The
# weights come from the volume (RS_WEIGHTS/<checkpoint>) when it holds a complete, marked copy, else stage.py fetch
# downloads them to the container disk at the same time as the build, and serve.py waits for them before loading
# (pgs_server.wait_for_weights). Once the server is ready, stage.py fill saves what this boot made locally to the volume
# (temporary names, size checks, marker last). The boot line names the release (RS_RELEASE_ID), which pgs_server.py
# also sends in its metadata. serve.py picks its fast path itself (RS_FAST=auto) and falls back to stock in the same
# process; if it exits with 3 before it is ready, it runs once more with RS_FAST=off and the reason in RS_PATH_REASON.
# Endpoint env (playground_serve.bundle.family_env): PORT, PORT_HEALTH, RS_MODEL, RS_FAMILY, RS_CHECKPOINT, RS_FAST,
# RS_ENV_TAR, RS_WEIGHTS, HF_HOME and the family's own settings. RS_SELF_FETCH=1 (a family's env): the model's own
# loader downloads the checkpoint (by repo id and revision into HF_HOME), so stage.py does not.
export RS_BOOT_T=$(date +%s.%N)
export RS_ENV_DIR=/opt/pgs
ENV_TAR=${RS_ENV_TAR:?RS_ENV_TAR is not set}
now() { date -u +%FT%T.%3NZ; }
echo "{\"rs\":\"boot\",\"epoch\":$RS_BOOT_T,\"utc\":\"$(now)\",\"release\":\"${RS_RELEASE:-}\",\"release_id\":\"${RS_RELEASE_ID:-}\",\"model\":\"${RS_MODEL:-}\",\"family\":\"${RS_FAMILY:-}\"}"
rm -f /tmp/rs_ready
python3 /rs/health.py &
if [ -f /opt/rs_env_ok ]; then  # a restarted container kept its disk
  ENV_FROM=local
elif python3 /rs/stage.py check-tar "$ENV_TAR"; then
  ENV_FROM=volume
else
  ENV_FROM=build
fi
KEY=${RS_CHECKPOINT#*://}
KEY=${KEY//[^A-Za-z0-9._@\/-]/_}
export RS_CKPT_VOLUME="${RS_WEIGHTS:-/runpod-volume/pgs/weights}/$KEY"
if [ "${RS_SELF_FETCH:-}" = 1 ]; then  # the model's own loader downloads RS_CHECKPOINT into HF_HOME on the volume
  unset RS_CKPT_VOLUME  # so stage.py neither fetches nor fills a checkpoint copy, and serve.py waits for none
elif python3 /rs/stage.py check "$RS_CKPT_VOLUME"; then
  export RS_CKPT_DIR=$RS_CKPT_VOLUME
else  # the environment build downloads too: it gets the bandwidth first (stage.py fetch's third argument)
  export RS_CKPT_DIR="/rs_local/weights/$KEY"
  python3 /rs/stage.py fetch "$RS_CHECKPOINT" "$RS_CKPT_DIR" $([ $ENV_FROM = build ] && echo /opt/rs_env_ok) &
fi
if [ $ENV_FROM = local ]; then
  echo "{\"rs\":\"unpack_skipped\",\"utc\":\"$(now)\"}"
elif [ $ENV_FROM = volume ]; then
  t=$(date +%s.%N)
  tar -C / -xf "$ENV_TAR" || exit 1
  touch /opt/rs_env_ok
  echo "{\"rs\":\"unpack_done\",\"s\":$(awk "BEGIN{print $(date +%s.%N) - $t}"),\"bytes\":$(stat -c %s "$ENV_TAR"),\"utc\":\"$(now)\"}"
else
  echo "{\"rs\":\"build_start\",\"utc\":\"$(now)\"}"
  bash /rs/build_env.sh || exit 1
  touch /opt/rs_env_ok
  echo "{\"rs\":\"build_done\",\"utc\":\"$(now)\"}"
  # A fresh venv has no bytecode: compile it in the background while the server imports (the tar keeps it).
  export RS_PYC_DONE=/tmp/rs_pyc_done  # stage.py fill tars the environment after this
  (nice -n 5 /opt/pgs/venv/bin/python -m compileall -q -j 16 -x '/(tests?|testing)/' \
    /opt/pgs/venv/lib/python3*/site-packages; touch $RS_PYC_DONE) >/dev/null 2>&1 &
fi
mkdir -p "${HF_HOME:-/runpod-volume/hf}" 2>/dev/null
# The container gets RUNPOD_CPU_COUNT CPUs (16 on an RTX 4090 worker) of a host with many more; uncapped, torch starts
# 64 threads, which the container's CPU share throttles. MolmoAct2 DROID fast path, same host, back to back: p90 117 to
# 126 ms with 16 threads against 124 to 176 ms with 64. Thread count only: no result changes.
[ -n "${RUNPOD_CPU_COUNT:-}" ] && export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$RUNPOD_CPU_COUNT}
python3 /rs/stage.py fill &
cd /opt/pgs/repo || exit 1
/opt/pgs/venv/bin/python -u /rs/serve.py
RC=$?
if [ $RC = 3 ] && [ ! -f /tmp/rs_ready ]; then  # the fast path and the stock path in its process both failed
  export RS_PATH_REASON="$(cat /tmp/rs_path_reason 2>/dev/null)"
  echo "{\"rs\":\"fallback\",\"utc\":\"$(now)\",\"reason\":\"restarting serve.py with RS_FAST=off\"}"
  RS_FAST=off exec /opt/pgs/venv/bin/python -u /rs/serve.py
fi
exit $RC
