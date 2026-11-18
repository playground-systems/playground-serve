#!/bin/bash
# Playground Serve worker boot (stock runpod/pytorch image, scripts in /rs from the RS_FILES env var, network volume at
# /runpod-volume). Health responder first. The environment comes from the volume's tar (RS_ENV_TAR, shared by all
# models) when it holds a complete one, else it is built on the container disk (build_env.sh). A gs:// checkpoint comes
# from the volume when it holds a complete copy, else it downloads to the container disk at the same time as the
# environment (stage.py fetch) and serve.py waits for it after its imports. Once the server is ready, stage.py fill
# saves what this boot made locally to the volume in the background, so the first start never waits on the volume.
# Model path (fastpath.py): the fast path where it was measured (pi05_libero on an RTX 4090) unless RS_FAST=off; its
# checked copy on the volume (RS_FAST_VOLUME) replaces the checkpoint at start; without one, serve.py converts the
# downloaded checkpoint itself, after a separate process compiled its GPU kernels with placeholder weights (fast_policy.py
# warm, while serve.py imports and the checkpoint downloads). If the fast server exits before its self-test passed, the
# stock openpi server starts in its place and says why (log, metadata). The boot line names the release (RS_RELEASE_ID).
# Endpoint env (playground_serve.bundle.endpoint_env) sets PORT, PORT_HEALTH, RS_ENV_TAR, OPENPI_DATA_HOME, RS_FAST,
# the model (RS_MODEL, RS_OPENPI_CONFIG, RS_CHECKPOINT) and the JAX cache variables.
export RS_BOOT_T=$(date +%s.%N)
ENV_TAR=${RS_ENV_TAR:-/runpod-volume/rs/env.tar}
now() { date -u +%FT%T.%3NZ; }
echo "{\"rs\":\"boot\",\"epoch\":$RS_BOOT_T,\"utc\":\"$(now)\",\"release\":\"${RS_RELEASE:-}\",\"release_id\":\"${RS_RELEASE_ID:-}\",\"model\":\"${RS_MODEL:-}\"}"
rm -f /tmp/rs_ready /tmp/rs_fast_failed
python3 /rs/health.py &
if [ -f /opt/rs_env_ok ]; then  # a restarted container kept its disk
  ENV_FROM=local
elif python3 /rs/stage.py check-tar "$ENV_TAR"; then
  ENV_FROM=volume
else
  ENV_FROM=build
fi
export RS_FAST_DIR=/rs_local/fast RS_FAST_VOLUME="/runpod-volume/fast/${RS_MODEL:-model}"
python3 /rs/fastpath.py decide "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" && RS_PATH=fast || RS_PATH=stock
ckpt() {
  case "$RS_CHECKPOINT" in
    gs://*)
      export RS_CKPT_VOLUME="${OPENPI_DATA_HOME:-/runpod-volume/openpi_data}/${RS_CHECKPOINT#gs://}"
      if python3 /rs/stage.py check "$RS_CKPT_VOLUME"; then
        export RS_CKPT_DIR=$RS_CKPT_VOLUME
      else  # the environment build downloads too: it gets the bandwidth first (stage.py fetch's third argument)
        export RS_CKPT_DIR="/rs_local/ckpt/${RS_CHECKPOINT#gs://}"
        python3 /rs/stage.py fetch "$RS_CHECKPOINT" "$RS_CKPT_DIR" $([ $ENV_FROM = build ] && echo /opt/rs_env_ok) &
      fi ;;
  esac
}
[ $RS_PATH = fast ] && python3 /rs/stage.py check "$RS_FAST_VOLUME" || ckpt
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
  # A fresh venv has no bytecode: the server's imports would compile it one module at a time (15 to 37 s measured).
  # Compile the packages it imports in parallel, biggest first, while it imports; the env tar keeps the bytecode.
  export RS_PYC_DONE=/tmp/rs_pyc_done  # stage.py fill tars the environment after this
  (cd /opt/rs/openpi/.venv/lib/python3*/site-packages && nice -n 5 /opt/rs/openpi/.venv/bin/python -m compileall \
    -q -j 16 -x '/(tests?|testing)/' torch transformers jax jaxlib scipy sympy flax orbax optax chex numpy etils \
    tensorstore equinox jaxtyping beartype treescope tyro rich augmax einops ml_dtypes safetensors sentencepiece \
    fsspec gcsfs aiohttp google tqdm typeguard websockets msgpack PIL pydantic numpydantic huggingface_hub \
    /opt/rs/openpi/src /opt/rs/openpi/packages/openpi-client/src; touch $RS_PYC_DONE) >/dev/null 2>&1 &
fi
# Upgrade cached environments before any untrusted image reaches Pillow. Fail closed.
/opt/rs/openpi/.venv/bin/python -c 'import PIL; assert int(PIL.__version__.split(".")[0]) >= 13' 2>/dev/null || \
  (python3 -m pip install -q 'uv==0.11.28' && uv pip install --python /opt/rs/openpi/.venv/bin/python 'pillow>=13.0.0') || exit 1
if [ $RS_PATH = fast ]; then
  RC=clone
  if python3 /rs/fastpath.py clone /opt/rs/realtime-vla; then
    if [ -n "$RS_CKPT_DIR" ]; then  # no checked copy to load: compile the kernels while serve.py imports and downloads
      export RS_WARM_DONE=/tmp/rs_warm_done
      rm -f $RS_WARM_DONE
      (cd /opt/rs/openpi && .venv/bin/python -u /rs/fast_policy.py warm; touch $RS_WARM_DONE) &
    fi
    RS_CKPT_VOLUME= python3 /rs/stage.py fill &  # the fast path saves its checked copy, not the checkpoint
    FILL=$!
    (cd /opt/rs/openpi && RS_PATH=fast exec .venv/bin/python -u /rs/serve.py)
    RC=$?  # only reached when the fast server gave up
    kill $FILL 2>/dev/null
  fi
  rm -f /tmp/rs_ready
  python3 /rs/fastpath.py fallback $RC
  [ -n "$RS_CKPT_DIR" ] || ckpt
fi
export RS_PATH_REASON="$(cat /tmp/rs_path_reason 2>/dev/null)"
RS_FAST_DIR= python3 /rs/stage.py fill &  # the stock path saves no fast copy
cd /opt/rs/openpi && RS_PATH=stock exec .venv/bin/python -u /rs/serve.py
