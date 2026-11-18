#!/bin/bash
# FLUX 3 Action environment (first boot only; see common/build_lib.sh), installed as docs/setup.md:22-45 does it: the
# repo's uv.lock with the `encoders` extra (transformers 5.16.1 and huggingface-hub; inference requires it), then the
# NATTEN wheel built for the locked torch 2.10.0+cu128 (not in uv.lock; the video VAE imports it). Left out: `data`
# (PyAV and pyarrow, for recorded episodes and training) and `serve` (websockets and msgpack, which finish_build
# installs for pgs_server.py at versions inside that extra's ranges). Python 3.12 (uv.lock requires ==3.12.*).
source /rs/build_lib.sh
start_build https://github.com/black-forest-labs/flux-action.git aea7863fbbbda1e282fa7bde2054e290bf09b3f8
cd "$ROOT/repo"
UV_PROJECT_ENVIRONMENT=$ROOT/venv uv sync --frozen --no-dev --python 3.12 --extra encoders
TORCH_TAG=torch2100 CUDA_TAG=cu128  # the wheel built for the locked torch 2.10.0+cu128
uv pip install --python "$ROOT/venv/bin/python" "natten==0.21.6+${TORCH_TAG}${CUDA_TAG}" -f https://whl.natten.org/
finish_build
