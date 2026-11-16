#!/bin/bash
# GR00T N1.7 environment (first boot only; see common/build_lib.sh). The repo ships a uv.lock, so the venv is the repo's
# own install (README.md: `uv sync --python 3.12`) with --frozen: torch 2.9.0+cu128, transformers 4.57.3, diffusers
# 0.35.1, flash-attn 2.8.3 (prebuilt wheel), msgpack 1.1.0, and gr00t itself, editable from $ROOT/repo. No extras (the
# only one, dev, is test tooling). Left out: TensorRT (a 4.3 GB wheel; only scripts/deployment imports it) and deepspeed
# (training only; an sdist that would be built here). Nothing under gr00t/policy, gr00t/model or gr00t/data imports
# either. Built and run on an RTX 4090 on 2026-11-16 (8.4 GB venv, 49 s; NOTES.md).
source /rs/build_lib.sh
start_build https://github.com/NVIDIA/Isaac-GR00T.git 3b49067943eb1ab2eb2203b93d461c4390c78da3
cd "$ROOT/repo"
UV_PROJECT_ENVIRONMENT="$ROOT/venv" uv sync --frozen --python 3.12 \
  --no-install-package tensorrt-cu12 --no-install-package tensorrt-cu12-libs \
  --no-install-package tensorrt-cu12-bindings --no-install-package deepspeed
finish_build
