#!/bin/bash
# Cosmos 3 policy environment (first boot only; see families/common/build_lib.sh), from the repo's own uv.lock. The
# policy server guide syncs --all-extras --group=cu130-torch213-train --group=policy-server
# (docs/action_policy_droid_server.md); this keeps what the server's code path imports: the `train` extra (the server
# imports data/generator/joint_dataloader.py, which imports webdataset at module top) and the cu130-torch213-train
# group (torch 2.13.0+cu130, natten 0.21.6+cu130.torch213, flash-attn-4, torchao, triton). Left out: the guardrail,
# serve, quantization and reasoner-eval extras (imported only with guardrails on or by other tools) and the
# policy-server group (openpi-server; pgs_server.py replaces that websocket server). The CUDA groups need Python 3.13;
# CUDA 13 wheels need host driver >= 580 (Runpod hosts ran 595 on 2026-11-16).
UV_VERSION=0.11.29  # pyproject.toml [tool.uv] requires exactly this uv
source /rs/build_lib.sh
start_build https://github.com/NVIDIA/cosmos-framework.git 1f1737931214166d5e7b253def319c562cf7a1eb
cd "$ROOT/repo"
UV_PROJECT_ENVIRONMENT=$ROOT/venv uv sync --frozen --no-dev --python 3.13 --extra train --group cu130-torch213-train
ts synced_repo
# The service downloads checkpoints with `uv run --project cosmos_framework/utils/hf_cli --locked hf download`
# (utils/checkpoint_db.py _hf_download); serve.py points IMAGINAIRE_HF_CLI_ENVIRONMENT at this prebuilt copy of that
# locked environment, so a boot does not build it from PyPI.
UV_PROJECT_ENVIRONMENT=$ROOT/hf_cli uv sync --frozen --python 3.13 --no-default-groups \
  --project "$ROOT/repo/cosmos_framework/utils/hf_cli"
finish_build
