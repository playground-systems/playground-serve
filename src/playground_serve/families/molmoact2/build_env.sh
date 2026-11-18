#!/bin/bash
# MolmoAct2 environment (first start only; see ../common/build_lib.sh). The repo has no lock file, so the inference
# subset of its pyproject.toml is installed: its exact torch and torchvision pins from the PyTorch cu128 index,
# transformers inside its range, and every other version as resolved on UV_EXCLUDE_NEWER, so a rebuild resolves the
# same versions. Left out: the FastAPI server, json-numpy, hf-transfer and the simulation packages (mani-skill,
# gymnasium, imageio, tyro and the 3-D transform helpers); the checkpoint's remote code imports torch, torchvision,
# transformers, einops, numpy, PIL and huggingface_hub.
source /rs/build_lib.sh
start_build https://github.com/allenai/molmoact2.git 6070080a20321b4f498ab30f28e1d09ac465edb7
uv venv -q --python 3.12 "$ROOT/venv"
# The PyTorch index gives no upload dates, so under UV_EXCLUDE_NEWER uv finds no torch at all: install it first.
uv pip install --python "$ROOT/venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
UV_EXCLUDE_NEWER=2026-11-13T00:00:00Z uv pip install --python "$ROOT/venv/bin/python" "transformers==4.57.1" \
  "accelerate>=1.0" "safetensors>=0.4" "huggingface-hub>=0.36" "numpy>=1.26,<3" "einops>=0.7" "sentencepiece>=0.2" \
  "protobuf>=4.25"
finish_build
