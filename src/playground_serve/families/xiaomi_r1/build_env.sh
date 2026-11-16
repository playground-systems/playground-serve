#!/bin/bash
# Xiaomi-Robotics-1 environment (first boot only; see common/build_lib.sh). The repo has no lock file. This is the
# deploy environment of docs/DEPLOYMENT.md at the pinned commit: Python 3.12, torch 2.8.0 / torchvision 0.23.0 cu128,
# transformers==4.57.1 exactly, and flash-attn 2.8.3 from the prebuilt wheel that page names (cu12, torch 2.8, cxx11
# ABI, cp312). The evaluation client's processor runs here too; it needs transformers, torchvision and Pillow. Versions
# the docs leave open take xr1/assets/requirements.txt's pins (accelerate, safetensors, numpy, Pillow); the rest
# resolve as of UV_EXCLUDE_NEWER. tqdm: deploy/server.py imports it (the boot check against the official server).
# Left out: torchaudio (DEPLOYMENT.md installs it; inference does not import it) and the simulator side.
source /rs/build_lib.sh
start_build https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git 0dd7aef8dc87296246aae812a1f59ccb708e5546
uv venv -q --python 3.12 "$ROOT/venv"
# The PyTorch index gives no upload dates, so under UV_EXCLUDE_NEWER uv finds no torch at all: install it first.
uv pip install --python "$ROOT/venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
FLASH_ATTN=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3
FLASH_ATTN=$FLASH_ATTN/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
UV_EXCLUDE_NEWER=2026-11-13T00:00:00Z uv pip install --python "$ROOT/venv/bin/python" "flash-attn @ $FLASH_ATTN" \
  "transformers==4.57.1" "accelerate==1.11.0" "safetensors==0.6.2" "numpy==2.1.3" "Pillow==11.3.0" \
  "huggingface-hub>=0.34,<1.0" "tqdm"
finish_build
