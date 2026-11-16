#!/bin/bash
# Flex-pi environment (first boot only; see families/common/build_lib.sh). The repo has no lock file; pyproject.toml
# pins every dependency exactly and docs/INSTALL.md §1b installs it with uv on Python 3.10 with torch 2.7.1 /
# torchvision 0.22.1 cu128, then torchcodec 0.5 from PyPI. This does the same, with two changes: deepspeed (training
# only: nothing on the inference or LIBERO eval path imports it, and its sdist needs torch at build time) is left out,
# so the package goes in with --no-deps and its other pins are listed here; and the LIBERO simulator extra is left out
# (the server takes observations from the client). Attention is torch SDPA, so no flash-attn. hf_transfer is added:
# the runpod/pytorch image sets HF_HUB_ENABLE_HF_TRANSFER=1, and huggingface_hub then refuses to download without it.
source /rs/build_lib.sh
start_build https://github.com/geyan21/flex-pi.git b27019ec454aaf5e09f9655368f02a521f6dffa5
uv venv -q --python 3.10 "$ROOT/venv"
PY="$ROOT/venv/bin/python"
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 uv pip install --python "$PY" \
  --index-strategy unsafe-best-match "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
uv pip install --python "$PY" --no-deps -e "$ROOT/repo"
uv pip install --python "$PY" "accelerate==1.12.0" "av==16.0.1" "boto3==1.35.99" "datasets==3.6.0" \
  "decord2==3.3.0" "einops==0.8.1" "gitpython==3.1.45" "huggingface-hub==0.29.2" "hydra-core==1.3.2" \
  "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "jsonlines==4.0.0" "modelscope==1.34.0" "numpy==1.26.4" \
  "omegaconf==2.3.0" "packaging==25.0" "pandas==2.2.3" "pillow==12.0.0" "pyarrow==23.0.0" "regex==2025.11.3" \
  "rich==14.2.0" "safetensors==0.5.3" "termcolor==2.5.0" "tqdm==4.66.5" \
  "timm==1.0.26" "transformers==4.49.0" "typing-extensions==4.15.0" "wandb==0.23.1" \
  "hf-transfer==0.1.9"
uv pip install --python "$PY" --reinstall --no-deps --index-url https://pypi.org/simple "torchcodec==0.5"
finish_build
