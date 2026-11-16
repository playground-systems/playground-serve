#!/bin/bash
# ABC-VLA environment (first boot only; see common/build_lib.sh). The repo's uv.lock is gitignored, so this installs
# the repo without its dependencies, then what the serving path imports (deploy/serve_policy.py -> deploy.policy,
# abc_minimal.policy / vla / gemma, deploy.websocket_server and its image codec): torch 2.11.0+cu128 as pyproject.toml
# pins it, sentencepiece>=0.2.0 (the Gemma 3 tokenizer), tyro>=1.0.5 (the config dataclasses), ftfy and regex (the DiT
# policy module imports them lazily), and OpenCV (the server's JPEG decode; the headless build, same cv2 API, no
# libGL). Left out: the simulator (mujoco, mujoco-warp, warp-lang, viser), training and data tools (wandb, av, mcap,
# torchcodec, imageio) and the robot drivers of the deploy extra (i2rt, dynamixel, realsense). Python 3.12.
source /rs/build_lib.sh
start_build https://github.com/amazon-far/abc.git d0832d12651d1b260a652861a14648dc5f3660c7
uv venv -q --python 3.12 "$ROOT/venv"
# The PyTorch index gives no upload dates, so under UV_EXCLUDE_NEWER uv finds no torch at all: install it first.
uv pip install --python "$ROOT/venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
UV_EXCLUDE_NEWER=2026-11-13T00:00:00Z uv pip install --python "$ROOT/venv/bin/python" "numpy" "sentencepiece>=0.2.0" \
  "tyro>=1.0.5" "ftfy" "regex" "opencv-python-headless"
uv pip install --python "$ROOT/venv/bin/python" --no-deps -e "$ROOT/repo"
finish_build
