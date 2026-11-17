#!/bin/bash
# The LLM agent's CPU worker (a Runpod load-balancing endpoint on a CPU worker, image python:3.12-slim; no volume, no
# GPU). RS_FILES is unpacked to /rs as for the other families. health.py answers Runpod's /ping on PORT_HEALTH (204
# until serve.py touches the ready file), the packages come from requirements.txt, and serve.py serves on PORT.
set -o pipefail
python3 /rs/health.py &
pip install --no-cache-dir --disable-pip-version-check -q -r /rs/requirements.txt || exit 1
cd /rs && exec python3 -u /rs/serve.py
