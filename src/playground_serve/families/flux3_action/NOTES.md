# FLUX 3 Action family (flux3_action)

**Status: experimental (owner decision, 2026-11-16). Served and verified on an A40 (stock path: the official eager
setting), 3.2 s per call.** The official compiled setting fails on this stack (below). No optimized path.

**License: FLUX Kommunity License v1.0, NON-COMMERCIAL.** Non-commercial use; commercial use of outputs only for a
Qualifying User (under US$5M gross annualized revenue with affiliates). Code: Apache-2.0
(`black-forest-labs/flux-action@aea7863f`). Text encoder Qwen3-VL-4B-Instruct: Apache-2.0. Self-hosting for
non-commercial use is what the license allows; a commercial user must qualify under the license first. The spec says
`NON_COMMERCIAL = True`, the metadata carries `"non_commercial": true` and the license text.

Entry `flux3_action_droid` (hosted id `flux-3-action-droid`): a 7B world action model on DROID, 32 x 8 absolute joint
commands and gripper closed fraction at 15 Hz, three cameras, no history.

## Access

Both `black-forest-labs/flux-3-action-droid` and `black-forest-labs/flux-3-action-base` (text encoder, video VAE) are
gated on Hugging Face. The owner accepted both licenses and saved a read token on the bring-up machine; the bring-up
script reads it at run time (`--hf-token-file`) and passes it only as the endpoint env value `HF_TOKEN` (the spec lists
it in `secrets`); its output and the saved worker logs were checked to hold no copy of it. `serve.py` stops at start
with a clear message when `HF_TOKEN` is missing.

## What the adapter does

- `serve.py` loads the pinned DROID root package (validated BF16) with `load_serving_policy`, as `flux-action
  serve-robolab` does, and calls the official `RoboLabPolicy.infer`; its `action` comes back as `actions`. A request
  may carry `rs_seed` (`RoboLabPolicy.infer`'s own `seed`); without it the official per-request counter seeds.
- Keys are the RoboLab ones: `observation/{wrist_image_left,exterior_image_1_left,exterior_image_2_left}` (or one
  `observation/image` composite), `observation/joint_position`, `observation/gripper_position`, `prompt`. The
  metadata's `image_size` is [360, 640], so the client's default jpeg mode sends 360 x 640 frames; the composite must
  be exactly 540 x 640.
- Options: `RS_COMPILE_DIT` (spec default "0", see below), `RS_SUBFOLDER`, `RS_OFFLOAD_TEXT_ENCODER=1`.

## Measured (A40, CA-MTL-1, 1.22 USD/hr, 2026-11-16)

- First start on a new volume: environment build 53 s (uv sync of the repo's lock plus the NATTEN wheel), the gated
  download (DROID package 13.9 GB plus the base repo's text encoder and VAE) and load, then the first eager request:
  260 s after boot. Later starts read the weights from the volume.
- The official compiled setting (the CLI's default `--compile-dit`: the prepared BF16 DiT and the VAE encoder compiled
  with CUDA graphs) failed at every start, during the official warm-up inside `load_serving_policy`:
  "Inplace update to inference tensor outside InferenceMode is not allowed" in `CUDAGraph.capture_begin` (torch
  2.10.0+cu128 from the repo's lock, NATTEN 0.21.6). It failed also with no eager inference earlier in the process
  and after capturing a trivial graph first. Not debugged further (owner decision: experimental entry, little more
  spending). So the entry serves the official eager setting (`--no-compile-dit`, `RS_COMPILE_DIT=0` in the spec),
  which is the one the repo's own `replay-fixtures` command uses to compare actions ("eager runs are reproducible";
  compiled runs "can differ by about 0.015 rad").
- Eager, in the worker: 3130 to 3150 ms per call after the first (3634 to 3756 ms), on the recorded DROID frames at
  360 x 640; the same request twice gives identical actions (0.0).

## Verified (A40, 2026-11-16)

- The official eager setting in its own process (`serve.py eager`: `load_serving_policy(compile_dit=False)` and
  `RoboLabPolicy.infer` at seed 7, recorded DROID observation 0 at 360 x 640) against the served worker: the
  in-worker wrapper equals `RoboLabPolicy.infer` exactly (0.0), the served process equals the other process exactly
  (0.0), and through the gateway in raw mode the actions equal the official ones to the log's rounding (1e-6 max).
- JPEG mode (the client's default, q90) against raw: 0.0048 mean / 0.021 max; a new seed moves the actions by 0.040
  mean / 0.183 max.
- Served `infer_ms` (through the gateway, US east coast client, CA-MTL-1): 3209 to 3228 ms raw, 3180 to 3223 ms JPEG;
  round trip 3263 to 3917 ms. GPU memory in use 33.6 GB of the A40's 46.
- Start from the volume (environment tar and weights there): boot to the first logged result 135 s, ready 202 s in a
  verify run (it loads the model twice).

| GPU (pool) | USD/hr | stock infer_ms (served) | optimized | Measured |
|---|---:|---:|---|---|
| A40 (AMPERE_48) | 1.22 | 3180 to 3228 | none (see above) | yes |
| faster GPUs | | not measured | | owner decision: one verified stock latency is enough |


- `flux-3-action-so101` is not built: it reads 8 past observation ticks and the commands already issued, needs
  `select_action` at every 30 Hz tick and `reset()` per episode, so the server must hold per-session state.
- The repo's guidance-distilled (`gd`) and step-distilled (`sd`) packages change the model or its steps, and the
  `fp8r` packages are FP8: if served, they would be separate entries labeled as such.

## Requests for the core

Status after the merge with the core (2026-11-16): see `families/cosmos3_policy/NOTES.md`. The core now has
`hosted_id`, `experimental` and `license` (this entry is marked experimental in `pgs models`, and its license text
says non-commercial), and endpoint secrets: `pgs up` reads `HF_TOKEN` from the environment variable or the token file of
`hf auth login`, puts it only into the endpoint's env and never saves or prints it. `verify_obs.json` ships only for
bring-up runs (the core ships files named `verify*` only with `RS_VERIFY=1`): the endpoint env is 28 kB, 86 kB in a
verify run, both under Runpod's request limit (about 100.3 kB, measured).

## Pools

Only the A40 (AMPERE_48) is in the spec (2026-11-16): the one card measured, at the lowest price of the 48 GB
cards. The 32 GB RTX 5090 is too small for the 33.6 GB in use; the L40S (ADA_48_PRO, not measured, not cheaper) was
dropped from the default pools.
