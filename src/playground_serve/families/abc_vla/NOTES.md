# abc_vla: ABC-VLA (Amazon FAR) on bimanual YAM

Entry `abc_vla_yam` (bimanual YAM; the ABC simulator's robot and the real one). Gemma 3 4B with SigLIP at 224 (bf16), an fp32 attention pool and a
42M fp32 diffusion head, 10 Euler steps; 4.34B parameters. Code `amazon-far/abc@d0832d12`; the checkpoint is the VLA
parent `prepare.py --vla-pretrained` downloads (`vla_abc130k_v2/200000.pt`, 8.80 GB, sha256 checked once).

**Status: served and verified on a GPU, stock and fast path (RTX 4090 and H100, 2026-11-16).** Every number below is
measured unless it says estimate. Round 2 (closing the gap to a hosted service) added the compiled variant; its
results are in "Round 2" below, the round 1 sections after it describe the bit-identical graph.

## License

Code Apache-2.0 (repo LICENSE). Weights: the checkpoint sidecar (`200000.json`, read 2026-11-16) says Apache-2.0 and
that the Gemma Terms of Use apply, since the backbone derives from Gemma 3. Self-hosting is allowed; Gemma's
prohibited-use policy passes through to anyone the endpoint serves. Not gated; hosted by the authors
(`abc-data.timehorizons.org`), not on Hugging Face. Recorded in `spec.LICENSE` and the metadata's `license`.

## What serves

- **stock**: the repo's own server path: `deploy.policy.vla_policy.Policy(Args(policy=InferenceConfig(checkpoint_path=...),
  policy_type="vla").vla_config())` and `deploy.websocket_server.infer_observation` per request. Zero noise unless the
  client sends `noise`, so the same observation always gives the same actions.
- **fast** (`abc_fast.py`, on the RTX 4090 and the H100): the same modules, weights, dtypes and shapes, run as one
  CUDA graph per (prompt, RTC prefix length), with the per-prompt work done once (tokens, text embeddings, attention
  masks, rotary tables). Round 2: the graph body compiled by inductor with emulated precision casts (default,
  `RS_FAST_COMPILE=1`), its cache on the volume per GPU type; the uncompiled graph (bit-identical to stock) is the
  fallback, then stock.
- Self-test at every start (`selftest_abc_vla_yam.json`): one stored observation (frames of the repo's
  `assets/bottles_real.mp4` as the client sends them, JPEG q90 at 224) with zero noise and with a 4-step RTC prefix.
  Stock must match the stored stock actions (warning only, max 0.1) and fast must match live stock within max 0.1 and
  mean 0.01 (the pi0.5 rule; round 1 used 0.01 and 0.001 for the bit-identical graph), twice. Else the next variant,
  and at last stock, with the reason in `path_reason`. A crash while the fast path builds leaves
  `/tmp/rs_fast_failed`, and the next start in that container serves stock.
- Request keys as the repo's server: `images` {top, left, right} (HxWx3 from the jpeg mode, transposed; 3xHxW and JPEG
  bytes also work), `state` (14), `prompt`, optional `action_prefix` + `prefix_length` (RTC) and `noise`. Reply
  `actions` (30, 14), absolute joint targets at 30 Hz. The repo's real-robot runs execute 20 of each chunk with RTC;
  the client decides how many to run.

## Round 2: the gap to a hosted service (2026-11-16)

A hosted service measured on this model (H100 in us-west-2): server inference 17.5 ms p50 and server total 19.5 ms,
of which its preprocess 1.9 ms (image decode and resize, reported apart), 20 x 14 actions per reply. Its first action
was within 0.0043 of our stored zero-noise stock action for the same frames: the same checkpoint. Chunk and steps: the
checkpoint's chunk is 30 actions (`chunk_length` in its config) and the server returns all 30; the repo's real-robot
runs execute 20 of each chunk with RTC, which is what the hosted service returns. The default and documented step
count is 10 (`InferenceConfig.diffusion_steps`; the repo's DAgger fine-tunes run 5, the VLA parent's own commands use
the default). Ours stays at 30 actions and 10 steps; the hosted service's steps are not stated.

Where our 37.0 ms (RTX 4090, round 1) went: the graph was GPU-bound (6,191 kernels, 34 ms of GPU time). On an H100 the
same bit-identical graph takes 28.6 ms: small kernels, about 4 to 5 microseconds each, launch- and latency-bound even
in a graph. The hosted service's number covers the model call; ours covered JPEG decode as well (1.1 ms on the 4090
host, about 4 ms on the H100 host).

What changed: the graph body compiled by inductor (the same body, fused kernels) inside the same CUDA graph, with
emulated precision casts, and inductor's cache on the volume per GPU type (`RS_COMPILE_CACHE`).

### Stage table (ms, p50)

| | RTX 4090 stock | RTX 4090 round 1 | RTX 4090 round 2 | H100 stock | H100 round 1 path | H100 round 2 |
|---|---:|---:|---:|---:|---:|---:|
| JPEG decode of 3 frames + server | | 1.2 (est.) | 1.2 to 2.0 | 2.8 | 3.1 | 4.4 |
| SigLIP and the embeddings | | 8.5 | | | | |
| Gemma prefill | | 19.0 | | | | |
| Pool and 10 head steps | | 6.4 | | | | |
| Graph replay (model) | | 33.9 | 26.2 | | 28.6 | 15.6 |
| Policy call (preprocess + model + postprocess) | 85.5 (in-process) | 33.6 to 34.5 (in-process) | 27.6 to 27.9 | 64.3 | 28.2 | 16.1 |
| **Served infer_ms p50 / p90** | **84.0** | **37.0 / 37.5** | **28.8 / 29.8** | **67.1 / 74.9** | **31.3 / 32.4** | **20.5 / 20.6** |

Sources: served rows through the gateway (25 to 40 calls) with the reply's policy_timing (round 2 and the H100); the
decode row is served minus the policy call; stage rows from graph replays of each stage of the bit-identical graph on
the 4090. JPEG decode alone of the 3 frames: 1.1 ms (4090 host). Like for like with the hosted service on the H100:
model 15.6 ms against its 17.5 ms inference; policy call 16.1 ms; served with decode 20.5 ms against its 19.5 ms total.

Accuracy (compiled against stock, same noise):
- RTX 4090 boot self-test: 0.0032 mean / 0.022 max (zero noise); served against round 1's served stock on 4
  observations: 0.002 to 0.005 mean, 0.010 to 0.033 max (before emulated casts).
- H100 boot self-test with emulated casts: 0.0012 / 0.0069 (zero noise), 0.0008 / 0.0087 (RTC prefix 4).
- For scale: a new noise draw moves the actions by 0.045 to 0.100 mean and 0.26 to 0.43 max.

Start times:
- RTX 4090, first start with a cold compile cache: ready 293 s after container start (the compile 216 s); later
  start with the cache on the volume: ready 70.6 s (the compile 25.8 s); connect to first action 99 s.
- H100, no volume: the compile 199 s on a cold cache. Emulated casts change inductor's cache key: one more cold compile
  per GPU type (seen on the 4090).

What is left: the model on the 4090 is GPU-bound matrix arithmetic (Gemma 19 ms of the 34 in the uncompiled graph); on
the H100 the compiled graph is 15.6 ms, under the hosted service's 17.5 ms. Decode and the server add 1 to 4 ms
depending on the host. The A40 (second pool) still serves stock: not measured.

## Round 1 measurements (RTX 4090, US-IL-1, 2026-11-16)

| What | Stock | Fast |
|---|---:|---:|
| Served `infer_ms` p50 (gateway, 20 to 40 calls, JPEG decode included) | 84.0 | 37.0 (p90 37.5) |
| In-process per call p50 | 85.5 to 106.6 | 33.6 to 34.5 |
| CUDA graph replay alone | | 33.9 (6,191 kernels, 34.1 ms of GPU time: GPU-bound) |
| Round trip from the US east coast (laptop on Wi-Fi), p50 | 173 | 98 |
| GPU memory in use (nvidia-smi) | 9.0 GB | 9.2 GB |

Action checks:
- Playground Serve path (client jpeg q90 -> gateway -> pgs_server -> adapter) against the official server and the
  official client (`deploy.websocket_server.WebsocketPolicyServer`'s own handler and
  `deploy.client.websocket_client_policy.WebsocketClientPolicy`, same weights, run in the worker) on the same decoded
  arrays: **0.0 difference** on 4 real-frame observations with 3 prompts.
- Fast against stock: **0.0** (mean and max) on the same 4 observations with zero noise, with a random noise and with
  a 4-step RTC prefix; through the gateway, served fast against served stock: 0.0 on all 4.
- For scale (stock, same 4 observations): a new noise draw moves the actions by 0.045 to 0.100 mean and 0.26 to 0.43
  max. The official client's own compression (cv2 letterbox, JPEG q75) instead of Playground Serve's (PIL letterbox,
  q90): 0.024 to 0.062 mean, 0.10 to 0.67 max. Raw 480x640 frames (the model letterboxes them on the GPU): 0.014 to
  0.041 mean, 0.08 to 0.32 max. Repeat calls: 0.0.

Measured and not used:
- Cutting the sequence at the prompt's last token instead of padding to 256 (the padding gets zero attention): 33.6
  instead of 34.3 ms, but actions move by 0.0017 to 0.0034 mean and up to 0.074 max (other GEMM shapes, other
  rounding). Not worth 0.7 ms.
- torch.compile (inductor, default mode) of the same body under the CUDA graph: replay 26.3 ms instead of 34.0, actions
  0.002 to 0.005 mean and 0.010 to 0.033 max from stock, and 126 s of compiling at the first call of every cold start
  (the inductor cache on the volume would cut that, not measured). Left out: not bit-identical and slow to start.
  The repo's own `--fast-inference` (max-autotune compile plus TF32 for the fp32 head) was not measured; TF32 is below
  the stock path's fp32 for the head and pool.

Start times:
- First start on an empty volume: worker ready 75.7 s after container start (environment build 49 s, 6.7 GB venv;
  checkpoint download 18.3 s at 480 MB/s from the authors' host; load 7.6 s); volume fill 87.6 s after ready.
- Later start from the volume: ready 53.3 s after container start (tar unpack 26.7 s for 7.4 GB, load 23.4 s from the
  volume, self-test 1.3 s); connect to first action 68.4 s from the laptop.

## Not covered

- The fast path runs on the RTX 4090 and the H100 (FAST in serve.py). The A40 (second pool) serves stock; adding it
  needs one boot there (and one cold compile).
- A new prompt captures (and, compiled, possibly recompiles nothing: the shapes are fixed at 256 tokens) a new graph:
  0.3 to 0.4 s for its first call.
- Batched observations (state (B, 14)) go to stock.
- More than 16 distinct (prompt, prefix length) pairs: the least recently used graph is dropped and recaptured when it
  comes back (0.3 to 0.4 s for the first call of a new pair).
- The ABC simulator side (abc_sim) and the YAM robot client are not part of this.

## Core support

What this family needed from the core is in it now: `hf://` and `https://` checkpoints in `stage.py fetch`, the family
bundle rules, the `license` and `hosted_id` spec fields, the policy's `path` and `path_reason` merged into the
metadata's `playground_serve` map, and the NOTICE entry for the self-test frames.

## Bring-up tool

`bringup.py` (this folder; `up FAMILY ENTRY` serves abc_vla, gr00t_n17 and xiaomi_r1) creates the volume and the
load-balancing endpoint as `pgs up` does (owner id in the names, workers min 0 max 1, idle 5 min, FlashBoot off).
`--dev` ships `devhook.py` so that a worker runs posted Python on its serving thread (bring-up only); `--only TYPE`
excludes the pool's other GPU types; `--no-volume --dc A,B` (or `all`) makes a one-off endpoint without a volume over
several data centers; `--max-usd` caps the hourly price (default 2); `--secret-env NAME=FILE` reads an env value such
as a token from a file at run time without printing it.

## GPU spend

Bring-up and measurements of abc_vla, gr00t_n17 and xiaomi_r1 on 2026-11-16: 112 worker minutes, about 4.14 USD at
Runpod's serverless list prices (an estimate from the workers' start and stop times; the volumes added under 0.01 USD).
The H100 runs (5.89 USD/hr) were about half of it.
