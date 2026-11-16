# xiaomi_r1: Xiaomi-Robotics-1 (XR-1) on RoboCasa365

Entry `xiaomi_r1_robocasa365`: Qwen3-VL-4B with a 36-layer flow-matching DiT head, 5.05B bf16 parameters, 5 flow
steps, 16 x 12 actions at 20 Hz. Code `XiaomiRobotics/Xiaomi-Robotics-1@0dd7aef8`; weights
`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365@3a6d0293` (10.12 GB, 21 files). The model code itself comes with the
weights (`modeling_mibot.py`, `trust_remote_code`).

**Status: served and verified on a GPU, stock and fast path (RTX 4090 and H100, 2026-11-16).** Numbers are measured
unless they say estimate. Round 2 (closing the gap to a hosted service) rebuilt the fast path; its results are in
"Round 2" below, and the round 1 sections after it describe the earlier path (the CUDA graph of the DiT loop alone).

## License

Apache-2.0 for the code and the weights (the Hugging Face repo's license tag and LICENSE file), not gated.
Self-hosting is allowed. Recorded in `spec.LICENSE` and the metadata's `license`.

## What serves

- **stock**: the official inference split in two, run in one process: the evaluator's request building
  (`eval_robocasa365/entry.py` EvalClient: centre crop 0.95, the chat messages ending `/no_cot`, state padded to 60,
  `processor.apply_chat_template(..., do_resize=False)`; processor loaded with `use_fast=False` as `deploy/client.py`),
  then `deploy/server.py`'s `model(**data)` with flash-attention 2 in bf16, then `processor.decode_action`.
- **fast** (`xr1_fast.py`, round 2): the request built with its per-instruction parts cached and its pixels made on
  the GPU (bit-identical to the processor's), and the whole forward (vision tower, prefill, DiT loop) as one CUDA
  graph per prompt, with the DiT step compiled by inductor (emulated precision casts). Variants tried in order at
  every start: the compiled step in the graph, then the graph alone (bit-identical), then stock.
- Flow noise: the model draws it from torch's default stream, seeded once at import (7 + the CUDA_VISIBLE_DEVICES
  index), as the official server. An optional request key `rs_seed` gives a call its own seed (fork_rng) and leaves
  the stream alone; the checks below use it. Without it, two calls on the same observation differ (by 0.049 mean, 2.0
  max: the gripper and mode columns are -1/+1).
- Self-test at every start (`selftest_xiaomi_r1_robocasa365.json`): an episode's first tick (one frame per camera as
  JPEG q75, repeated over the history, and one state) with seeds 7 and 8. Stock must match the stored actions (warning
  only, max 0.1) and fast must match live stock within max 0.1 and mean 0.01 (the pi0.5 rule; round 1 used 0.01 and
  0.001 for the bit-identical path), twice; else the next variant, and at last stock.
- Request keys: the three `video.robot0_*` cameras, each the 4 frames of ticks t-6, t-4, t-2, t at 256 x 256
  (`spec.sample_history` builds them from the robot's last 7 frames, as entry.py); `state` (4, 14) at the same ticks;
  `instruction`. A camera is a (4, 256, 256, 3) array (raw), a list of 4 frames, or a map {"0".."3"} of frames
  (`spec.as_frames_map`): only the map form gets JPEG in the client's jpeg mode, since the codec encodes H x W x 3
  arrays inside maps.

## Round 2: the gap to a hosted service (2026-11-16)

A hosted service measured on the same model (H100 in us-west-2; the logs are kept outside this repository):
server inference 35.6 ms p50 (35.5 ms when every frame of every request was new, so no cache of earlier frames), its
preprocess field 0.0 ms, 16 x 12 actions, 5 flow steps, bf16, the 4-frame history of all three cameras sent with every
request (12 frames; no history kept on its server). Its timer's scope is not documented: it may or may not include
image decode and the processor.

### Where our old 121.4 ms (H100, round 1) went

- GPU class: the same round 1 path took 148.6 ms on the RTX 4090; about 27 ms of the gap was the 4090 itself.
- Span: on the H100 host of round 1, the 12-frame JPEG decode and the server took about 14 ms and building the request
  on the CPU (12 PIL centre crops, the slow video processor, the chat template and tokenizer) about 17 ms; work that
  may sit outside the hosted service's timer.
- Real: the model call itself, about 90 ms against 35.6: the Hugging Face prefill's host synchronizations (199
  `.item()` and 285 stream synchronizations per call), the vocabulary logits of all 476 tokens that nothing reads,
  the vision tower's per-block length reads, and the DiT loop's 9,000 small kernels (round 1 had put only that loop in
  a graph, 35.6 ms on the H100).

### What changed

1. The whole forward as one CUDA graph per prompt (`FastForward`): the per-prompt values (rope positions, rotary
   tables, causal mask, flash-attention lengths, vision position embeddings and sequence lengths, placeholder
   indices, DiT masks and rotary tables) computed once with the model's own functions; the unused final norm and
   vocabulary logits dropped; the vision attention given its maximum length as an int. Bit-identical to stock.
2. The request built once per instruction for its text parts, and its pixels on the GPU (`FastRequest`,
   `GpuCenterCrop`: Pillow's fixed-point bilinear arithmetic, bit-identical, checked against the processor at the
   first call of every instruction).
3. The DiT step compiled by inductor inside the graph, with emulated precision casts (it rounds between fused ops as
   eager does; that halved the differences). Not bit-identical: checked by the self-test, with the graph alone as
   the fallback.
4. Inductor's cache on the volume, per GPU type (`RS_COMPILE_CACHE`): only a GPU type's first start compiles.

### Stage table (ms, p50; "served" is the server's infer_ms, the rest from the reply's policy_timing or graph replays)

| | RTX 4090 stock | RTX 4090 round 1 | RTX 4090 round 2 | H100 stock | H100 round 1 | H100 round 2 |
|---|---:|---:|---:|---:|---:|---:|
| JPEG decode of 12 frames + server | 12.0 | 11.7 | 6.7 | 6.4 | 15.0 | 11.3 |
| Request (crops, processor, tokens) | 16.6 | 16.6 | 1.3 to 1.5 | 24.1 | | 3.3 |
| Vision tower | 17.2 | 17.2 | 10.6 | | | 5.1 |
| Text prefill (with the unused logits in stock) | 53.8 | 53.8 | 31.4 | | | 14.7 |
| DiT, 5 steps | 154.2 | 32.0 | 16.4 | | 35.6 | 12.2 |
| Model call (vision + prefill + DiT) | 225.2 | 120 (est.) | 58.6 to 58.7 | 146.4 | | 32.8 |
| **Served infer_ms p50 / p90** | **267.9 / 341.4** | **148.6 / 226.7** | **65.5 / 71.3** | **177.0 / 188.1** | **121.4 / 151.5** | **47.6 / 55.7** |
| Without decode and server (request + model) | 255.9 | 136.9 | 60.1 | 170.5 | 106.4 | 36.1 |
| Model only | 225.2 | 120 (est.) | 58.7 | 146.4 | | 32.8 |

Sources: served rows through the gateway (20 calls); "without decode and server" in-process (round 1 columns) or
from the reply's policy_timing (round 2 and H100 stock); the decode row is served minus that. Stage rows: round 1's
eager breakdown on the 4090 (stock) and graph replays of each stage (round 2). JPEG decode alone of the 12 frames:
5.8 ms on the 4090 host, 4.3 ms on the H100 host (isolated).

Round 2 in-process on the H100: 33.6 ms per call (request 1.1, graph 32.0: vision 5.1, prefill 14.7, DiT 12.2). Stock
ran on two H100 hosts: 262.5 ms served (48.9 ms of request) on an AP-IN-1 host earlier that day, 177.0 ms on the host
of the final run, which the table gives. Like for like with the hosted 35.6 ms: our model call 32.8 ms, with the
request 36.1 ms, with decode and server 47.6 ms.

### Accuracy

- RTX 4090, served round 2 against served stock (same frames, same seeds, 3 observations): 0.0006 to 0.0008 mean,
  0.004 to 0.008 max. Boot self-test against live stock: 0.0015 mean, 0.023 max (seed 7), 0.0005 / 0.008 (seed 8).
- H100 boot self-test: 0.0014 mean / 0.070 max (seed 7), 0.0005 / 0.008 (seed 8). Without emulated precision casts the
  compiled step moved seed 7 by 0.45 max on the H100 and failed the self-test (the worker then served the graph alone,
  as designed); the H100's own stock differs from the RTX 4090's stored stock actions on that seed by 0.21 max.
- For scale: another seed moves the actions by 0.05 to 0.19 mean and about 2.0 max.
- The graph alone (variant 2) is bit-identical to stock on both GPUs (0.0 on 4 observations and two unseeded calls).

### Start times

- RTX 4090, later start (environment, weights and inductor cache on the volume): ready 30.7 s after container start,
  the compiled variant built in 8.7 s; connect to first action 65.7 s from the laptop. The first start of a GPU type
  compiles: 45.5 s on the 4090 (ready 101.9 s). With emulated casts a new compile was needed once more (51 s).
- H100, no volume (one-off runs): ready 124.9 s after container start, of which the weights download 87.7 s at 115
  MB/s and the compile 30.6 s.

### What is left (no large item)

- The text prefill (31.4 ms on the 4090, 14.7 on the H100) is matrix arithmetic: 3.5 TFLOP for 476 tokens; on the
  4090 that is 21 ms at the card's peak (estimate). Compiling its layers would fuse the norms and rotary products for
  a few ms (estimate), at more compile time and more numerical difference; not done.
- The DiT (16.4 / 12.2 ms) is about 5 x 36 layers over 21 tokens: launch and latency bound even compiled.
- JPEG decode and the server (6.7 ms on the 4090 host, 11.3 on the H100 host): `pgs_server` decodes the 12 frames one
  after another with PIL before the policy sees them (shared code; request for the core below).
- The fast path's FAST list still includes the RTX 5090 and the L40S from round 1; round 2 was measured on the RTX
  4090 and the H100 only, so on the 5090 and L40S the self-test is the only check.
- The pool order in `spec.py` after the core's merge: RTX 4090, RTX 5090, A40.

## Round 1 measurements (RTX 4090, US-IL-1, 2026-11-16)

| What | Stock | Fast |
|---|---:|---:|
| Served `infer_ms` p50 / p90 (gateway, 20 calls, 12 JPEG frames decoded in the call) | 267.9 / 341.4 | 148.6 / 226.7 |
| In-process per call p50 / p90 | 255.9 / 259.0 | 136.9 / 161.8 |
| Round trip from the US east coast, frames map in jpeg mode (143 kB), p50 | 385 | 257 |
| Round trip with raw stacks (2.36 MB per request), median of 5 | 951 | 742 |
| GPU memory (nvidia-smi after warm-up) | 10.6 GB | 10.8 GB |

### Faster GPUs (2026-11-16, one short run each, same self-test)

Stock served `infer_ms` stayed above 200 ms on the RTX 4090, so the model ran once on each faster GPU with Serverless
stock. Each run: a fresh worker (no volume, except the 4090), stock served through the gateway, the shipped boot
self-test (fast path against live stock, both seeds), fast served through the gateway with the same requests (20
calls, frames map in jpeg mode). Prices are Runpod list prices read right before each create. All numbers measured.

| GPU (pool) | Data center | USD/hr | Stock `infer_ms` p50 / p90 | Fast `infer_ms` p50 / p90 | Fast in-process p50 | DiT loop replay | Self-test fast vs stock |
|---|---|---:|---:|---:|---:|---:|---|
| RTX 4090 (ADA_24) | US-IL-1 | 1.34 | 267.9 / 341.4 | 148.6 / 226.7 | 136.9 | 32.0 ms | 0.0 |
| RTX 5090 (ADA_32_PRO) | EUR-IS-2 | 1.89 | 223.6 / 294.8 | 125.6 / 162.0 | 102.4 | 28.8 ms | 0.0 |
| H100 80GB HBM3 (ADA_80_PRO) | EU-FR-1 | 5.89 | 212.2 / 260.7 | 121.4 / 151.5 | 106.4 | 35.6 ms | 0.0 |
| L40S (ADA_48_PRO) | US-MO-1 | 1.89 | 883.5 / 1245.9 | 623.1 / 1182.7 | 335.6 | 32.7 ms | 0.0 |

- Served fast actions equal served stock actions (0.0) on every GPU, for JPEG frames and raw stacks.
- Stock on another GPU against the RTX 4090's stored stock actions (warning only in the self-test): 5090 0.0007 to
  0.0009 mean, 0.008 max; H100 0.003 and 0.0009 mean, 0.21 and 0.008 max; L40S 0.019 and 0.0009 mean, 1.60 and 0.008
  max (one gripper column flipping sign on one seed). Rounding differences between GPUs, not path differences.
- What the numbers say: the faster GPUs save little. After the fast path the call is bound by the host CPU (the
  processor, the Hugging Face prefill's host synchronizations, JPEG decoding) and by the launch-bound DiT loop, not by
  GPU arithmetic: the loop replays in 29 to 36 ms on all four. The L40S host (US-MO-1) was slow on the CPU side: its
  loop replay matched the others, the rest was 3 to 4 times slower, and its cold start took 314 s.
- Pools in `spec.py`: RTX 5090 first (best latency at 1.89 USD/hr), RTX 4090 second (15 % slower, 29 % cheaper), A40
  last (stock path, not measured for this model). The H100 is 3 % faster than the 5090 at 3.1 times the price; the
  L40S was the slowest. Caveat for the owner: at the time of the runs the 5090 had Serverless stock only in Europe
  (EUR-IS-2 without network volumes, EUR-NO-1, EU-RO-1, EU-CZ-1), and `pgs up` takes the first pool with stock, so a
  user in the US would get a European worker and about 90 ms more round trip (estimate), more than the 23 ms the 5090
  saves. Putting ADA_24 first instead is the other sensible order.
- The fast path is enabled (FAST in serve.py) on the four GPUs above; every start still runs the self-test.

Stock breakdown in-process (median of 10): processor on the CPU 21.6 ms, `model(**data)` 225.2 ms, of which the VLM
prefill (476 tokens) 71.0 ms and the 5-step DiT loop the rest (about 154 ms). The captured loop replays in 32.0 ms.

Action checks:
- Playground Serve's stock path against the official server and evaluator client (`deploy/server.py`'s Server with the
  same model object on a localhost socket, `eval_robocasa365/entry.py`'s EvalClient, seeded the same way): **0.0
  difference** on 4 observations (1, 4 and 7 ticks of history, two instructions).
- Fast against stock at a fixed seed: **0.0** on the same 4 observations in-process; continuing the unseeded stream
  from the same seed: 0.0 for two calls in a row; through the gateway, served fast against served stock: 0.0 on 3
  observations, both with JPEG frames and raw stacks.
- For scale: another seed moves the actions by 0.051 to 0.188 mean and 1.92 to 2.01 max. JPEG q90 frames (the jpeg
  mode) against raw frames at the same seed: 0.0009 to 0.0012 mean, 0.008 to 0.014 max.

Measured and not used: torch.compile (inductor, default mode) of the 5-step loop under the CUDA graph: 17.0 ms instead
of 32.0, actions 0.0014 to 0.0017 mean and 0.0078 max from stock, and 173.5 s of compiling at the first call of every
cold start. Not bit-identical, and slow to start.

Not tried: capturing the VLM prefill as a graph. Profiled on the RTX 4090: 67.6 ms per prefill (vision tower 17.2 ms),
with 199 `.item()` calls, 46 `nonzero` and 285 stream synchronizations per call (rope index on the CPU, boolean-mask
writes of the deepstack features, flash-attention varlen lengths). The arithmetic is about 45 ms (estimate: 3.4 TFLOP
for 476 tokens plus the measured vision tower), so a rewritten, captured forward could save about 20 ms (estimate);
not done.

Start times:
- First start on an empty volume: ready 101.2 s after container start (environment build 49 s, 7.4 GB venv; weights
  42.6 s at 238 MB/s from Hugging Face; load 6.8 s); volume fill 110 s after ready.
- Later start from the volume: ready 60.0 s after container start (tar unpack 22.5 s for 8.2 GB, load 33.5 s, self-test
  3.0 s); connect to first action 83.3 s from the laptop.

## Not covered

- The fast path runs on the RTX 4090, RTX 5090, H100 80GB HBM3 and L40S (FAST in serve.py); the A40 serves stock.
- The RoboCasa365 simulator and the robot-side history buffer are the user's (spec.sample_history is the helper).
- Self-test frames are real camera frames of another robot (the ABC repo's video), not RoboCasa renders.

## Core support

What this family needed from the core is in it now: `hf://` checkpoints, JPEG frame stacks (`jpeg_stack`), the
`History` helper, threaded JPEG decoding, the RTX 5090's pool and price, the release check's exceptions for
`robocasa365` and `cu12torch2`, and the family bundle rules.

## GPU spend

Counted with abc_vla's and gr00t_n17's in `abc_vla/NOTES.md`.
