# MolmoAct2 family

Ai2's MolmoAct2 ([paper](https://arxiv.org/abs/2605.02881), [code](https://github.com/allenai/molmoact2)): a 5.4B
vision-language-action model, Molmo2-ER plus a flow-matching action expert that reads the VLM's key-value cache.

## License

- Code: Apache-2.0 (`LICENSE` at `allenai/molmoact2@6070080a`).
- Weights: the repo README (section 7) says "This model is licensed under Apache 2.0. It is intended for research and
  educational use in accordance with Ai2's Responsible Use Guidelines." The Hugging Face repos are not gated and carry
  no license tag. Self-hosting is allowed; the `pgs models` license note repeats the research-use statement.

## Entries

| Entry | Checkpoint (pinned revision) | Cameras (request keys, in order) | State | Actions per call |
|---|---|---|---:|---|
| `molmoact2_libero` | `allenai/MolmoAct2-LIBERO@0d24a92b` | `agentview_cam`, `wrist_cam` | 8 | 10 x 7, delta end-effector |
| `molmoact2_droid` | `allenai/MolmoAct2-DROID@d8c1abd8` | `external_cam`, `wrist_cam` | 8 | 15 x 8, absolute joints |
| `molmoact2_so101` | `allenai/MolmoAct2-SO100_101@152569fe` | `top_cam`, `side_cam` (any two views) | 6 | 30 x 6, absolute joints |
| `molmoact2_yam` | `allenai/MolmoAct2-BimanualYAM@8dcbed66` | `top_cam`, `left_cam`, `right_cam` | 14 | 30 x 14, absolute joints |

Every request also carries `state` (raw robot units) and `instruction` (the task). Each checkpoint is 21.78 GB of
float32 safetensors.

- Stock path: the repo's servers (`examples/droid/host_server_droid.py`, `examples/yam/host_server_yam.py`), bfloat16
  as their run scripts, CUDA graph off as their default. The model card's own example runs float32 with the action
  expert's CUDA graph on; its reported results are float32 (about 24 to 26 GB of GPU memory, so a 48 GB GPU).
- DROID: the server sends two images; the card's example sends three (the exterior camera twice); this follows the
  server. The training data has three cameras (`exterior_1_left`, `exterior_2_left`, `wrist_left`).
- Images: the processor resizes every frame to 378 x 378 (torchvision bilinear, no antialias, no padding), so the
  client sends frames at their own size (`image_size: "native"`), as JPEG q90 by default.

## Paths

- **stock**: the repo's server call (above). Checked against the official code on the same inputs and seeds (2026-11-16,
  RTX 4090, the model cards' sample inputs, seeds 0 to 2): equal bit for bit to the repo's own server class
  (`examples/droid/host_server_droid.py` for LIBERO and DROID, `examples/yam/host_server_yam.py` for YAM; SO-101 has no
  server and the DROID class takes an 8-value state only). Against the model card's bfloat16 snippet (`torch.autocast`,
  CUDA graph on): LIBERO 0.00035 mean / 0.0039 max, DROID 0.00094 / 0.0156, SO-101 0.0124 / 1.0, YAM 0.0012 / 0.035; a
  new noise draw: 0.0069 / 0.062, 0.059 / 0.25, 0.76 / 14, 0.039 / 1.3. The card's own processor call fails under the
  pinned transformers 4.57.1 (`extra_special_tokens` list); the servers' workaround is needed.
- **fast** (RS_FAST=auto, RTX 4090 only): the same model, bfloat16 weights, 10 flow steps and inputs; the action
  expert's own CUDA graph (`enable_cuda_graph=True`, the card's default) plus `torch.compile` (dynamic shapes, Inductor
  rounding as eager) of one function shared by the expert's 36 blocks, its final layer and its per-layer context
  projection; the image processor's resize and normalization on the GPU (the same torchvision bilinear resize and
  float32 arithmetic; `GpuImages`, checked against the official processor at every start: 0 of 1.29 million pixel
  values differed for YAM's frames); and CUDA graphs of the vision encoder (one input shape per entry) and of the
  language model's prefill (one per prompt length, at most 4 kept; `ShapeGraphs`). The last two left the self-test's
  actions unchanged to the last digit. Where the time went (LIBERO, stock 516 ms in-process): processor 9 ms on the CPU, vision 23 ms, language
  model 38 ms, context projection 12 ms, and the 10-step flow over 36 small blocks the rest (about 440 ms of kernel
  launches). The graph alone gives 151 ms, bit-identical; the compile 110 ms. Tried and left out: compiling the vision
  and language layers (no gain), static shapes (a new prompt length recompiled for 37 s), automatic dynamic shapes (44 s
  on the first new prompt), one compiled graph of all 36 blocks (cold compile 112 s, 47 s from the cache).
- Self-test at every start (`selftest_<entry>.json`): the card's sample frames shrunk to 128 or 160 px and encoded as
  JPEG q90, the card's state and task, seed 7, the stock path's actions, and its actions at seed 8 (a new noise draw).
  Fast must be within 0.1 of that draw's mean difference and 0.5 of its max, twice; else stock serves in the same
  process (and if that fails too, boot.sh starts serve.py again with RS_FAST=off). Checked: an altered reference
  (actions + 0.1) made the worker serve stock at 512 ms with the reason in its metadata.
- Compile time at start: 25 s with an empty cache, 8 s from the Inductor cache on the volume
  (`/runpod-volume/pgs/molmoact2/inductor/<GPU>`); a new prompt length later costs one call of about 0.5 s (the action
  graph is captured again).

## Measurements (RTX 4090, US-IL-1, 2026-11-16; measured unless marked)

| Entry | In-process stock / fast (ms) | Served `infer_ms` p50 stock / fast | Fast against stock, fixed seed (mean / max) | New noise draw |
|---|---|---|---|---|
| `molmoact2_libero` | 516 / 110 | 527.8 / 120.6 (40 calls; outliers 194 to 1940) | 0.00025 / 0.0029 served raw | 0.0095 / 0.062 |
| `molmoact2_droid` | 492 / 111 | 580.7 / 604 (one slow host, below) | 0.0003 / 0.0156 served raw | 0.060 / 0.19 |
| `molmoact2_so101` | 552 / 116 | 583.4 / 155.7 (20 calls; fast min 118) | 0.009 to 0.018 / 0.25 to 1.0 served raw | 1.30 / 14 |
| `molmoact2_yam` | 549 / 148 | 549.7 / 203.4 (20 calls; fast min 158) | 0.0004 / 0.0078 served raw | 0.020 / 0.34 |

- Server-side `infer_ms` on one RTX 4090 worker (the timer the gateway returns: JPEG decode, preprocessing and model;
  the card's sample as JPEG q90 at its own size, 40 calls after 3 warm-ups, threads capped at the container's 16 CPUs).
  Current fast path (2026-11-16 18:30 to 18:38 UTC) and, in brackets, the fast path before GPU preprocessing and the
  vision and prefill graphs (18:00 to 18:12, another worker):

  | Entry | Stock p50 / p90 (ms) | Fast p50 / p90 (ms) | [before] |
  |---|---|---|---|
  | `molmoact2_libero` | 520.7 / 552.4 | 104.2 / 107.3 | [114.5 / 119.5] |
  | `molmoact2_droid` | 532.0 / 568.9 | 104.8 / 110.0 | [113.0 / 117.2] |
  | `molmoact2_so101` | 565.5 / 661.1 | 111.8 / 117.4 | [117.9 / 131.3] |
  | `molmoact2_yam` | 574.8 / 601.0 | 139.3 / 141.7 | [159.7 / 169.9] |

- Where YAM's 141 ms go now (stage timers with a sync around each, in-process): JPEG decode 3.6, processor 3.0 (its
  image part 1.2 on the GPU), vision encoder 30.4, language-model prefill 49.2, context projection 3.5, action flow
  44.3. GPU kernel time is 122 of the 141 ms (13,478 kernels per call): the path is GPU-bound now. The vision encoder's
  attention runs in float32 by the checkpoint's config, and the prefill's matrix products run near the card's peak
  (about 750 tokens). What is left would take kernel work (fused flow blocks, a fused float32 attention), not serving
  changes.

  The same host earlier, at a load average of 86 and with torch's 64 threads: DROID fast 260.5 / 404.5, stock 797.2 /
  981.8. Served numbers through the gateway (table above) were taken during first starts, while the volume fill ran.
- JPEG q90 against raw frames, stock, LIBERO, fixed seed: 0.0007 / 0.0039.
- `molmoact2_libero` first start (stock): ready after 2:31 (environment build 44 s; checkpoint 21.78 GB at 181 MB/s in
  120 s; load 15 s; warm-up 3 s); fill 110 s. Later start from the volume: 70 s to ready (env tar unpack 21 s, weights
  load 44 s); with the fast path 1:29 (`pgs up` wall time 91 s, compile 9 s).
- `molmoact2_so101` first start with the fast path: ready after 2:03 (environment build 46 s; checkpoint at 310 MB/s;
  load 8 s; compile 28 s; self-test 0.044 mean / 1.0 max against bounds of 0.26 / 3.75). Later start (stock): 1:39.
- `molmoact2_yam` first start with the fast path: ready after 2:29 (checkpoint at 254 MB/s; compile 30 s; self-test
  0.00055 / 0.0078 against bounds of 0.028 / 0.87). Later start (stock): 1:21. Served `infer_ms` includes the JPEG
  decode and the processor's CPU resize of three 640 x 360 frames (about 50 ms more than in-process).
- `molmoact2_droid` first start (stock): 1:45. Its later start with the fast path landed on a slow host: env tar unpack
  149 to 172 s (43 MB/s, against about 350 MB/s for the LIBERO volume), weights load 50 to 131 s, compile 99 s, and its
  served calls took 604 ms (fast) though its self-test passed (0.000385 / 0.0078); the container restarted once and
  `pgs up` gave up at its 15-minute limit 38 s before the worker was ready. Cause not found (estimate: a slow or busy
  host); the same code served 110 to 120 ms on the dev worker and for LIBERO.
- Faster GPUs: `molmoact2_yam` served at 203 ms (p50) on the RTX 4090, at the 200 ms line, and `molmoact2_droid` was
  run for a like-for-like line against a hosted H100 service. One short run per GPU, first start with the fast path
  (RS_FAST=measure: the fast path on a GPU outside FAST_GPUS, still gated by the self-test), then stock, through the
  gateway, 30 calls (20 for the first YAM rows):

  | Entry | GPU (pool), data center | USD/hr | Stock `infer_ms` p50 / p90 | Fast `infer_ms` p50 / p90 | Fast against stock, fixed seed | Self-test |
  |---|---|---:|---|---|---|---|
  | `molmoact2_yam` | RTX 4090 (ADA_24), US-IL-1 | 1.34 | 549.7 / not kept | 203.4 / not kept (min 158) | 0.0004 / 0.0078 | passed |
  | `molmoact2_yam` | H100 80GB HBM3 (ADA_80_PRO), US-NE-1 | 5.89 | 384.5 / not kept | 99.8 / not kept (max 108) | 0.0003 / 0.0078 | passed |
  | `molmoact2_droid` | H100 80GB HBM3 (ADA_80_PRO), US-NE-1 | 5.89 | 403.3 / 459.7 | 85.1 / 97.1 | 0.0001 to 0.0003 / 0.002 to 0.0156 | passed |
  | `molmoact2_droid`, current fast path | H100 80GB HBM3 (ADA_80_PRO), US-NE-1 | 5.89 | 301.9 / 315.5 | 69.1 / 72.8 | 0.0001 to 0.0003 / 0.002 to 0.0156 | passed |

  The L40S and the RTX 5090 were not run (the 5090 has Serverless stock only in Europe). Pools stay RTX 4090 first,
  A40 second: the H100 halves the fast path's time for 4.4 times the price.
- A hosted service's `molmoact2-droid` (H100, TensorRT backend, three 180 x 320 cameras, chunk 15 x 8 at 15 Hz, step
  count not stated) reported a server model time of 78.0 ms p50 / 80.7 p90 (read from its own timing fields; what its
  timer covers is not documented). Ours on an H100, two cameras as the repo's server sends them, with JPEG decode and
  preprocessing inside the timer: 69.1 / 72.8 ms with the current fast path (85.1 / 97.1 before GPU preprocessing and
  the vision and prefill graphs). Same chunk; our flow steps are the checkpoint's 10.
- Before GPU preprocessing and the vision and prefill graphs, the RTX 4090 fast path spent about 9 ms in the processor
  on the CPU for two small frames; YAM's three 640 x 360 frames cost about 20 ms more than that.
- Precision: bfloat16, as the repo's servers and run scripts. The model card's reported results are float32 (24 to 26 GB
  of GPU memory: an A40 or L40S); not offered as an option here (owner's decision).

## GPU spend

Bring-up, measurements and checks on 2026-11-16 (with a `pi05_droid` first start, an `abc_vla_yam` check of the
launcher and a JPEG decode benchmark on a bare worker): about 184 worker minutes, about 5.93 USD at Runpod's serverless
list prices (an estimate from the workers' start and stop times; the volumes added under 0.05 USD), 2.57 USD of it on
H100s.
