# Cosmos 3 policy family (cosmos3_policy)

**Status: experimental (owner decision, 2026-11-16), both entries.** `cosmos3_edge_droid` serves on an RTX 5090, an
A40 and an H100 and is verified against the official code; it serves the fast path (same model call, bit-identical
sample) after its boot self-test, else stock. Default GPU: the RTX 5090. `cosmos3_nano_droid` is built and
unit-tested, not run.

NVIDIA's Cosmos 3 world action models post-trained on DROID (Franka, three cameras). Code and weights: OpenMDW-1.1,
not gated, commercial use allowed. Code at `NVIDIA/cosmos-framework@1f173793`.

| Entry | Hosted id | Weights | Status |
|---|---|---|---|
| `cosmos3_edge_droid` | `cosmos3-edge-policy-droid` | `nvidia/Cosmos3-Edge-Policy-DROID@a74e848c`, 3.86B, 9.15 GB | experimental; served and verified on an RTX 5090, an A40 and an H100 |
| `cosmos3_nano_droid` | `cosmos3-nano-policy-droid` | `nvidia/Cosmos3-Nano-Policy-DROID@5e9dd43c`, 15.75B, 32.9 GB | experimental; built and unit-tested, **not run**: needs an 80 GB card (A100 2.99 USD/hr, H100 5.89); the owner kept it unrun |

Both return 32 x 8: absolute joint positions (7) and gripper for the next 32 steps at 15 Hz.

## How it serves

- `serve.py` builds the repo's `RobolabPolicyService` (the object `action_policy_server_robolab.py` hands to openpi's
  websocket server) with the docs' launch flags (Edge: `--format-prompt-as-json True --guidance-interval 960 1001`)
  and the pinned revision, and runs each request as the service's `infer` does on one rank: `_build_sample`,
  `_generate`, `_format_outputs`. Same 4 UniPC steps, guidance 3.0, shift 5.0, `resolution 480`.
- Keys are the RoboLab ones: `observation/{wrist_image_left,exterior_image_1_left,exterior_image_2_left}` (or one
  `observation/image` mosaic), `observation/joint_position` (7), `observation/gripper_position` (1, closed
  fraction), `prompt`. The metadata says `image_size` [360, 640]: the client's default jpeg mode then sends 360 x 640
  frames (16:9 cameras scale without bars) and the service's mosaic is exactly 540 x 640.
- Seeds: the official service draws a seed per request from an RNG seeded with 0. A request may carry `rs_seed`
  (int), as on the openpi worker, to fix the noise; without it the official RNG is used.
- The service downloads the checkpoint itself into `HF_HOME` on the volume (its `uv run ... hf download`; the hf CLI
  environment is prebuilt in `build_env.sh`).
- Environment (`build_env.sh`): the repo's `uv.lock`, `--extra train --group cu130-torch213-train` (torch 2.13
  cu130, Python 3.13). CUDA 13 wheels need driver >= 580; Runpod hosts ran 595.91 (CUDA 13.2), which works.

## Measured (A40, CA-MTL-1, 1.22 USD/hr; 2026-11-16)

All numbers measured unless marked estimate.

- First start on a new volume: worker listed 15:21:53, container started 15:25:55 (about 4 min of Runpod's image load
  before the container's first line), boot to ready 191.6 s: environment build 58 s (9.2 GB venv on a 96-CPU host),
  checkpoint download plus load 105 s, warm-up 27 s. `connect` to first action: 431 s.
- Later start from the volume: the environment tar (10.5 GB) unpacked in 105 s, slower than the 58 s build on that
  host; service built 122 s after unpack (checkpoint read from the volume and compile).
- GPU memory in use after warm-up: 9.2 GB (stock). An RTX 4090 (24 GB) would hold it (not tried: Ada is not in the
  model card's list).
- Stock `infer_ms` (served, through the gateway, raw mode): 850 to 961 ms; in-process 837 to 860 ms.
- Round trip from a US east coast client to CA-MTL-1, raw mode (three 180 x 320 frames): 1000 to 1250 ms.

## Verified against the official code

- In the worker (RS_VERIFY=1), on a recorded DROID observation (droid_obs.json, observation 0): the official
  `RobolabPolicyService.infer` with its own seeded RNG against this wrapper at the same seed: identical (max diff 0.0).
- Through the gateway (raw mode) at that seed: max diff 1e-6 from the official actions (the log rounds to 6 decimals).
- Same seed, same process: bit-identical every time. Same seed on another worker (another host and process):
  mean 0.0022 / max 0.016 (observation 0) and 0.0055 / 0.039 (observation 1). A new seed: mean 0.100 / max 0.458.

## Optimized path

Where the time goes (stock, A40, median of 3, 180 x 320 recorded frames): `_build_sample` 94 to 110 ms (CPU),
`_prepare_inference_data` 92 to 99 ms (first-frame VAE encode, tokenization), sampling 658 to 666 ms (4 UniPC steps
and one unconditional forward inside the guidance interval), `_format_outputs` 0.2 ms. The repo already compiles the
decoder blocks (torch.compile, dynamic shapes) and already encodes only the first frame.

Tried, in the worker, against stock at the same seed:

| Option (all official knobs) | infer ms | vs stock (mean / max) | Kept |
|---|---:|---|---|
| stock | 837 to 875 | reference | |
| `generate_samples_from_batch(use_batched_cfg=True)` | 1107 to 1150 | 0.0026 / 0.016 | no: slower on the A40 |
| `OmniSetupOverrides(use_cuda_graphs=True, compile_dynamic=False)` | fails | | no: "Cannot copy between CPU and CUDA tensors during CUDA graph capture" |
| `compile_dynamic=False` alone | 847 to 908 | 0.011 / 0.085 | no: no gain, larger numeric change |
| first-frame-only clip resize and pad (`fast.py`) | 755 (360 x 640 frames) | 0.0 / 0.0 | **yes** |

The fast path (`fast.py`): `_build_sample` makes a 33-frame clip whose frames after the first are zeros and its
transform pipeline pads (and for other sizes resizes) all 33 frames on the CPU. `ZeroTailFunctional` does those two
per-frame operations on the first frame only and fills the other frames with zeros, which is exactly what the
operations give for zero frames; any clip with a non-zero later frame goes to torchvision unchanged. Same model call,
same seed handling. Measured: the sample is equal bit for bit (checked at every start), the actions are identical at
the same seed in the same process (0.0 / 0.0), and with 360 x 640 frames (what the client sends for this model)
`_build_sample` drops from 76 to 8 ms and the call from 830 to 755 ms (median of 3, in-process). With 180 x 320
frames the mosaic resize (an official op, kept) dominates and the gain is 12 ms.

Self-test at every start (`serve.py load`): the fast sample of the stored observation must equal the stock sample
exactly, then two fast calls at seed 7 must match the stock actions stored for that GPU (A40, H100 and RTX 5090 stored) within
0.1 max / 0.02 mean.
Noise floor that sets these bounds: the same seed on other A40 workers moved stock actions by 0.004 to 0.0055 mean and
0.026 to 0.039 max; a new seed by 0.089 to 0.10 mean and 0.26 to 0.49 max. On any other GPU, with `RS_FAST=off`, or
when the check fails, the stock path serves and the metadata says why.

Served, through the gateway from a US east coast client, JPEG mode (360 x 640 frames), A40: the fast path 828 to
859 ms (median 839, 20 requests, boot self-test passed: sample equal, actions 0.0031 mean / 0.017 max from the stored
stock reference); stock 840 to 866 ms (median 853, 6 requests, another worker). That is 14 ms (1.6 %), not the 74 ms
of the in-process breakdown, and the two numbers come from different workers. On the H100 with 16 torch threads (one
worker, requests alternating): fast 235 ms median, stock 274 (below). **Decision: the fast path serves on the A40
and the H100** (`fast.FAST_GPUS`) after its self-test; on other GPUs, with `RS_FAST=off`, or when the check fails,
stock serves. The A40 numbers were measured before the thread cap; no A40 run since. RTX 5090 (2026-11-16,
below): the fast path serves there too.

| GPU (pool) | USD/hr | stock infer_ms (served, JPEG) | fast infer_ms (served, JPEG) | Measured |
|---|---:|---:|---:|---|
| A40 (AMPERE_48) | 1.22 | 840 to 866, median 853 | 828 to 859, median 839 (another worker) | yes |
| H100 SXM (ADA_80_PRO), 16 torch threads | 5.89 | 270 to 352, median 274.1 | 232 to 283, median **235.3** (same worker, alternating) | yes |
| H100 SXM, torch's default 112 threads | 5.89 | 269 to 336, median 308.5 | 299 to 362, median 341 (same worker, alternating) | yes |
| RTX 5090 (ADA_32_PRO), 16 torch threads | 1.89 | 444 to 701, median **514.0** (p90 643.3) | 405 to 531, median **439.5** (p90 502.6) (same worker, alternating, 20 + 20) | yes |
| RTX 5090, merged core (15 threads: the container's CPUs) | 1.89 | 482 to 509, median 498.9 | 445 to 454, median 451.7 (same worker, alternating, 10 + 10) | yes |
| L40S (ADA_48_PRO) | 1.89 | not measured | not measured | no worker allocated in 8 min (US-TX-3, stock LOW); endpoint and volume deleted, nothing billed |

RTX 5090 (EUR-NO-1, 2026-11-16; two verify runs on two workers, the second on the code merged with
the core): boot to ready 174 s on the new volume (environment build 66 s, checkpoint download and service build 106 s)
and 109 s from the volume (environment tar unpacked in 41 s). In the worker (180 x 320 recorded frames): stock 442 to
503 ms, fast 405 to 452 ms, the fast sample equal bit for bit and its actions identical to stock (0.0). The fast path
against the stored references: A40 0.0046 mean / 0.028 max, H100 0.0093 / 0.053, both inside the 0.1 / 0.02 bounds.
The stock actions at seed 7 were the same on both workers (16 and 15 torch threads; 0.0, also through the gateway),
and they are now stored as the RTX 5090 reference, so the fast path serves on the 5090 after its self-test. Served
through the gateway from a US east coast client, JPEG mode, requests alternating on one worker: in the table above. GPU
memory in use 9.6 GB of 32. A new noise seed moved the stock actions by 0.099 mean / 0.467 max.

H100 stage profile (in the worker, median of 3, ms; observation 0, 180 x 320 frames):

| Stage | stock, 112 threads (torch's default) | stock, 16 threads | fast, 16 threads |
|---|---:|---:|---:|
| `_build_sample` (CPU: compose, transform pipeline, tokenization) | 138 to 246 | 42.4 | 11.3 |
| `_prepare_inference_data` (first-frame VAE encode, text) | 64 to 100 | 62.1 | 63.3 |
| sampling (GPU, 4 steps and the guidance pass) | 116 to 150 | 148.2 | 146.4 |
| `_format_outputs` | 0.1 | 0.2 | 0.1 |
| total in the worker | 358 to 554 | 255 to 258 | 221 to 236 |

The H100 host has 224 cores and torch made 112 CPU threads; the CPU steps then took longer than the GPU's sampling,
and the first-frame-only resize and pad gave nothing (`_build_sample` 164 to 190 ms with it). `serve.py` now sets 16
threads at start (`RS_TORCH_THREADS`, never more than torch's default; "0" keeps it), before anything compiles: on
the same host, 32 threads gave 272.5 ms, 16 gave 271.9, 8 gave 294.0 (one process, threads changed between
measurements; changing them again made dynamo recompile until its cache limit stopped the process, so the count is
set once). Measured in a fresh process with 16 threads: the stock actions and the official service's actions equal
the 112-thread runs' bit for bit (0.0 mean and max), and the fast path gives the stock actions exactly (0.0). The
official knobs the hosted service names for this model gave nothing on the H100 (112 threads): `use_batched_cfg` 383
to 433 ms (0.009 mean / 0.048 max from stock) and `compiled_region="language"` 361 to 412 ms (0.012 / 0.073).

Not tried: the post-training recipe trains this model family in task mode `policy`, and the transform pipeline has a
`policy` sample mode that conditions on the observation image without the 32 future frames. The official RoboLab
server builds `wam` samples (video and actions generated together). Serving `policy` samples would skip most of the
generated tokens, but the maker documents only the `wam` server for these checkpoints, so it stays out.

Pools in the spec (2026-11-16): ADA_32_PRO (RTX 5090) first, the best served latency under 2 USD/hr (fast
439.5 ms median, 1.89 USD/hr); AMPERE_48 (A40) as the cheaper fallback (839 ms, 1.22 USD/hr). The L40S (not measured,
no worker allocated in 8 minutes) is out of the default pools. The H100 serves 235.3 ms at 5.89 USD/hr: `pgs up
cosmos3_edge_droid --gpu H100`. The model used 9.2 GB of GPU memory, so a 24 GB RTX 4090 would hold it (not tried: Ada
is not in the model card's list). Caveat: on 2026-11-16 the 5090 had Serverless stock only in Europe; a user in the
US gets a European worker, about 90 ms more round trip (estimate), still far below the 400 ms the 5090 saves.

## Recorded test data

`tests/families/data/droid_obs.json` (outside the worker bundle): two observations from `lerobot/droid_1.0.1` (Hugging Face dataset, Apache-2.0 per its card),
episode 1 ("Put the blue block in the green bowl"), frames 60 and 120: the three cameras as recorded (180 x 320, JPEG
q90), joint position, gripper position. `selftest_cosmos3_edge_droid.json` holds observation 0, its seed and the stock
actions the fast path must reproduce.

## Bring-up script

`bringup.py` (this folder; also serves `flex_pi` and `flux3_action`) creates the volume and the load-balancing
endpoint the way `launcher.py` does (stock image, RS_FILES, FlashBoot off, owner id in every name), rolls releases,
waits on the worker log, sends requests with the package client, and deletes by id. Learned on the way:

- After an env update, a worker of the old release kept serving: `update` sets workers max 0 until none is listed.
- Connecting through the gateway while no worker was ready made Runpod start up to three workers on a max-1
  endpoint (listed RUNNING, IDLE and INITIALIZING at once). `update --warm` keeps one worker (min 1), `wait` follows
  its log without touching the gateway, and only then does `act` connect; `down` sets max 0.

## GPU spend

The last bring-up day of cosmos3_policy, flex_pi and flux3_action (2026-11-16, from 20:00 UTC: two Cosmos verify runs
and `pgs up flex_pi_libero` on an RTX 5090): 15.6 worker minutes, about 0.50 USD at Runpod's serverless list prices
(an estimate from the worker list). The request-size probes used endpoints that never had a worker. GPU time before
that day's evening was not logged.

## Requests for the core

Status after the merge with the core (2026-11-16). Done:

1. Checkpoints that the model code downloads itself: `boot.sh` skips `stage.py fetch` (and so the checkpoint part of
   `stage.py fill`) when the spec's env has `RS_SELF_FETCH=1`, and sets no `RS_CKPT_DIR`; the three `serve.py` files
   no longer remove it (`tests/test_families.py::test_self_fetch_skips_the_checkpoint_download`).
2. Spec fields: `hosted_id`, `experimental` and `license` are core fields, and these specs pass them directly; the
   three experimental entries show "; experimental" in `pgs models`, and `pgs up` says so. Secrets (`HF_TOKEN` of
   `flux3_action_droid`) come from the user's machine at `pgs up` and are never saved.
3. Minimum CUDA per entry: `ProposedSpec.min_cuda` ("13.0" for both Cosmos entries); `pgs up` asks Runpod for hosts
   with at least that (`launcher.spec_cuda`), not the image's 12.8.
4. NOTICE: the DROID and LIBERO frames of these families, and Flex-pi's MIT code that `flex_pi/serve.py` adapts.
5. Bundle and Runpod's request limit: the core ships every file of the family folder except `spec.py`, `NOTES.md`,
   `__init__.py`, `bringup.py` and `devhook.py`, and files named `verify*` only for bring-up runs (env
   `RS_VERIFY=1`): `flex_pi/verify.py`, `flux3_action/verify_obs.json` and `cosmos3_policy/verify.py` (the verify code
   moved out of `serve.py`). `WORKER_FILES` / `VERIFY_FILES` are gone from the specs. Measured 2026-11-16 with dummy
   endpoints (workers max 0, deleted at once): Runpod accepted a 100,300-byte create request and refused a
   100,400-byte one (HTTP 413). The first `pgs up flex_pi_libero` after the merge, with every file shipped (env
   99,530 bytes), was refused in all three data centers it tried (the launcher deleted its empty volumes again).
   A test kept every entry's env under 99,000 bytes (env then: `cosmos3_edge_droid` 96,752, `flex_pi_libero` 92,950,
   `flux3_action_droid` 28,153); item 6 replaced it.
6. Done after the merge (core, 2026-11-16): RS_FILES is tar + xz (Python's `lzma` on both sides; the image's
   `python3` unpacks it), an entry ships only its own `selftest_<entry>.json`, and a family's Python files ship without
   comments and docstrings, line for line (`bundle.slim`). Largest request now: `cosmos3_edge_droid` 79,604 bytes
   (was 97,567), its verify run 81,438 (was 100,681), `flex_pi_libero` 74,508 (verify 78,438). `pgs up` measures its
   request before it creates anything and stops with a message over 100,300 bytes; `tests/test_families.py` keeps every
   entry and verify run 15 % under that. Storing the frames as binary saves almost nothing under xz (JPEG bytes do not
   compress; xz codes base32 text near 5 bits a character). Checked on Runpod with the new bundle (2026-11-16):
   `pgs up pi05_libero` (openpi bundle) ready after 1:44 on a new volume in US-NC-1, fast path, `pgs demo` 34 ms model
   time; `pgs up abc_vla_yam` (slim family bundle) ready after 8:16 with a cold compile cache, fast path self-test
   passed (0.015 max from stock), 30 ms model time.

Still open:

7. A worker whose health stayed 204 for about 9.5 minutes was restarted by Runpod (twice, flex_pi verify runs, 16:10
   and 16:20 to 16:29:44 UTC). A start that compiles for longer than that (flex_pi's loop-scope compile took 9 minutes
   on an A40) never becomes ready. Verify runs here mark health ready early; serving starts must stay under it.
8. Runpod listed more than `max` workers: two or three (RUNNING, IDLE, INITIALIZING) on max-1 endpoints, with and
   without gateway requests, on A40 and H100 endpoints, and on another project's endpoint; again at 20:16 on the RTX
   5090 endpoint (one THROTTLED, one INITIALIZING). A throttled worker has no GPU; the others may bill.
9. A new endpoint created with workers min 0 max 1 got a worker within a minute with no request (listed IDLE). The
   bring-up script creates endpoints with max 0 and raises it when a worker is wanted.
10. Measured: unpacking this family's 10.5 GB environment tar from a HIGH_PERFORMANCE volume took 94 to 105 s, while
    building it took 58 s on a 96-CPU host. A per-family switch to skip the tar would start this family faster.
