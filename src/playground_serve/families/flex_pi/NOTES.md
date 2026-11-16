# Flex-pi family (flex_pi)

Flex-pi (UW with Ai2, arXiv 2608.10860): a Wan2.2-TI2V-5B video expert and an action expert in one
Mixture-of-Transformers, co-denoising video, DINOv3 and pointmap streams with 32 actions (4 flow steps).

- License: MIT for code (`geyan21/flex-pi@b27019ec`) and weights (`flex-pi/flexpi-libero@f853cb49`, not gated). The
  vision encoder it loads (`timm/vit_base_patch16_dinov3.lvd1689m`, not gated) is under the DINOv3 License; the VAE and
  T5 (Wan2.2) come from ModelScope (`DiffSynth-Studio/Wan-Series-Converted-Safetensors`, `Wan-AI/Wan2.1-T2V-1.3B`
  tokenizer). Self-hosting is allowed.
- Entry `flex_pi_libero` (hosted id `flexpi-libero`): 32 x 7 delta end-effector actions at 20 Hz; the evaluator runs 10
  and replans.

## Regime: action-only by default

A hosted service measured on this model (H100, PyTorch backend, no CUDA graphs) answered in 64 ms of server
inference at the median; our first numbers were 1152 ms stock and 906 ms optimized on an A40. The GPU explains about
3.4 times (full joint on an H100: 341 to 347 ms stock, measured below). The rest is the regime: the evaluator's default
is full joint, which generates the video, DINO and pointmap futures together with the actions. The checkpoint's
card publishes LIBERO success for both regimes of this "stream dropout" checkpoint (action-only 98.4 %, full joint
98.5 %), says it "serves both the fast action-only path and full joint generation", and gives the action-only
evaluation as "the same command plus three flags" (`INFER_JOINT_VIDEO/DINO/POINTMAP=false`); the README calls it
"the fast path" (60 ms on an RTX 5090 compiled). The hosted service's observation contract takes no depth (agentview
and wrist at 256 px, 8-D state, prompt; its first attempt failed on a 224 px image, not on depth).

So `RS_FLEXPI_JOINT` now defaults to "false": the official action-only regime, same weights, 4 steps, seed and
inputs; full joint stays available with "true". `verify.py` checks either regime against EVAL's own
`_predict_action_chunk` with that regime's flags.

**Default and full joint.** `pgs up flex_pi_libero` serves the official action-only regime. For the evaluator's full
joint regime (video, DINO and pointmap futures generated with the actions; 236 ms fast / 341 stock on an H100, 903 /
1162 on an A40), set `"RS_FLEXPI_JOINT": "true"` in the entry's `env` in `families/flex_pi/spec.py` and run `pgs up
flex_pi_libero` again: the changed env is a new release, and its workers start in that regime. The fast path has
full-joint references for the A40 and the H100 only, so on an RTX 5090 full joint serves the stock path. Without depth the call passes `present_pointmap=False` (regime (b)
of the repo's regime table, "no depth, action only"; this checkpoint was trained with every presence dropped at
p = 0.5, but no LIBERO success rate without depth is published).

## How it serves

- There is no LIBERO server upstream. `serve.py` follows the evaluator, `experiments/libero/eval_libero_single.py`
  (EVAL): start-up as `eval_single_process`, a call as `_predict_action_chunk` on the T-shape float path, with
  `sim_libero.yaml`'s values (4 steps, seed 42 on every call, CPU noise, stretch resize) and the launcher's full-joint
  regime. EVAL imports the simulator at module level, so its tensor helpers are copied into `serve.py`, cited.
- Request keys are openpi's LIBERO keys, so a `pi05_libero` client works: `observation/image`,
  `observation/wrist_image` (upright, as openpi's LIBERO example sends them), `observation/state` (8: eef position,
  axis-angle, gripper qpos), `prompt`; optional `observation/image_depth` and `observation/wrist_image_depth` (uint16
  mm) turn the pointmap stream on, as in the published evaluation. The metadata's `image_size` is [256, 256] (the
  evaluator's render size).
- Without depth the call passes `present_pointmap=False`. This checkpoint was trained with stream dropout, but no
  success rate without depth is published, and a client that sends no depth gets that untested regime.

## Weights at first start

- `flex-pi/flexpi-libero` (12.08 GB torch pickle, config, dataset stats) and the DINOv3 encoder come from Hugging Face
  into `HF_HOME` on the volume; the LIBERO camera intrinsics from the `flex-pi/libero_mujoco3.3.2_depth` dataset.
- The VAE (1.41 GB), the umT5 encoder (11.36 GB) and its tokenizer come from ModelScope only. The model's loader
  fetches them one stream per file: measured 1.3 to 1.6 MB/s from CA-MTL-1 (the encoder alone would take about two
  hours). `prefetch.py` fetches the same files to the same paths first, in 32 parallel byte ranges (the openpi worker's
  `stage.py` range fetch), checks size and SHA-256 against a listing read from ModelScope's API on 2026-11-16 (which
  also pins the content: the loader takes ModelScope's latest revision), then renames them into place. Measured: VAE
  8.3 s (170 MB/s) plus 3.4 s check, encoder 121.5 s (94 MB/s, written to the network volume) plus 26.1 s check, all
  six files 167.5 s. Later starts find them on the volume.

## Verified against the official code (A40, CA-MTL-1)

`verify.py` (RS_VERIFY=1) imports `experiments/libero/eval_libero_single.py` from the checkout with the simulator
stubbed and replaces only its three simulator reads (`get_libero_image`, `_extract_sim_state`,
`get_libero_per_cam_depth_uint16_mm`) by the client's arrays; everything else is EVAL's own code: the hydra config
(`sim_libero.yaml` plus the launcher's regime overrides, then `_maybe_swap_eval_cfg_with_saved`),
`_obs_to_model_input`, the `infer_action` keywords, `_denormalize_action` and the gripper post-processing. On the
recorded LIBERO observation (`selftest_flex_pi_libero.json`), on two worker starts:

- EVAL's config after the saved-config swap equals the one `serve.py` builds (model and processor): yes.
- EVAL's `_predict_action_chunk` against `serve.py`'s `Policy.infer` on the same model: identical (0.0 mean, 0.0
  max), with depth and without.
- Stock (EVAL's setting, eager): 1145 to 1505 ms per call in the worker, full joint regime.

- Noise floor: stock is bit-identical across three worker starts (seed 42 on every call, as EVAL). A new noise seed
  (43) moves the actions by 0.007 mean and 0.060 max (with depth).

## Optimized path

The repo's own `prepare_for_inference` options, measured in the worker against stock (A40, full joint regime):

| Option (official) | ms per call | vs stock (mean / max) | Kept |
|---|---:|---|---|
| stock (EVAL: eager) | 1145 to 1505 (served median 1152 to 1162) | reference | |
| `torch_compile=True`, `reduce-overhead`, step scope (the RoboTwin launchers' default) | 896 to 936 (served median 903 to 906) | 0.0006 to 0.0008 / 0.005 to 0.0075 | **yes** |
| + `attn_backend="auto"` | 910 to 921 | 0.0007 / 0.0075 | no: no gain on the A40 |
| `torch_compile_scope="loop"` | 900 to 944 | 0.0007 / 0.0075 | no: no gain, 6 to 12 min compile |
| + `glue_cache=True, encoder_cuda_graph=True` | fails | | no: "accessing tensor output of CUDAGraphs that has been overwritten" |

The fast path (`fast.py`) is the first row's options on the same model; same 4 steps, seed 42, inputs and post
processing. Self-test at every start: the stored observation, with and without depth, twice each, must match the
stock actions stored for this GPU within 0.03 max / 0.003 mean (measured fast vs stock: 0.0075 / 0.0008; a new seed:
0.060 / 0.007). Its cost: compiling both regimes takes about 190 s at start (100 s with depth, 90 s more without).
Served A/B on one A40 worker (requests alternating stock and fast, `rs_path`, verify runs only), recorded LIBERO
observation with and without depth:

| Mode | stock infer_ms (median) | fast infer_ms (median) | fast vs stock actions (max) |
|---|---:|---:|---:|
| raw | 1151.5 | 906.0 | 0.0075 |
| jpeg (client default, 256 x 256 q90) | 1162.0 | 903.2 | 0.0075 |

## Action-only: stage profile, optimized path and GPUs

Stage profile (`verify.py stage_profile`: each model stage CUDA-synchronized, median of 3 calls, ms; recorded LIBERO
observation, action-only regime, H100 SXM, measured 2026-11-16):

| Stage | stock, no depth | fast (step scope), no depth | fast (loop scope), no depth | fast (step), depth | fast (loop), depth |
|---|---:|---:|---:|---:|---:|
| CPU input tensors | 5.9 | 2.9 | 3.4 | 5.0 | 4.1 |
| text encoder (umT5) | 29.5 | 0 (cached per prompt) | 0 | 0 | 0 |
| VAE first frame | 9.0 | 6.2 | 6.3 | 6.3 | 6.3 |
| DINOv3 first frame | 24.1 | 4.8 | 4.9 | 4.9 | 4.9 |
| pointmap (depth) | | | | 6.7 | 6.6 |
| video-expert prefill | 46.3 | 21.0 | 23.5 (prefill and loop) | 23.2 | 26.5 (prefill and loop) |
| 4 denoise steps | 173.4 | 17.7 | (above) | 17.3 | (above) |
| glue (masks, noise, merges) | 19.4 | 10.4 | 11.1 | 11.7 | 11.1 |
| post-processing | 0.5 | 0.2 | 0.2 | 0.2 | 0.2 |
| total | 322 | 63.5 | 49.6 | 75.4 | 59.7 |

What each change is (all the repo's own `prepare_for_inference` options, documented there as the same math, plus
one cached input):
- `torch_compile=True`, `reduce-overhead`: each denoise step (and the prefill) as a CUDA graph; the eager steps were
  launch-bound (43 ms each).
- `attn_backend="auto"`: the action path's all-True attention masks are dropped, so SDPA can run flash.
- `glue_cache=True`: per-call constants (RoPE freqs, masks) memoized ("bit-identical to a rebuild").
- `compile_encoders=True`: the VAE, DINOv3 and pointmap first-frame encoders compiled.
- The prompt's text context, encoded once by the model's own `encode_prompt` and passed as `infer_action`'s
  `context`/`context_mask` (its documented alternative to `prompt`; the repo's latency benchmark uses "precomputed
  text context"). Same tensors.
- Scope "loop" (`torch_compile_scope="loop"`: prefill and the 4 steps in one graph) is 10 to 14 ms faster still but
  compiled for 2.5 + 2.5 minutes on the H100 and 3 + 6 on the A40 at start, against the ~9.5 minutes after which
  Runpod restarted a worker that was not healthy. The served fast path uses step scope; loop stays measured, not
  default.

Accuracy: fast vs stock at the same seed 0.0005 to 0.0008 mean, 0.004 to 0.0075 max on every GPU; a new noise seed
moves stock by 0.007 mean, 0.047 to 0.071 max. Self-test at every start: the stored observation with and without
depth, twice each, within 0.03 max / 0.003 mean of the stock actions stored for that GPU and regime; stored references
for the A40 and the H100 (both regimes); the A40 and H100 references also pass on the other card. Elsewhere, or on
failure, stock serves.

Served (through the gateway from a US east coast client, JPEG mode = the client's default, 256 x 256 q90; requests
alternating stock and fast on one worker, `rs_path`, verify runs only; median of 8 per row; "fast" here is the variant
the verify run picked: loop scope on the H100 and A40 runs). Reference: a hosted service, 65.3 ms p50 server time on
an H100, action-only, no depth.

| GPU (pool) | USD/hr | regime | observation | stock infer_ms | fast infer_ms |
|---|---:|---|---|---:|---:|
| H100 SXM (ADA_80_PRO) | 5.89 | action-only | no depth | 152.4 | **65.4** |
| H100 SXM (ADA_80_PRO) | 5.89 | action-only | depth | 164.9 | 77.2 |
| A40 (AMPERE_48) | 1.22 | action-only | no depth | 318.5 | 131.9 |
| A40 (AMPERE_48) | 1.22 | action-only | depth | 411.1 | 187.4 |
| RTX 5090 (ADA_32_PRO) | 1.89 | action-only | no depth | 310.6 | **125.2** |
| RTX 5090 (ADA_32_PRO) | 1.89 | action-only | depth | 321.5 | 136.8 |
| RTX 5090, before the thread cap | 1.89 | action-only | both, raw | 450 to 495 | 256 to 272 |
| H100 SXM | 5.89 | full joint | both | 341 | 236 |
| A40 | 1.22 | full joint | both | 1162 | 903 |
| RTX 4090 (ADA_24) | 1.34 | | | does not fit | |

In-process (worker, no gateway), action-only, no depth: H100 stock 312 to 322 ms, fast 64.1 to 65.0 (step) and 49.6
(loop, profiled); A40 stock 301 to 314, fast 133 (step) and 124 (loop).

- RTX 4090: not run. The worker used 25.4 to 26.4 GB of GPU memory (H100, A40), over the card's 24 GB; the repo's
  INSTALL says one 32 GB GPU.
- RTX 5090 and CPU threads: torch sizes its CPU thread pool by the host's cores, not by what the container gets. On
  the 5090 host in EUR-NO-1 (120 cores, no CPU quota, 120 torch threads) the CPU input step took 124 to 590 ms (3 to
  6 ms on the A40 and H100 hosts) and served times swung by two times between requests. `serve.py` now sets 16
  threads at start (`RS_TORCH_THREADS`, at most the CPU quota and torch's default; "0" keeps torch's default), before
  anything compiles. Measured on another 5090 host of the same pool (120 cores) with 16 threads: the stock actions
  equal the 120-thread runs' bit for bit (0.0 mean and max, with and without depth), the input step 3.0 to 4.2 ms,
  stock 319 to 340 ms in the worker (482 to 620 before), the fast path 97 to 108 ms without depth (224 to 349 before).
  The fast path passed its self-test there against its own stock (0.0006 / 0.0075), the A40 reference (0.0008 /
  0.0062) and the H100 reference (0.0008 / 0.005); the 5090's action-only stock reference is now stored (no
  full-joint one: with `RS_FLEXPI_JOINT=true` a 5090 serves stock).
- Stage profile on the RTX 5090 (16 threads, median of 3, ms; the "before" row is the coarse split of the 120-thread
  run, no depth):

| Stage | stock, no depth | fast (step), no depth | fast (step), depth |
|---|---:|---:|---:|
| CPU input tensors | 3.0 | 3.2 | 3.6 |
| text encoder (umT5) | 34.7 | 0 (cached) | 0 |
| VAE first frame | 10.3 | 8.8 | 9.4 |
| DINOv3 first frame | 29.6 | 9.0 | 9.8 |
| pointmap (depth) | | | 9.8 |
| video-expert prefill | 54.6 | 46.6 | 51.0 |
| 4 denoise steps | 230.3 | 19.2 | 21.6 |
| glue | 14.3 | 12.1 | 14.6 |
| post-processing | 0.7 | 0.3 | 0.4 |
| total | 380 | 99.6 | 120.6 |
| before (120 threads): input step / model call | 124 to 589 / 353 to 555 | | |

  On the 5090 the video-expert prefill (46.6 ms, compiled) is the largest stage left; on the H100 it is 21 ms.
- Pools in the spec (2026-11-16): AMPERE_48 (A40) first (131.9 ms, 187.4 with depth,
  1.22 USD/hr), ADA_32_PRO (RTX 5090) second (125.2 ms, 136.8 with depth, 1.89 USD/hr). The 5090 is 6.7 ms faster
  without depth, but on 2026-11-16 it had Serverless stock only in Europe (EUR-NO-1, EUR-IS-2): a user in the US got a
  European worker and more round trip than those 6.7 ms (estimate: about 90 ms), at 0.67 USD/hr more. The 5090 was first
  until then; `pgs up flex_pi_libero --gpu "RTX 5090"` still picks it. The RTX 4090 does not fit (26
  GB in use). The H100 (65.4 ms, the hosted service's level, 5.89 USD/hr) is not in the default pools: `pgs up
  flex_pi_libero --gpu H100`.

## Through `pgs up` (2026-11-16)

`pgs up flex_pi_libero` on the merged branch, a new 60 GB STANDARD volume in EUR-NO-1, RTX 5090 (the first pool):
environment build 51 s, ModelScope prefetch 71.6 s (VAE 157 MB/s, umT5 encoder 250 MB/s), checkpoint download and
model load 170.8 s, fast path build and self-test 157.7 s (passed: 0.00061 mean / 0.005 max from the stored 5090
reference); worker boot to ready 456 s, `pgs up` ready after 7:54. `pgs demo flex_pi_libero` (random observation, no
depth): actions (32, 7), server time median 103 ms, round trip from a US east coast client median 264 ms. GPU memory in
use 25.2 GB of 32. Then `pgs delete flex_pi_libero --volume`. The first try was refused by Runpod (request over its
size limit, see `cosmos3_policy/NOTES.md`); the core now ships `verify.py` only for bring-up runs.

## Requests for the core

Status after the merge with the core (2026-11-16): see `families/cosmos3_policy/NOTES.md`. The hf_transfer request is
done: `bundle.family_env` sets `HF_HUB_ENABLE_HF_TRANSFER=0` (`build_env.sh` still installs `hf-transfer==0.1.9`, so
either setting works). The CPU thread cap: `boot.sh` sets `OMP_NUM_THREADS` to the container's CPUs, so torch's default
no longer follows the host's cores; `serve.py` keeps its own cap of 16 below that (measured above).
