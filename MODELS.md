# Models

Playground Serve serves openpi policies at openpi commit `215abfb2`, and the entries of other model families ([Model families](#model-families) below). Every openpi model runs the same worker, image (`runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404`) and environment tar; only the openpi config and the checkpoint differ. The registry is `src/playground_serve/models.py`; `pgs models` prints it with prices.

**`pi05_libero` and `pi05_droid` have run on a real GPU.** The other seven entries are registered and pass the CPU checks below, but no one has served them on a GPU with this code. Treat them as untested: they may fail at weight loading, run out of memory, or be slow.

| Model | Robot | Actions per call | Checkpoint | GPU pools | Ran on a GPU | Checks |
|---|---|---|---:|---|---|---|
| `pi05_libero` | LIBERO (sim) | 10 x 7 | 11.6 GiB | ADA_24, AMPERE_48 | yes: RTX 4090 (fast path) and A40 (stock path) | CPU checks |
| `pi05_droid` | DROID | 15 x 8 | 11.6 GiB | ADA_24, AMPERE_48 | yes: RTX 4090, stock path (2026-11-16: first start 1:13, `infer_ms` p50 86.6 ms) | CPU checks, worker smoke test |
| `pi0_fast_droid` | DROID | 10 x 8 | 10.1 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks (inputs only) |
| `pi0_droid` | DROID | 10 x 8 | 11.2 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks |
| `pi05_aloha` | ALOHA | 50 x 14 | 11.6 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks |
| `pi0_aloha_towel` | ALOHA | 50 x 14 | 11.2 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks, worker smoke test |
| `pi0_aloha_tupperware` | ALOHA | 50 x 14 | 11.2 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks |
| `pi0_aloha_pen_uncap` | ALOHA | 50 x 14 | 11.2 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks |
| `pi0_aloha_sim` | ALOHA (sim) | 50 x 14 | 11.2 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks |

GPU pools: `ADA_24` is the RTX 4090, `AMPERE_48` the A40; `pgs up` takes the first pool with stock. `pgs up MODEL --gpu H200` (pool `HOPPER_141`) uses the H200 instead.

- CPU checks (`tests/test_models_openpi.py`, no weights; they run only where openpi is installed): the config exists at the pinned commit, the worker's warm-up observation (openpi's `make_libero_example`, `make_droid_example` or `make_aloha_example`, picked from the config's robot transform) passes the config's input transforms with the shapes and dtypes the model takes, and the output transforms return the action shape above. For `pi0_fast_droid` the output side decodes FAST tokens and is not checked; its shape comes from the config. The worker smoke test (`tests/test_bundle_worker.py`) runs the bundled `serve.py` on CPU with a stand-in for the model.
- Not checked without a GPU boot: weight loading, GPU memory, compile and inference time, and RAM. `min_container_ram_gb` (16) is an estimate; the volume size (40 GB) covers the 8.2 GB environment tar plus one checkpoint, with room for a second.
- Checkpoint sizes are read from the public `gs://openpi-assets` bucket. `pi05_aloha` is the base model `pi05_base` (openpi's `serve_policy --env ALOHA` default), not fine-tuned on a task; send a prompt.
- ALOHA cameras are channel-first (3 x H x W). The client's jpeg mode only encodes H x W x 3 images, so ALOHA frames go raw.
- `pi0_fast_droid` loads its FAST tokenizer from Hugging Face (`physical-intelligence/fast`, remote code) at each fresh container.

## Model paths: fast and stock

The worker serves one of two paths (`worker/fastpath.py`); the metadata frame's `playground_serve` map says which (`path`) and why (`path_reason`), and so do the worker log's `path`, `fallback` and `ready` lines.

- **fast** (default for `pi05_libero` on an RTX 4090 and on an H200): realtime-vla's Triton kernels captured as one CUDA graph ([dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) at `eb46c8d`, MIT), behind openpi's own tokenizer, normalization and output transforms (`worker/fast_policy.py`). Same checkpoint and bf16 weights (realtime-vla's converter, no quantization, no fp16), same 10 flow steps, the flow state in float32 as openpi keeps it, same JPEG q90 transport. It drops only tokens that get zero attention: the all-zero, masked third LIBERO image and the masked prompt padding (32 slots; LIBERO prompts have at most 21 tokens; a longer prompt is refused with an error, never cut). Measured on the RTX 4090: served `infer_ms` 31 to 34 ms against 75 to 79 for stock, actions 0.0014 mean / 0.018 max from stock at fixed noise against 0.036 / 2.0 for a new noise draw ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
- Safety: every start runs a self-test before the worker reports ready: a stored real frame (JPEG bytes, seed 7) must reproduce stock openpi's stored actions (`worker/selftest_pi05_libero.json`) within 0.1 max and 0.02 mean, twice. If realtime-vla cannot be fetched, the conversion or graph build fails, or the self-test fails, the fast server exits and the stock path starts in its place with the reason in the log and metadata. The unused prompt slots are zeroed on every call (an early version left them uninitialized and gave NaN or garbage actions in 1 of 4 processes), and a reply with non-finite actions becomes an error instead.
- Not on the fast path: the RTX 5090 (realtime-vla needs torch 2.7.1+cu128 swapped into the environment tar that all models share; left out), the L40S and H100 (measured in-process only, and in no model's GPU pools), the A40 (`pi05_libero`'s second pool, not measured), every other model and custom checkpoints (no stored stock reference; their views, prompt lengths and `discrete_state_input` were not checked).
- **stock**: openpi's own `Policy` as `scripts/serve_policy.py` builds it, unchanged. Everywhere else, with `pgs up MODEL --stock-path`, and as the fallback. Faster variants of the stock path that were measured (input trims, a one-jit host path) were not merged into it: it is the fallback and the reference the fast path is checked against; the 32-slot trim would cut prompts over 32 tokens (exact only below that, unless a second compiled program covers longer ones, which costs start time); dropping the third image is valid only for configs whose third slot is always masked; and the one-jit path replaces openpi's `Policy.infer` and its compile-ahead for about 8 ms.
- Measured with the fast path in the packaged worker (2026-11-14, RTX 4090, US-NC-1): worker boot to ready 106, 120 and 173 s on three fresh starts (checkpoint download 327, 225 and 119 MB/s, then conversion 35 to 44 s) and 53 s from the volume; self-test 0.0022 mean / 0.011 max every time; 24 observation/seed pairs through the gateway: 0.0014 / 0.019 against a live stock run (JPEG q90: 0.0050 / 0.044; a new noise draw: 0.036 / 2.0); US-NC-1 round trip p50 / p95 61.1 / 68.8 ms back to back and 65.7 / 75.5 ms at 5 Hz.
- Measured with the current first-start code (2026-11-14): worker boot to ready 47 to 52 s on hosts that download at 391 to 438 MB/s and 88 and 105 s at 175 and 140 MB/s (kernels compiled in their own process during the download, the checkpoint converted on the GPU in the serving process, same bf16 weights: all 45 tensors identical to realtime-vla's converter); `pgs up` on a fresh account 60.3 to 118.3 s; from the volume the worker is ready in 50 to 52 s.
- All measured numbers with their conditions: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Third-party code and data: [NOTICE](NOTICE).

## Volume layout

`/runpod-volume/rs/env.tar` (with `env.tar.size`) is shared by all models. A `gs://` checkpoint lives at `/runpod-volume/openpi_data/<bucket>/<path>` (with `.rs_complete`, the list of its files and sizes), so each model has its own. JAX compilation caches are per model and GPU type: `/runpod-volume/jax_cache/<model>/<GPU name>`. The fast path keeps its checked copy at `/runpod-volume/fast/<model>/`: `model.pkl` (6.7 GB, realtime-vla's converted weights), `assets/` (norm stats), `triton_cache/` and `.rs_complete`; with it a later start needs no checkpoint, so a start that served the fast path does not copy the 12 GB checkpoint to the volume (a fallback downloads it, about 40 s).

A first start does not wait for the volume: the worker builds the environment on its own disk while it downloads the checkpoint there (`worker/stage.py fetch`, parallel HTTP ranges from the public bucket), compiles the model with zero weights while the download runs (filling the JAX cache), then loads the real weights and serves; the fast path instead compiles its kernels with placeholder weights while the download runs, then converts the checkpoint in the serving process (the converter's arithmetic on the GPU, same bf16 weights), builds its graph and runs its self-test, and saves the converted weights for the volume after it serves. Once it serves, `stage.py fill` writes the environment tar and the checkpoint to the volume under temporary names, checks their sizes, renames them and writes the size marker last; a later boot trusts only complete, marked copies, so a fill cut short (worker scaled down mid-copy) only means the next start builds and downloads again. Other checkpoint URIs and paths load through openpi as before.

## Serving your own checkpoint

Give any openpi config name at the pinned commit and your checkpoint, as `CONFIG@CHECKPOINT` (`pgs up CONFIG@CHECKPOINT`, `models.get`) or `models.custom(config, checkpoint)`, for example `pgs up pi05_droid@gs://my-bucket/pi05_droid_finetune/30000`. Its name is `custom-<config>-<hash>`: use that name (or the same `CONFIG@CHECKPOINT`) with `pgs down`, `pgs delete` and `connect`.

- The checkpoint is a `gs://` URI the worker can read (it downloads once into the volume) or an absolute path on the worker, such as a directory you copied to `/runpod-volume/checkpoints/...`. It needs openpi's layout: `params/` and `assets/<asset id>/norm_stats.json`, as openpi's training writes it.
- The config must use a LIBERO, DROID or ALOHA input transform; the worker reads which one from the config and warms up with that robot's example. Other robots need an entry in `worker/warmup.py`.
- The config name is only checked at the worker's policy load. Each config and checkpoint pair gets its own name and JAX cache. Custom checkpoints have not been run on a GPU (untested). `pgs demo` does not cover them; call `connect()` with your robot's observation.

## Model families

Besides openpi, the registry holds the entries of other model families. A family is one folder,
`src/playground_serve/families/<family>/`: `spec.py` declares its entries (`SPECS`, `ProposedSpec` from
`families/common/proposed_spec.py`: checkpoint, GPU pools in order of preference, volume size, RAM, chunk shape,
robot, license, secrets, extra env), `serve.py` serves one checkpoint behind the same websocket protocol
(`families/common/pgs_server.py`), `build_env.sh` builds the family's Python environment, `NOTES.md` holds its license,
measurements and spend. `models.py` finds the folders itself, so a new family changes no shared file. Each entry is a
model like an openpi one: `pgs up ENTRY`, `pgs down`, `pgs delete`, `pgs status`, `pgs models`,
`connect("ENTRY").act(obs)`, `pgs demo ENTRY` (with the family's example observation), `--gpu`, `--stock-path`.

- Volume layout (`bundle.family_env`): the family's environment tar `/runpod-volume/pgs/<family>/env.tar`, the
  checkpoint's complete copy `/runpod-volume/pgs/weights/<repo>@<revision>[/<subfolder>]`, `HF_HOME=/runpod-volume/hf`,
  and the family's own caches under `/runpod-volume/pgs/<family>/`. A first start builds and downloads on the container
  disk at the same time and fills the volume after it serves, as for openpi.
- Checkpoints: `hf://<repo>@<revision>[/<subfolder>]` (parallel ranges from Hugging Face's CDN; `RS_CKPT_FILES` in the
  spec's env keeps only matching files), a `gs://` directory or one `https://` file.
- Secrets: an entry's `secrets` (HF_TOKEN for a gated backbone) come from the user's machine at `pgs up` (the
  environment variable, or the token file of `hf auth login`) and go only into the endpoint's env on the user's own
  Runpod account; `pgs up` stops before it creates anything when one is missing.
- Images: a family's metadata may ask for `image_size: "native"` (frames at their own size) and advertise `jpeg_stack`
  (frame stacks as one JPEG per frame); `playground_serve.History` builds multi-tick observations from the metadata.
  The worker decodes a request's JPEG frames on up to 8 threads (`server_decode.py`), pixel for pixel as one after
  another: 12 frames of 256 x 256 took 5.3 to 5.6 ms against 5.5 to 8.4 ms serially (RTX 4090 host, three runs of 200;
  part of each decode holds Python's lock). Replies carry `server_timing.decode_ms` and the families'
  `policy_timing` (docs/WIRE.md).
- Fast paths: each family's `serve.py` serves its fast path only where it was measured, only after a boot self-test
  against stored stock actions, and falls back to its stock path (the official inference code) otherwise;
  `pgs up --stock-path` turns it off. The metadata's `playground_serve` map says which path serves and why.

An entry marked "experimental" (spec field `experimental`) is slow or not fully verified: `pgs models` marks it and
`pgs up` says so before it starts; it serves like any other entry.

Numbers are the server's `infer_ms` p50 (the worker's own timer, also returned through the gateway) on the entry's
default GPU, the first pool of its spec, unless the column names another GPU; the RTX 4090 is pool ADA_24 (1.34
USD/hr), measured in US-IL-1. "Fast against stock" is the action difference (mean / max) at the same noise; a new noise
draw moves the actions by much more (the family's NOTES.md has both). "First start" is a start on a new volume as the
family's NOTES.md measured it (for Cosmos 3, Flex-π and FLUX 3 Action: worker boot to ready; the GPU is named where it is not the
default).

| Entry | Robot | Actions | License | Default GPU, USD/hr | First start | `infer_ms` p50 stock / fast, default GPU (p90) | Other GPUs, stock / fast | Fast against stock (mean / max) |
|---|---|---|---|---|---|---|---|---|
| `abc_vla_yam` | YAM (ABC sim) | 30 x 14 | Apache-2.0 + Gemma terms | RTX 4090, 1.34 | 76 s; 293 s with a cold compile cache | 84.0 / 28.8 (p90 29.8) | H100 67.1 / 20.5 | 0.0032 / 0.022 (self-test, zero noise) |
| `xiaomi_r1_robocasa365` | RoboCasa365 | 16 x 12 | Apache-2.0 | RTX 4090, 1.34 | 102 s (compile of a new GPU type) | 267.9 / 65.5 (341.4 / 71.3) | H100 177.0 / 47.6 | 0.0006 to 0.0008 / 0.004 to 0.008 |
| `gr00t_n17_libero` | LIBERO | 16 x 7 | NVIDIA Open Model License; gated backbone (HF_TOKEN) | RTX 4090, 1.34 | 121 s | 118.4 / 68.5 (127.4 / 98.7) | | 0.0 / 0.0 |
| `gr00t_n17_libero_goal`, `_object`, `_10` | LIBERO | 16 x 7 | same | RTX 4090, 1.34 | not run (same code, other weights; serve stock) | | | |
| `molmoact2_libero` | LIBERO | 10 x 7 | Apache-2.0, research and educational use | RTX 4090, 1.34 | 151 s (stock) | 520.7 / 104.2 (552.4 / 107.3) | | 0.00025 / 0.0029 |
| `molmoact2_droid` | DROID | 15 x 8 | same | RTX 4090, 1.34 | 105 s (stock); 129 s (fast) | 532.0 / 104.8 (568.9 / 110.0) | H100 301.9 / 69.1 | 0.0003 / 0.0156 |
| `molmoact2_so101` | SO-101 | 30 x 6 | same | RTX 4090, 1.34 | 123 s (fast) | 565.5 / 111.8 (661.1 / 117.4) | | 0.009 to 0.018 / 0.25 to 1.0 (degrees) |
| `molmoact2_yam` | bimanual YAM | 30 x 14 | same | RTX 4090, 1.34 | 149 s (fast) | 574.8 / 139.3 (601.0 / 141.7) | H100 384.5 / 99.8 (earlier fast path) | 0.0004 / 0.0078 |
| `cosmos3_edge_droid` (experimental) | DROID | 32 x 8 | OpenMDW-1.1 (code and weights; commercial use allowed) | RTX 5090, 1.89 | 174 s on the 5090 (verify run); A40: 192 s | 514.0 / 439.5 (643.3 / 502.6); merged core: 498.9 / 451.7 | A40 853 / 839; H100 274.1 / 235.3 | 0.0 / 0.0 (same seed; sample bit-identical) |
| `cosmos3_nano_droid` (experimental) | DROID | 32 x 8 | same | A100, 2.99 | not run (needs an 80 GB card) | | | |
| `flex_pi_libero` | LIBERO | 32 x 7 | MIT (code and weights); DINOv3 License (vision encoder) | A40, 1.22 | 456 s on the RTX 5090 (`pgs up` ready after 7:54) | 318.5 / 131.9 | RTX 5090 310.6 / 125.2; H100 152.4 / 65.4 | 0.0005 to 0.0008 / 0.004 to 0.0075 |
| `flux3_action_droid` (experimental) | DROID | 32 x 8 | FLUX Kommunity License v1.0, non-commercial; gated (HF_TOKEN) | A40, 1.22 | 260 s to the first request | 3180 to 3228 (range; stock only) | | no fast path |

Three entries are marked experimental by the owner (spec field `experimental`): `cosmos3_edge_droid` and
`cosmos3_nano_droid` (verified, but slow for a policy, and Nano not run) and `flux3_action_droid` (stock only, 3.2 s
per call). Each family's NOTES.md has the conditions: served through the gateway or by the worker's own timer (the
same `infer_ms`), JPEG frames, 10 to 40 calls (Cosmos 3, Flex-π and FLUX 3 Action: medians of 8 to 20 alternating stock and fast
requests on one worker). ABC-VLA's fast path compiles its graph body
(inductor, emulated precision casts) with the cache on the volume per GPU type: the first start on a new GPU type takes
3 to 4 minutes (216 s of compiling on the RTX 4090), later starts about 70 s; `RS_FAST_COMPILE=0` in its spec's env
gives the exact graph path (bit-identical to stock, 37.0 ms).

- MolmoAct2 (Ai2, `families/molmoact2/NOTES.md`): stock is the repo's own server call (`examples/droid` and
  `examples/yam` servers: bfloat16, CUDA graph off); it equals the official server class bit for bit on the same
  inputs and seeds (LIBERO, DROID, YAM; SO-101 has no server). The fast path uses the same bfloat16 weights, 10 flow
  steps and inputs, with the action expert's own CUDA graph (the model card's default) and `torch.compile` of the
  expert's blocks and context projection (rounding as eager does), the image resize and normalization on the GPU
  (checked against the official processor at every start) and CUDA graphs of the vision encoder and of the language
  model's prefill per prompt length; it is not bit-exact, 1 to 4 % of a new noise draw's mean difference. Self-test bounds are fractions of the stored noise draw (mean 0.1, max 0.5). Fast path on the RTX
  4090 only; the A40 serves stock. The model card's reported results are float32 (24 to 26 GB, a 48 GB GPU); the
  served precision follows the repo's servers.
- MolmoAct2's `infer_ms` above is the server's own timer measured on one RTX 4090 worker (40 calls, the gateway returns
  the same timer); its NOTES.md has the gateway runs, the slow-host runs and the H100 runs.
- The faster-GPU rule (entries above about 200 ms on the RTX 4090): `xiaomi_r1_robocasa365` qualified; its pools are
  RTX 4090, RTX 5090 (Serverless stock only in Europe), A40. MolmoAct2's fast path serves at 104 to 139 ms (p50) on the
  RTX 4090; on an H100 DROID's fast path serves at 69.1 ms (p50), 1.5 times faster for 4.4 times the price, so its
  pools stay RTX 4090, then A40.
- Cosmos 3 Edge and Nano on DROID (NVIDIA, `families/cosmos3_policy/NOTES.md`): stock is the repo's RoboLab policy
  service (`RobolabPolicyService`, the docs' launch flags, 4 UniPC steps, guidance 3.0), equal to the official
  `infer` bit for bit at the same seed. The fast path is the same service and model call; only the pre-processing of
  the 33-frame clip (all frames after the first are zeros) resizes and pads the first frame alone, which gives the same
  sample bit for bit (checked at every start) and the same actions. Fast path on the RTX 5090, A40 and H100. Pools:
  RTX 5090 first, A40 as the cheaper fallback; `--gpu H100` for 235 ms. The checkpoints download through the model's
  own loader into `HF_HOME` (`RS_SELF_FETCH=1`), and the CUDA 13 environment needs hosts with CUDA 13.0 or newer
  (`min_cuda`).
- Flex-π on LIBERO (`families/flex_pi/NOTES.md`): the default is the official action-only mode (the model card's
  "fast action-only path"; LIBERO success 98.4 % against 98.5 % for full joint), same weights, 4 flow steps, seed 42,
  inputs as openpi's LIBERO client (a `pi05_libero` client works; depth optional). The fast path uses the repo's own
  `prepare_for_inference` options (each denoise step a CUDA graph, compiled encoders, cached constants) and the
  prompt's text encoding cached per prompt. Full joint mode (video, DINO and pointmap futures with the actions; H100
  341 / 236 ms, A40 1162 / 903): set `"RS_FLEXPI_JOINT": "true"` in the entry's env in `families/flex_pi/spec.py` and
  run `pgs up flex_pi_libero` again (a new release); the RTX 5090 then serves stock. Pools: A40 first, RTX 5090 second
  (the RTX 4090 is too small: 26 GB in use); `--gpu H100` for 65.4 ms.
- FLUX 3 Action on DROID (Black Forest Labs, `families/flux3_action/NOTES.md`): non-commercial license, both
  repositories gated (accept the license on Hugging Face; `pgs up` takes `HF_TOKEN` from this machine). It serves the
  official eager setting (`--no-compile-dit`); the official compiled setting failed at every start on this stack.
- The RTX 5090 had Serverless stock only in Europe on 2026-11-16: an entry with the 5090 first gets a European worker
  for a user in the US (about 90 ms more round trip, estimate). For Cosmos Edge that is still far below the 400 ms the
  5090 saves over the A40, so it stays first there; for Flex-π it saves 6.7 ms, so the A40 is first and `pgs up
  flex_pi_libero --gpu "RTX 5090"` picks the 5090.

## LLM agents (no GPU)

`families/llm_agent/` serves a language model as a robot policy behind the same websocket wire. The agent loop is
Inspect Robots' LLM agent policy (`inspect-robots-agent`, MIT, a pinned dependency): the model sees the state as text and
the camera images and calls tools (`move_to` or `move_joints`, `sleep`, `take_pic`, `done`, `give_up`); the server turns
the targets into the environment's actions. Any provider of that package works: Anthropic, OpenAI, Google, xAI, Groq,
Mistral, DeepSeek, OpenRouter, and any OpenAI-compatible server by base URL. Each episode starts from an embodiment
card that the client sends (the robot, its action space, its cameras and keys; built-in: `libero_franka`).
[NOTES.md](src/playground_serve/families/llm_agent/NOTES.md) has the design, the harnesses compared, the providers'
test status and the measurements; its Wire section the request and reply keys.

It has no `spec.py`, so it is not a registry entry and `pgs up` does not create it yet: it runs on a CPU worker without
a volume, which the launcher does not make. Run it on any machine with the packages of its `requirements.txt`:

```bash
D=src/playground_serve
PYTHONPATH=$D/families/llm_agent:$D/families/common:$D/worker ANTHROPIC_API_KEY=... \
  PGS_AGENT_MODEL=anthropic/claude-haiku-5-5 python $D/families/llm_agent/serve.py --port 8000
```
