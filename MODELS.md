# Models

Playground Serve serves openpi policies at openpi commit `215abfb2`. Every model runs the same worker, image (`runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404`) and environment tar; only the openpi config and the checkpoint differ. The registry is `src/playground_serve/models.py`; `pgs models` prints it with prices.

**Only `pi05_libero` has run on a real GPU.** The other eight entries are registered and pass the CPU checks below, but no one has served them on a GPU with this code. Treat them as untested: they may fail at weight loading, run out of memory, or be slow.

| Model | Robot | Actions per call | Checkpoint | GPU pools | Ran on a GPU | Checks |
|---|---|---|---:|---|---|---|
| `pi05_libero` | LIBERO (sim) | 10 x 7 | 11.6 GiB | ADA_24, AMPERE_48 | yes: RTX 4090 (fast path) and A40 (stock path) | CPU checks |
| `pi05_droid` | DROID | 15 x 8 | 11.6 GiB | ADA_24, AMPERE_48 | no (untested) | CPU checks, worker smoke test |
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

- **fast** (default for `pi05_libero` on an RTX 4090 and on an H200): realtime-vla's Triton kernels captured as one CUDA graph ([dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) at `eb46c8d`, MIT), behind openpi's own tokenizer, normalization and output transforms (`worker/fast_policy.py`). Same checkpoint and bf16 weights (realtime-vla's converter, no quantization, no fp16), same 10 flow steps, the flow state in float32 as openpi keeps it, same JPEG q90 transport. It drops only tokens that get zero attention: the all-zero, masked third LIBERO image and the masked prompt padding (32 slots; LIBERO prompts have at most 21 tokens; a longer prompt is refused with an error, never cut). Measured served `infer_ms` 31 to 34 ms against 75 to 79 for stock, actions 0.0014 mean / 0.018 max from stock at fixed noise against 0.036 / 2.0 for a new noise draw ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
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
