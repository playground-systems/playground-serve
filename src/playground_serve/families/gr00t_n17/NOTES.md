# gr00t_n17: NVIDIA GR00T N1.7 on LIBERO

Entries `gr00t_n17_libero` (suite `libero_spatial`), `gr00t_n17_libero_goal`, `gr00t_n17_libero_object`,
`gr00t_n17_libero_10`: one checkpoint folder per LIBERO suite in `nvidia/GR00T-N1.7-LIBERO@65854032`, identical
configs. 3B: the Cosmos-Reason2-2B (Qwen3-VL) backbone plus an alternating cross/self-attention DiT, 4 flow steps,
16 x 7 actions at 20 Hz. Code `NVIDIA/Isaac-GR00T@3b490679`.

**Status: `gr00t_n17_libero` served and verified on a GPU, stock and fast path (RTX 4090, 2026-11-16).** The other
three suites run the same code with other weights and were not run (no self-test reference: they serve stock). Numbers
are measured unless they say estimate.

## License and the gated backbone

Code Apache-2.0. Weights `nvidia/GR00T-N1.7-LIBERO`: NVIDIA Open Model License (commercial use and self-hosting
allowed), not gated. The code loads the backbone `nvidia/Cosmos-Reason2-2B` by name from Hugging Face at every start
(`gr00t/model/modules/qwen3_backbone.py:183`, `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:129`); that repo is
gated (`gated: auto`), with its own terms that the account behind `HF_TOKEN` must accept on Hugging Face. The owner
accepted them; the bring-up passed the owner's read token to its own endpoint as the env value `HF_TOKEN`, read from
the local token file at run time and never printed or stored (checked: the value is in no saved log, output or repo
file). The backbone is downloaded once into `HF_HOME` on the volume. Recorded in `spec.LICENSE` and the metadata.

## What serves

- **stock**: what `gr00t/eval/run_gr00t_server.py --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper` runs:
  `Gr00tSimPolicyWrapper(Gr00tPolicy("LIBERO_PANDA", <suite dir>, device="cuda", strict=True)).get_action(obs)`.
- **fast** (`gr00t_fast.py`, default for `gr00t_n17_libero` on the RTX 4090): the same call with the action head's
  `get_action_with_features` replaced, line for line, by a version whose 4-step loop is captured once per input shape
  as a CUDA graph; the backbone and the noise draw stay eager. Bit-identical to stock. RTC calls (an "action" in the
  action input) and batches above 1 run the stock method.
- Flow noise: `torch.randn` from torch's default stream, as the official server. An optional request key `rs_seed`
  gives a call its own seed (fork_rng) and leaves the stream alone; the checks use it.
- Self-test at every start (`selftest_gr00t_n17_libero.json`): frame 80 of episode 0 of the repo's
  `demo_data/libero_demo` (both cameras as JPEG q90, the recorded state and task) with seeds 7 and 8. Stock must match
  the stored actions (warning only, max 0.1) and fast must match live stock within max 0.01 and mean 0.001, twice;
  else stock serves.
- Request keys: the official flat ones (`video.image`, `video.wrist_image`, `state.x` ... `state.gripper`,
  `annotation.human.action.task_description`) without the batch and time dims; the reply is `actions` (16, 7) plus
  the official `action.<key>` arrays. Images 256 x 256 (metadata `image_size`, so the jpeg mode does not resize LIBERO
  frames). Client side as the official LIBERO env (`libero_env.py`): rotate both frames 180 degrees, send the eef
  position, axis-angle and the two gripper finger positions, map the gripper action to sign(2a - 1) and negate it,
  execute 8 of the 16 actions per call. The server does none of this.

## Measurements (RTX 4090, US-IL-1, 2026-11-16)

| What | Stock | Fast |
|---|---:|---:|
| Served `infer_ms` p50 / p90 (gateway, 20 calls, jpeg mode, 23 kB requests) | 118.4 / 127.4 | 68.5 / 98.7 |
| In-process per call p50 / p90 | 117.6 / 148.9 | 61.9 / 83.3 |
| Round trip from the US east coast, p50 / p90 | 186.8 / 268.8 | 132.6 / 215.3 |
| GPU memory (nvidia-smi after warm-up) | 6.7 GB | |

Stock breakdown in-process (median of 10): 128.4 ms per call, of which the backbone 39.1 ms and the action head's 4
steps 74.9 ms. The captured loop replays in 17.5 ms (capture 0.24 s per new shape).

Action checks (5 real frames of the LIBERO demo episode, frames 0 to 160, with its recorded states and task):
- Playground Serve's stock path against the official `PolicyServer` (ZMQ) and `PolicyClient`
  (`gr00t/policy/server_client.py`) on the same wrapper object, same seeds: **0.0 difference** on all 5.
- Fast against stock: **0.0** on the same 5 in-process at fixed seeds, and for two unseeded calls continuing the stream
  from the same seed; through the gateway, served fast against served stock: 0.0 on all 5.
- For scale: another seed moves the actions by 0.015 to 0.031 mean and 0.10 to 0.98 max.

Faster GPUs: not needed under the owner's rule (stock is 118 ms served, under 200 ms); not measured.

Start times:
- First start on an empty volume: ready 120.7 s after container start (environment build 49 s, 8.4 GB venv; checkpoint
  6.91 GB in 12.9 s at 536 MB/s; load 57.3 s, which includes downloading the 4.9 GB backbone into HF_HOME). The
  very first container failed at that download (below) and was replaced after the fix.
- Later start from the volume: connect to first action 73.6 s from the laptop.

Not tried: a captured backbone (39 ms; the same Hugging Face Qwen3-VL code as xiaomi_r1's prefill, with host
synchronizations), torch.compile of the loop (not bit-identical, minutes of compiling per cold start for xiaomi_r1).

## Round 2 (2026-11-16)

No change. A hosted service measured 76.1 to 76.9 ms of server inference for this model on an H100; ours serves at
68.5 ms (p50, JPEG decode included) on the RTX 4090. The techniques of xiaomi_r1's round 2 would apply here too (its
backbone is the same Hugging Face Qwen3-VL code, 39 ms of the 4090 call; the captured 4-step loop is 17.5 ms): not
done, and the GR00T replies now carry `policy_timing` (`policy_ms`) like the other families.

## Open points

- The backbone is not pinned: the official code loads it at `main` and `Gr00tPolicy` takes no revision (`9ce19a19` on
  2026-11-16). A pinned revision would need a change in the official code or a local snapshot under HF_HOME.
- The stock image sets `HF_HUB_ENABLE_HF_TRANSFER=1` without the `hf_transfer` package, so the backbone download
  failed until the spec set it to 0 (`spec.env`).
- The official ZMQ server imports the pip package `msgpack_numpy`, which the worker's own `/rs/msgpack_numpy.py`
  shadowed (it broke the official server in the in-worker check until the real package was loaded by path; the
  core's file is now `pgs_msgpack_numpy.py`). The serving path does not use the official server.

## Core support

What this family needed from the core is in it now: secrets from the user's machine (`HF_TOKEN`), `hf://` checkpoints
with a subfolder and file patterns (`RS_CKPT_FILES`), `pgs_msgpack_numpy.py` (named so that it no longer shadows the
pip package `msgpack_numpy`), a fetch that keeps a complete copy on a container restart, and the family bundle rules.

## GPU spend

Counted with abc_vla's and xiaomi_r1's in `abc_vla/NOTES.md`.
