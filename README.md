# Playground Serve

Playground Serve runs an openpi robot policy (π0.5) on a serverless GPU in your own Runpod account. One command
starts it, and three lines of Python on your robot or simulator get actions from it.

## Quick start

You need a [Runpod](https://www.runpod.io) account with credit and Python 3.10 or later.

```bash
pip install git+https://github.com/playground-systems/playground-serve
pgs up pi05_libero   # start the GPU server: about 1 to 2 minutes the first time
pgs demo             # send one dummy observation, print the actions' shape and the round trip
pgs down             # stop GPU billing now
```

The first `pgs up` asks for a Runpod API key. It prints the console page where you create one and the permissions to
pick, reads the key with hidden input, checks it with one read, and saves it for later commands.

What you see (shortened; your data center, times and prices will differ):

```text
$ pgs up pi05_libero
GPU:    RTX 4090 (pool ADA_24) in US-NC-1, serverless stock HIGH: 1.34 USD/hr while a worker runs; the worker stops by itself 5 min after the last request (scale to zero)
Volume: 40 GB STANDARD in US-NC-1: about 2.80 USD/month until deleted
...
Ready after 1:05.
Websocket: wss://<endpoint-id>.api.runpod.ai/ws

$ pgs demo
Actions: shape (10, 7) (horizon x action size; the observation was random, so the values mean nothing).
Round trip over 5 more calls: median 65 ms, max 75 ms; model time on the server median 33 ms.

$ pgs down
pi05_libero: no worker kept ... a running worker stops within about a minute once no client is connected, and GPU billing stops then.
```

Cost of this quick start: a few minutes of RTX 4090 time at 1.34 USD/hr, about 0.10 to 0.20 USD (estimate), plus
the 40 GB volume at about 2.80 USD/month until you delete it with `pgs delete pi05_libero --volume`.

No install: put `uvx --python 3.12 --from git+https://github.com/playground-systems/playground-serve` in front of
each command, for example:

```bash
uvx --python 3.12 --from git+https://github.com/playground-systems/playground-serve pgs up pi05_libero
```

## Use it from Python

```python
from playground_serve import connect, example_obs
policy = connect("pi05_libero")
print(policy.act(example_obs()).shape)   # (10, 7)
```

Replace `example_obs()` with your robot's observation. For `pi05_libero` it is the layout of openpi's LIBERO example:
`observation/image` and `observation/wrist_image` (uint8, height x width x 3), `observation/state` (8 floats) and
`prompt` (a string). `policy.act(obs)` returns the action chunk as a numpy array; `policy.infer(obs)` returns openpi's
whole reply. `connect` finds the endpoint in the state file that `pgs up` wrote, or on your Runpod account. It also
takes the `wss://` URL that `pgs up` printed.

- `ActionQueue` keeps a control loop fed while requests are in flight.
- `playground_serve.status("pi05_libero").state` says `warm`, `cold` or `missing` without starting a worker.
- The client runs on Python 3.8 or later. `pgs` needs Python 3.10 or later.
- Wire format, for clients in other languages: [docs/WIRE.md](docs/WIRE.md).

## Requirements

- A Runpod account with credit. A first test costs well under 1 USD; see [Costs](#costs-and-when-it-bills).
- A Runpod API key. Create it at https://console.runpod.io/user/credentials (API Keys, Create API Key). Permission:
  All, or Restricted with Read/Write for Serverless and for Network volumes.
- Python 3.10 or later for `pgs`, and pip or [uv](https://docs.astral.sh/uv/). Installing from GitHub needs git.

## Install

```bash
pip install git+https://github.com/playground-systems/playground-serve      # pip
uv tool install --python 3.12 git+https://github.com/playground-systems/playground-serve   # or as a uv tool
```

Both install the commands `pgs` and `playground-serve` (the same program).

## The API key

`pgs` and `connect()` read the key from, in order:

1. `--key-file PATH` (or `connect(api_key=...)`, `connect(key_file=...)`),
2. the `RUNPOD_API_KEY` environment variable,
3. the file `~/.config/playground-serve/key` (`$XDG_CONFIG_HOME/playground-serve/key` if that is set).

If none is set and `pgs` runs in a terminal, it asks for the key once and saves it to that file, readable only by you.
Outside a terminal (scripts, CI) it prints what to do and exits. To replace a saved key, delete the file or set
`RUNPOD_API_KEY`. The key is sent only to Runpod, in the Authorization header, and `pgs` never prints it.

A robot needs only a key that can call its endpoint. In the Runpod console, create a Restricted key with Read/Write on
the `playground-serve-<owner>-<model>-<hex>` endpoint and nothing else. Such a key cannot list endpoints, so pass the
URL that `pgs up` printed to `connect()` or set `PLAYGROUND_SERVE_URL`.

## Commands

| Command | What it does |
|---|---|
| `pgs up MODEL` | Finds or creates a network volume and a serverless endpoint for the model, starts a worker, and waits until it serves. Prints the GPU, the price and when billing stops before it creates anything. |
| `pgs demo [MODEL]` | Sends a dummy LIBERO observation (default `pi05_libero`) and prints the actions' shape and the round trip. Starts a worker if none runs. |
| `pgs status` | Your endpoints and volumes, worker state and the price now. Read only. |
| `pgs down [MODEL]` | Keeps no worker: a running worker stops within about a minute once no client is connected. The volume stays. |
| `pgs delete MODEL [--volume]` | Deletes the endpoint, and with `--volume` the volume. Asks first. |
| `pgs models` | Every registered model with its GPU, hourly price and whether it ran on a GPU. Works without a key. |

Options of `pgs up`:

- `--warm` keeps one worker running (and billing) until `pgs down`.
- `--idle MIN` sets how long a worker stays after its last request or client (default 5 minutes).
- `--region DC` picks the data center of a new endpoint, for example `US-NC-1`.
- `--stock-path` serves openpi's own policy instead of the fast path.
- `--adopt [ID]` and `--new`: see [Ownership](#ownership).

## Costs and when it bills

`pgs up` prints the live prices before it creates anything. Runpod list prices, read 2026-11-13: RTX 4090 1.34 USD/hr
and A40 1.22 USD/hr while a worker runs.

- GPU: by default no worker is kept (scale to zero). A worker that a request starts stops by itself 5 minutes after the
  last request if no client is connected. A connected client keeps it running; it stops 5 minutes after the last
  client disconnects. The next connect starts a worker again, which takes about 1 minute.
- `pgs up MODEL --warm` keeps one worker running until `pgs down`: 1.34 USD/hr, about 32 USD per day on an RTX 4090.
- `pgs down` sets a 5 second idle time: a running worker stops within about a minute once no client is connected.
- Region probe: when `pgs up` creates a new endpoint, it measures the round trip to nearby data centers while the
  worker starts. For that it creates one short-lived probe endpoint per data center (a CPU worker where Runpod offers
  one, else a GPU worker), deletes each one right after its measurement, and caches the result for a day. This costs a
  few cents at most (estimate).
- Network volume: 40 GB, about 2.80 USD/month (STANDARD) or 5.60 USD/month (HIGH_PERFORMANCE, where that is the only
  type), billed until `pgs delete MODEL --volume`. It holds the Python environment, the checkpoint and compile caches,
  so later starts skip the download. `pgs down` keeps it.

## Measured performance

`pi05_libero` on an RTX 4090 in US-NC-1, client on home Wi-Fi on the US east coast, JPEG quality 90:

- First start, nothing cached: 60 to 118 s over nine runs (median 65 s). It depends on the GPU host's download speed;
  on a slow host it can take over 2 minutes (estimate).
- Later start, from the volume: 61 to 89 s over three runs.
- Round trip: p50 61 ms and p95 69 ms back to back; p50 66 ms and p95 76 ms at 5 requests per second. While the
  client's Wi-Fi stalled, p95 rose to 113 to 140 ms.
- Model time on the GPU: 31 to 34 ms on the fast path, 75 to 79 ms with stock openpi; 146 ms with stock openpi on an
  A40.

Details and conditions: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Models

| Model | Robot | Status |
|---|---|---|
| `pi05_libero` | LIBERO (simulation) | Ran on a real GPU (RTX 4090 and A40). |
| `pi05_droid`, `pi0_fast_droid`, `pi0_droid` | DROID | Registered, untested on a GPU. |
| `pi05_aloha`, `pi0_aloha_towel`, `pi0_aloha_tupperware`, `pi0_aloha_pen_uncap`, `pi0_aloha_sim` | ALOHA | Registered, untested on a GPU. |

Only `pi05_libero` has run on a GPU. The others pass CPU checks against openpi but may fail on a GPU. You can also
serve your own checkpoint of any openpi config: `pgs up CONFIG@CHECKPOINT`, for example
`pgs up pi05_droid@gs://my-bucket/pi05_droid_finetune/30000` (untested on a GPU). Details: [MODELS.md](MODELS.md).

## How it works

`pgs up` creates a Runpod load-balancing serverless endpoint on Runpod's stock PyTorch image and a network volume in
the same data center. The worker scripts travel in an environment variable of the endpoint, so there is no container
image to build. On its first start the worker installs openpi at a pinned commit, downloads the checkpoint from
openpi's public bucket, and serves; then it saves the environment and checkpoint to the volume for later starts.
Clients talk to it through Runpod's gateway over one websocket with openpi's wire protocol.

On `pi05_libero` with an RTX 4090 the worker serves the fast path: realtime-vla's Triton kernels in one CUDA graph,
with the same bf16 weights, the same 10 flow steps and openpi's own transforms. At every start it must reproduce stock
openpi's stored actions for a real frame, or it serves openpi's own policy instead. `policy.metadata["playground_serve"]`
says which path serves (`path`) and why (`path_reason`). Everywhere else the worker serves openpi's own policy.

### Ownership

The state file (`~/.config/playground-serve/state.json`, or `$PLAYGROUND_SERVE_HOME/state.json` for one per project)
holds a random owner id that is part of every resource name `pgs` creates. `up`, `down` and `delete` act only on
yours. If another state file or user on the same Runpod account has an endpoint for the model, `up` changes nothing
and offers `--adopt [ID]` (take it over) or `--new` (create your own).

## Limits and known issues

- Only `pi05_libero` has run on a GPU. All other models and custom checkpoints are untested.
- The fast path runs only for `pi05_libero` on an RTX 4090. On the A40 and for every other model the worker runs
  stock openpi: 75 to 79 ms per inference on the RTX 4090 and about 146 ms on the A40, against 31 to 34 ms on the
  fast path.
- A first start depends on the GPU host's download speed (12.4 GB checkpoint plus about 3.5 GB of packages).
- On some later starts Runpod's gateway took 21 to 34 s to route to a worker that was already ready.
- `pgs demo` and `example_obs()` know only the LIBERO observation.
- ALOHA camera images are channel-first, so the client sends them uncompressed (larger requests).
- `pi0_fast_droid` downloads its tokenizer from Hugging Face at each fresh container.
- Runpod caps the sum of max workers over all endpoints of an account (5 on the account used for testing). Each
  endpoint `pgs` creates uses 1, and the region probe uses free slots for a short time.
- A new endpoint stays in its data center. To move it, `pgs delete MODEL --volume`, then `pgs up MODEL --region DC`
  (a first start again).
- Tested on macOS. Linux and Windows have not been tested.
- The error messages for an account without credit match Runpod's wording by keywords; that case was not seen live.

## Development

```bash
git clone https://github.com/playground-systems/playground-serve
cd playground-serve
uv run pytest -q
```

`tests/client/test_import.py` describes how to run the client tests on Python 3.8. `scripts/check_release.sh` checks
the tree for secrets and private names before a release.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party code and data, including openpi (Apache 2.0) and realtime-vla
(MIT), are listed in [NOTICE](NOTICE). The openpi checkpoints are subject to the Gemma Terms of Use.
