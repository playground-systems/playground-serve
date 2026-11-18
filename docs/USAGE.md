# Using Playground Serve

The reference for `pgs` and the Python client: install, the API key, commands and options, costs, how it works, and
the known limits. The [README](../README.md) has the quick start; [MODELS.md](../MODELS.md) every model entry;
[BENCHMARKS.md](BENCHMARKS.md) the measured numbers.

## Requirements

- A Runpod account with credit. A first test costs well under 1 USD; see [Costs](#costs-and-when-it-bills).
- A Runpod API key. Create it at https://console.runpod.io/user/credentials (API Keys, Create API Key). Permission:
  All, or Restricted with Read/Write for Serverless and for Network volumes.
- Python 3.11 or later for `pgs`, and pip or [uv](https://docs.astral.sh/uv/). Installing from GitHub needs git.

## Install

```bash
pip install playground-serve      # pip
uv tool install --python 3.12 playground-serve   # or as a uv tool
```

Both install the commands `pgs` and `playground-serve` (the same program).

No install: put `uvx --python 3.12 --from playground-serve` in front of
each command, for example:

```bash
uvx --python 3.12 --from playground-serve pgs up pi05_libero
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
- The client runs on Python 3.11 or later. `pgs` needs Python 3.11 or later.
- Wire format, for clients in other languages: [WIRE.md](WIRE.md).

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
| `pgs delete MODEL [--volume]` | Deletes the endpoint, and with `--volume` the volume. Asks first; `--yes` deletes without asking (needed in a script, where there is no terminal to ask on). |
| `pgs models` | Every registered model with its GPU, hourly price and whether it ran on a GPU. Works without a key. |
| `pgs run ROBOT` | Drives a robot (`so101`) with a running model, with limits on the robot side; `--sim` for a simulated arm. Starts no GPU. [RUN.md](RUN.md). |

Options of `pgs up`:

- `--warm` keeps one worker running (and billing) until `pgs down`.
- `--idle MIN` sets how long a worker stays after its last request or client (default 5 minutes).
- `--region DC` picks the data center of a new endpoint, for example `US-NC-1`.
- `--gpu GPU` uses another GPU than the model's default pools, by name or Runpod pool, for example `H200`. It bills
  that GPU's price (H200 6.99 USD/hr against 1.34 for the RTX 4090, list prices read 2026-11-13), only the RTX 4090
  and the H200 get the fast path (other GPUs serve stock openpi), and the endpoint keeps that GPU on a later `pgs up`.
- `--stock-path` serves openpi's own policy instead of the fast path.
- `--direct` also serves on a TCP port of the worker's own public address; the Python client then sends its requests
  there instead of through Runpod's gateway (see [Direct port](#direct-port)).
- `--adopt [ID]` and `--new`: see [Ownership](#ownership).

The fast-path GPUs above are `pi05_libero`'s; each model family lists its own in [MODELS.md](../MODELS.md).

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

## How it works

`pgs up` creates a Runpod load-balancing serverless endpoint on Runpod's stock PyTorch image and a network volume in
the same data center. The worker scripts travel compressed in an environment variable of the endpoint, so there is no
container image to build; Runpod refuses an endpoint request over about 100 kB, so `pgs up` measures its request
before it creates anything and stops with a message if it is too large. On its first start the worker installs
openpi at a pinned commit, downloads the checkpoint from openpi's public bucket, and serves; then it saves the
environment and checkpoint to the volume for later starts.
Clients talk to it through Runpod's gateway over one websocket with openpi's wire protocol; the Python client sends
each JPEG request as two halves on two sockets (`connect(..., split=False)` turns that off; [WIRE.md](WIRE.md)).

On `pi05_libero` with an RTX 4090 (or an H200, `--gpu H200`) the worker serves the fast path: realtime-vla's Triton kernels in one CUDA graph,
with the same bf16 weights, the same 10 flow steps and openpi's own transforms. At every start it must reproduce stock
openpi's stored actions for a real frame, or it serves openpi's own policy instead. `policy.metadata["playground_serve"]`
says which path serves (`path`) and why (`path_reason`). Everywhere else the worker serves openpi's own policy.

The other model families work the same way with their own environment, checkpoint and fast path:
[MODELS.md](../MODELS.md#model-families).

### Direct port

`pgs up MODEL --direct` (an endpoint made with it keeps it) also lists a TCP port on the endpoint. Runpod gives a
serverless worker a public address and maps that port to it; the worker serves the same protocol there behind TLS,
with a certificate and a token it makes at every boot and sends in its metadata. The Python client reads them through
the gateway socket (which checked your API key), opens a second connection straight to the worker and sends its
requests there. The gateway socket stays open: it starts and keeps the worker, carries the keep-alive, and takes the
requests again if the direct connection cannot be opened or is lost. Nothing changes in billing, scale to zero or
start time.

- Measured from the US east coast to an RTX 4090 in US-CA-2 (California), same worker, alternating blocks of
  requests: the direct port saved about 10 to 12.6 ms per request against the gateway path, and `hot=True` (below)
  2.3 ms more. Round trip p50 111.6 ms direct with `hot` against 124.8 ms through the gateway, p95 119.5 against
  133.8 ([BENCHMARKS.md](BENCHMARKS.md)).
- What you give up: the direct connection does not pass Cloudflare or Runpod's load balancer. It needs outbound TCP
  to a high port of a plain IP address (some company networks block that; the client then stays on the gateway), its
  TLS certificate is the worker's own (pinned by the client, not signed by a public authority), and the port is open
  to the internet behind the token. A reply is one packet: when it is lost, the worker's TCP waits a whole
  retransmission timeout and that request takes about 385 ms (0.3 to 2 % of direct requests in earlier runs, 1 in
  2000 in the measured one). `connect(..., direct=False)` keeps a client on the gateway.
- `connect(..., hot=True)` (any endpoint) asks the worker not to let its thread or GPU idle for about a second after
  each request: 2.3 ms less model time per request at a remote client's pace on an RTX 4090. The worker's CPU
  core and GPU stay busy meanwhile (the GPU draws power as if it were working), so it is off by default.

### Ownership

The state file (`~/.config/playground-serve/state.json`, or `$PLAYGROUND_SERVE_HOME/state.json` for one per project)
holds a random owner id that is part of every resource name `pgs` creates. `up`, `down` and `delete` act only on
yours. If another state file or user on the same Runpod account has an endpoint for the model, `up` changes nothing
and offers `--adopt [ID]` (take it over) or `--new` (create your own).

## Development

```bash
git clone https://github.com/playground-systems/playground-serve
cd playground-serve
uv run pytest -q
```

`tests/client/test_import.py` describes how to run the client tests on Python 3.11. `scripts/check_release.sh` checks
the tree for secrets and private names before a release. `python3 docs/assets/make_assets.py` redraws the README's
pictures (the chart reads its numbers from MODELS.md). [CONTRIBUTING.md](../CONTRIBUTING.md) has the rest: what the
tests on every pull request run, which tests skip and why, and how to add a model family.


## Security upgrade compatibility

The image decoder now requires Pillow 13 and Python 3.11 or later. Upgrade older Python installations before
installing this release. Worker environment caches use a new generation, so the first start rebuilds them; existing
cached environments must pass the Pillow version check before serving. Flex-π also moves to Python 3.11.

Automatically discovered Runpod keys are sent only to `wss://<endpoint>.api.runpod.ai` on the default TLS port.
Custom URLs, including URLs in the state file or `PLAYGROUND_SERVE_URL`, receive no implicit credential. Pass
`api_key=` or `key_file=` explicitly when a custom service needs its own credential.


## LLM episode ownership

Workers advertising `playground_serve.episode_tokens` require a secret `episode_token` to resume a named episode
on another connection. The first request may supply a random token (at least 32 ASCII characters); otherwise the
worker generates one. Each reply returns it as `agent.episode_token`. Treat it as a credential and do not log it.
The Python client generates and retains tokens for its most recent 128 episode IDs across reconnects. To resume
from another client instance, supply the token explicitly. An episode ID alone cannot resume or replace a session.

## Input validation

Array dtypes must be boolean, integer, floating point, byte string or Unicode string, without object references.
Array byte lengths must match their shapes; each array is limited to 32 MiB and 32 dimensions. These checks apply
in the client and in both worker server implementations, including the wrapped openpi handler.
