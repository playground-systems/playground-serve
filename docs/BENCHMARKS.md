# Measured numbers

All numbers are for `pi05_libero`, measured on Runpod Serverless on 2026-11-13 and 2026-11-14 with the code in this
repository or its direct predecessors. They are small samples from one client location. Your numbers will differ with
the GPU host, the data center, and your network. Lines marked "estimate" were not measured.

## Conditions

- GPU: RTX 4090 (Runpod pool `ADA_24`), data center US-NC-1, unless a line says otherwise.
- Client: a laptop on home Wi-Fi on the US east coast, one websocket, the package client, JPEG quality 90 (about
  19 kB per LIBERO request).
- Model: openpi `pi05_libero`, bf16 weights, 10 flow steps. The fast path (realtime-vla kernels in one CUDA graph)
  unless a line says "stock".

## Start time

First start: a fresh account state, no volume, nothing cached. Time from launching `pgs up pi05_libero` until it
exits with the server ready.

| | Result |
|---|---|
| First start, nine runs | 60.3 to 118.3 s, median 64.6 s |
| Of that, worker boot to ready | 47 to 52 s on hosts that downloaded at 391 to 438 MB/s; 88 and 105 s at 175 and 140 MB/s |
| Slowest two runs | 118.3 s (Runpod took 46 s to start the container), 115.4 s (download at 140 MB/s) |
| First start on a host that downloads at about 120 MB/s | about 133 s (estimate) |
| Later start, from the filled volume after `pgs down`, three runs | 61.4, 73.6 and 88.8 s |
| Of that, worker boot to ready | 49.3 to 51.7 s |

- The worker must download the 12.4 GB checkpoint and about 3.5 GB of Python packages before it can serve, so a first
  start depends on the host's download speed. 11 of 13 download rates measured were 175 MB/s or more.
- On some later starts Runpod's gateway routed nothing to the ready worker for 21 to 34 s. That time is included above.
- Connecting a client to an endpoint with no running worker starts one. Connect to first action took 42.7 to 119.1 s
  (median 63 s) over ten such starts from the volume, measured with an earlier worker build.

## Round trip from the robot

Round trip = request sent to actions received and decoded, through Runpod's gateway. N = 1000 requests per run,
after 3 warm-up requests.

| Run | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Packaged worker, back to back | 61.1 | 68.8 |
| Packaged worker, 5 requests per second | 65.7 | 75.5 |
| Fast-path test worker, two starts, back to back | 62.8 and 63.3 | 68.6 and 69.2 |
| Fast-path test worker, two starts, 5 requests per second | 65.8 and 67.1 | 75.2 and 77.2 |

- These runs were on a clean Wi-Fi link. While the laptop's Wi-Fi link stalled, p95 rose to 113 to 140 ms with the
  same worker. An empty message alone then took 98.6 to 111.3 ms at p95. The client's network matters as much as the
  server.
- The network part (round trip minus the server's model time) was about 31 ms at p50. A small message through the
  gateway to US-NC-1 took 24.8 ms at p50 (29.7 ms for a 19.5 kB request); to US-IL-1 the network part was 54.8 ms.
  Moving the model from US-IL-1 to US-NC-1 took 24 ms off every request. This is why `pgs up` picks a nearby data
  center.
- Stock path (openpi's own policy), RTX 4090 in US-IL-1, an earlier build, one run: p50 158.7 ms, p95 306.3 ms. The
  server's model time rose during that run; the cause was not found.
- Sending images without JPEG (`resize` mode, about 300 kB per request) did not reach a 100 ms p95 from this client.

## Direct port and `hot`, far from the robot

RTX 4090 in US-CA-2 (California), client on home Wi-Fi on the US east coast, 2026-11-15, one worker reached two ways
in alternating blocks of 50 requests on the same link (40 rounds, 1000 requests per path and pacing): through its
direct port with `connect(..., hot=True)` (`pgs up --direct`), and through Runpod's gateway (the default). The link was
clean (under 2 % late router echoes in all but 2 blocks). 95 % intervals from a block bootstrap.

| Path | p50 back to back (ms) | p95 | p99 | p50 at 5 per second | p95 | p99 | model time p50 | network part p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct port, `hot=True` | 111.6 | 119.5 | 125.2 | 111.7 | 119.8 | 128.0 | 31.7 | 79.8 |
| Gateway (default) | 124.8 | 133.8 | 187.7 | 125.2 | 138.7 | 200.3 | 33.9 | 91.0 |

- Direct port with `hot` against the gateway, paired by round: 13.2 ms less at p50 back to back (12.3 to 14.1) and
  13.5 ms at 5 per second (12.8 to 14.1), in 20 of 20 rounds each. About 11 ms of it is the path (the gateway adds a
  Cloudflare edge and Runpod's load balancer; TCP connect time to the worker's own address was 75 ms, a small message
  through the gateway took 84 ms), 2.3 ms is `hot` (model time 31.7 against 33.9 to 34.0 ms).
- Why `hot` helps: at a remote client's pace the worker waits about 80 ms between requests, and the next request
  then runs slower than back to back: 1.5 ms in the CUDA graph (the GPU slows down when it is mostly idle; a small
  kernel run back to back in the gaps prevents it, one kernel per millisecond or a 64 x 64 one did not hold at 5
  requests per second) and 0.8 ms in the host-side steps (the server's thread polls instead of sleeping).
- The first request after 15 to 40 s without one took 111.8 and 114.4 ms on the direct port (p50 of 20 each).
- Loss on the direct path: in four earlier runs of 4 to 7 minutes, 0.3 to 2 % of the direct requests took about
  385 ms: the reply is one packet, and when it is lost the worker's TCP waits a whole retransmission timeout (275
  ms). In the run above it happened once in 2000 requests. A ping frame sent behind every reply (`rs_probe` in the
  URL) cut such requests to about 205 ms but made every round trip 2.5 ms longer, so it is off.
- A later start of a worker hung on its first read of the network volume in US-CA-2 in 2 of 5 release changes that
  day (12 minutes once, never ready once); a first start on a new volume took 1:15 to 3:54.

## Model time per GPU

Model time = observation in to actions out on the GPU worker, without the network.

| GPU | Stock openpi (ms, p50) | Fast path (ms, p50) | Where measured |
|---|---:|---:|---|
| RTX 4090 | 75 to 79 | 31 to 34 | served by `pgs` (server `infer_ms`) |
| A40 | 145.5 | not available | served by an earlier worker build |
| RTX 5090 | 61.3 | 24.1 | Runpod pod, in-process only |
| L40S | 72.5 | 33.2 | Runpod pod, in-process only |
| H100 80GB | 52.1 | 20.6 | Runpod pod, in-process only |

- `pgs` uses the RTX 4090 pool first and the A40 pool second. The fast path runs only on the RTX 4090. The RTX 5090,
  L40S and H100 rows are not reachable through `pgs` today; they are listed for comparison.
- Runpod list prices read 2026-11-13, per hour while a worker runs: RTX 4090 1.34 USD, A40 1.22 USD.

## Same policy on the fast path

The fast path uses the same checkpoint, the same bf16 weights and the same 10 flow steps as stock openpi. Differences
in the actions, absolute values, at the same noise seed:

| Comparison | Mean | Max |
|---|---:|---:|
| Fast path against stock openpi, same input | 0.0014 | 0.018 to 0.019 |
| JPEG q90 input against the raw image, fast path | 0.0050 | 0.044 to 0.045 |
| Stock openpi against itself with a new noise draw | 0.036 | 2.0 |

Every worker start runs a self-test: a stored real frame must reproduce stock openpi's stored actions within 0.02 mean
and 0.1 max, or the worker serves stock openpi instead. In the measured starts the self-test gave 0.0022 mean and
0.011 max.
