# Playground Serve wire rules

For a second client (another language, or a copy that cannot import `playground_serve`): what `src/playground_serve/client/client.py` does. The protocol is openpi's websocket policy server (commit 215abfb) plus two optional features. `tests/client/test_wire_doc.py` checks the table at the end against the code.

## Connect

- URL: `wss://<endpoint-id>.api.runpod.ai/ws`, as `pgs up` and `pgs status` print it (`ws://host:port` for a local
  server). The path reaches the server unchanged. The Python client also reads it from `URL_ENV`.
- Upgrade request header: `Authorization: Bearer <Runpod API key>`. Runpod's gateway checks it and removes it; the
  worker never sees it. A key that may call this endpoint is enough; it needs no right to read the account.
- No permessage-deflate, no frame size limit (a raw LIBERO observation is about 400 kB), TCP_NODELAY on.
- Answer the server's ping frames (every 20 s; libraries do). Pings do not keep a worker (measured through Runpod's
  gateway) and client pings are optional; the no-op below is the liveness check.

## Opening the socket: cold start and retries

An upgrade to an endpoint with no running worker starts one; the gateway holds the upgrade meanwhile. Connect to first
action: `COLD_WAIT_S` seconds for a fresh container (more on a new volume's first start), `WARM_WAIT_S` with a worker
running. A held upgrade is not reliably handed over at ready (502 or 400 came instead, up to 26 s later), so:

- Cap each attempt at `OPEN_TIMEOUT_S` seconds: the upgrade (TCP, TLS, HTTP), and again the wait for the metadata.
- Retry, until `WAIT_S` seconds after the first attempt, when an attempt
  - timed out, or the connection was refused, reset or cut off during the handshake;
  - got an HTTP status in `RETRY_STATUS`;
  - got HTTP 400 with `RETRY_400_BODY` in the body (any other 400 is final).
- Pause before the next attempt: none if the failed attempt lasted at least 90 % of its cap; otherwise `BACKOFF_S[0]`
  seconds, doubling after each such pause up to `BACKOFF_S[1]`.
- Final: 401 and 403 (the key may not call this endpoint), 404 and every other status, a failed DNS lookup.

## Messages

Every frame is binary msgpack-numpy, except a server error: a text frame with the Python traceback, then close code
1011; open a new socket for the next request.

1. Server, once, right after the upgrade: the metadata, a map (the policy's own metadata, often empty).
2. Client: one request, the observation map. Field names are the model's, as in openpi's example for its config;
   LIBERO: `observation/image` and `observation/wrist_image` (uint8 H x W x 3), `observation/state` (float), `prompt`.
3. Server: one reply map: `actions`, an array (horizon, action_dim), for example (10, 7); `server_timing`, a map with
   `infer_ms` (the policy call, JPEG decode included) and, from the second reply on a socket, `prev_total_ms` (the
   previous request's total server time); other keys the policy adds (openpi's `policy_timing`).
4. Repeat 2 and 3. One request in flight per socket: send the next request, no-op included, only after the reply to the previous one. For parallel requests open more sockets (scaling requestCount 4, as `pgs up` sets it, admitted three sockets at once on one worker).
5. Split requests, only when the features say `"split": true` and the mode is `jpeg`: open a second socket to the same URL with the query `rs_aux=<session id>` (`SPLIT_PATH`; any random string per connection; the server sends nothing on it, not even metadata). Send every second image map, largest first, as one frame on it and the rest of the observation as the request frame, both with the extra key `SPLIT_KEY`: [session id, request number]. The server merges the two maps (it waits up to 5 s for the second frame, then answers with an error). Each frame of a LIBERO request is then under 11 kB and fits the first flight of a TCP connection that was idle; a 19.7 kB frame sent 0.2 s or more after the previous one took one more round trip of the path through Runpod's gateway (about 70 ms to US-CA-2). If the second socket fails, send whole requests on the first.

msgpack-numpy, as openpi's `msgpack_numpy.py`: Python str is msgpack str, bytes is bin. A numpy array is the map
`{b"__ndarray__": True, b"data": <bin: the bytes in C order>, b"dtype": <str: numpy dtype.str such as "<f4" or "|u1">,
b"shape": [<int>, ...]}`; a numpy scalar is `{b"__npgeneric__": True, b"data": <the value>, b"dtype": <str>}`. The keys
of these two maps are bin, every other key is str. Object, void and complex dtypes are refused. When decoding, turn
each map that has the bin key `__ndarray__` back into an array; replies use the same encoding.

## Features in the metadata

A Playground Serve worker adds `"playground_serve": {"jpeg": true, "noop": true}` to the metadata (the old wire key
`"robo_serve"` is still accepted for compatibility); use the first key of `FEATURE_KEYS` whose value is a map. A stock openpi server sends neither: no jpeg
mode, no no-op. Read them again after every reconnect. For logs: `"path"` (fast, stock), `"path_reason"`, `"release"`.

## Images

Each uint8 array of shape (H, W, 3), at any depth, goes in one of `MODES`: the metadata's `"image_mode"` if it names
one (`jpeg` only if advertised), else `jpeg` if advertised, else `resize`. Target size H' x W': the metadata's
`"image_size"` ([H', W'] or one int), else `IMAGE_SIZE` square.

- `raw`: unchanged; the bytes equal openpi-client's.
- `resize`: openpi's `image_tools.resize_with_pad` to H' x W'. Unless already that size: r = max(W / W', H / H'); PIL
  bilinear resize to w = int(W / r) by h = int(H / r); paste on black at (int((W' - w) / 2), int((H' - h) / 2)).
- `jpeg`: resize as above, encode JPEG with PIL at quality `JPEG_QUALITY` (PIL defaults otherwise: baseline, 4:2:0),
  and send the map `{b"__jpeg__": True, b"data": <bin: the JPEG bytes>}` (bin key `JPEG_KEY`) in place of the array.
  The worker decodes it to RGB uint8 before the policy; any other image format is an error. A LIBERO request is then
  about 19 kB instead of 300 to 400 kB.

## Keep-alive and liveness

Only when the features say `"noop": true` (a stock server would run the policy on it and fail):

- After `KEEPALIVE_S` seconds with nothing sent on the socket, send `NOOP` (str key). The worker answers exactly
  `{"rs_noop": true}` and runs no inference. It is the one request in flight until its reply arrives.
- No reply within `NOOP_TIMEOUT_S` seconds, or any other reply: the worker is gone. Close the socket and connect again
  at once (a cold start). A socket whose worker was lost stays open, often for minutes and with no close frame, and
  messages on it get no reply, ever (measured through Runpod's gateway).

## Close codes and failures seen through the gateway

- 1000: closed by the client. 1011: server error, after the traceback text frame.
- 1006, no close frame: the worker's container was stopped, at once or minutes after the worker stopped answering.
- No reply to an inference within `RECV_TIMEOUT_S` seconds: close the socket and fail; do not resend (it may have run).
- Socket lost before a reply: connect again and resend the same request, at most `RETRIES` more times.

## Readiness without starting a worker

GET `HEALTH_URL` (endpoint id in the braces) with the bearer header and an explicit User-Agent: Runpod's API answers
`{"jobs": {...}, "workers": {"idle": n, "initializing": n, "ready": n, "running": n, "throttled": n, "unhealthy": n}}`
and starts no worker. `running` >= 1 is warm (counted from container start, so it may still be booting), else cold;
404: no such endpoint. Never probe the gateway (`https://<endpoint-id>.api.runpod.ai/...`): it starts a worker.

## Constants

| Constant | Value | Code |
|---|---|---|
| `OPEN_TIMEOUT_S` | `10` | `Policy(open_timeout=)` |
| `WAIT_S` | `600` | `Policy(wait=)` |
| `RETRY_STATUS` | `408, 429, 500, 502, 503, 504` | `client.RETRY_STATUS` |
| `RETRY_400_BODY` | `failed to assign worker` | `client.RETRY_400` |
| `BACKOFF_S` | `0.5, 10` | `client.BACKOFF_S` |
| `RECV_TIMEOUT_S` | `300` | `Policy(recv_timeout=)` |
| `RETRIES` | `2` | `Policy(retries=)` |
| `KEEPALIVE_S` | `20` | `Policy(keepalive=)` |
| `NOOP` | `{"rs_noop": true}` | `client.NOOP` |
| `NOOP_TIMEOUT_S` | `5` | `client.NOOP_TIMEOUT` |
| `FEATURE_KEYS` | `playground_serve, robo_serve` | `client.FEATURE_KEYS` |
| `MODES` | `raw, resize, jpeg` | `codec.MODES` |
| `IMAGE_SIZE` | `224` | `client.IMAGE_SIZE` |
| `JPEG_QUALITY` | `90` | `Policy(jpeg_quality=)` |
| `JPEG_KEY` | `__jpeg__` | `codec.JPEG_KEY` |
| `SPLIT_KEY` | `rs_split` | `client.SPLIT_KEY` |
| `SPLIT_PATH` | `rs_aux` | `client.SPLIT_PATH` |
| `HEALTH_URL` | `https://api.runpod.ai/v2/{}/health` | `resolve.HEALTH_URL` |
| `COLD_WAIT_S` | `40, 120` | `resolve.COLD_WAIT_S` |
| `WARM_WAIT_S` | `0, 1` | `resolve.WARM_WAIT_S` |
| `URL_ENV` | `PLAYGROUND_SERVE_URL` | `resolve.URL_ENV` |
