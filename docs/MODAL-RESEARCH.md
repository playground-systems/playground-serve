# Runpod and Modal for Playground Serve

Research as of 2026-11-17; repository at `1194dd24`. This is a documentation and repository review, not a Modal deployment or an independent benchmark. Prices are public list prices and should be checked against the account and availability before deployment.

## Recommendation

Keep Runpod as the default for now. Test Modal as an optional backend, initially for startup time and higher-memory GPUs. Modal does not offer the RTX 4090 that most current robot policies use, so changing providers also changes GPU and potentially the tested fast path. Modal could produce lower costs for very short, disconnected sessions or for an 80 GB GPU, and its GPU snapshots could shorten initialization, but none of these outcomes has been measured for this project.

The main potential improvement is not raw π0.5 GPU speed. The [project's benchmark](BENCHMARKS.md) recorded 31–34 ms for the shipped 4090 fast path versus 33.2 ms for an L40S fast path tested in-process on Runpod. The packaged L40S path was not validated; these are not a controlled provider comparison. The same benchmark's H100 fast-path result was 20.6 ms in-process, but that would need integration and validation on Modal. Project fast paths are model- and GPU-specific and fall back to stock where unsupported ([MODELS.md](../MODELS.md)).

## Price comparison

Prices below are per running GPU-hour, not per model inference. Modal rates are GPU **only**; CPU, RAM, storage, egress and location fees may be added. Runpod's [Serverless pricing](https://docs.runpod.io/serverless/pricing) describes Flex workers as scale-to-zero and billed from start to full stop, including initialization and idle timeout. The [Runpod Serverless product page](https://www.runpod.io/product/serverless) and [Modal pricing](https://modal.com/pricing) give these displayed list rates:

| GPU | Runpod Serverless Flex | Modal GPU component | Qualification |
| --- | ---: | ---: | --- |
| RTX 4090, 24 GB | $1.34 | Not offered | Project's default for most robot models |
| A40/A6000, 48 GB pool | $1.22 | A40 not offered | Project's Flex-π default |
| L40/L40S/6000 Ada pool, 48 GB | $1.89 | L40S ~$1.95 | Runpod pool can contain different hardware |
| A100, 80 GB | $2.99 | ~$2.50 | GPU variant and other resources may differ |
| H100, 80 GB | $5.89 | H100 SXM5 ~$3.95 | GPU variant and other resources may differ |
| H200, 141 GB | $6.99 | H200 SXM ~$4.54 | GPU variant and other resources may differ |
| L4, 24 GB | L4/A5000/3090 pool $0.75 | ~$0.80 | Model compatibility and speed unmeasured |

Runpod's general pricing page has had different displayed Serverless values from the product page; use the live API/console before acting. The project prints live prices with `pgs up`, and its docs record list prices from earlier dates. Modal's [GPU catalog](https://modal.com/docs/guide/gpu) can automatically upgrade an H100 request to an H200; ensure the actual device and price behavior are recorded in experiments.

Modal additionally bills CPU at $0.04716 per physical core-hour and memory at $0.007992 per GiB-hour, based on the higher of the amount requested or used; physical cores are approximately two vCPUs. As an illustrative reservation, one core and 32 GiB add ~$0.30/hour, making the L40S ~$2.25/hour before region and other costs. This is not a measurement of this workload's resource consumption. A specified compute region also incurs a 1.15× multiplier for a broad region such as `us`, or 1.75× for a narrow region such as `us-east`; the multiplier applies to compute resource costs. Routing region and container region are distinct. Sources: [Modal pricing](https://modal.com/pricing), [resource configuration](https://modal.com/docs/guide/resources), [region selection](https://modal.com/docs/guide/region-selection).

Both providers charge while a GPU worker is running, including warm idle time. The project's default is five minutes after the last client disconnects, while Modal Functions default to up to 60 seconds after the last input and allow a configurable window; an open Modal WebSocket counts as one ongoing function call, so the 60-second window does not start during a live connection. For a purely illustrative isolated session, ignore startup and CPU/RAM, assume five minutes connected and no further activity, and compare 10 billed minutes on a $1.34/hour Runpod 4090 ($0.223) with six billed minutes on a $1.95/hour Modal L40S ($0.195). This is a scenario, not a quote: startup, connection lifetime, worker teardown, account credits, region, and CPU/memory can reverse it. The Runpod idle window is adjustable too. Sources: [project usage](USAGE.md), [Modal Web Functions](https://modal.com/docs/guide/webhooks), [Modal cold starts](https://modal.com/docs/guide/cold-start).

Storage matters when usage is low: Runpod lists network volumes at $0.07/GB-month below 1 TB, and project docs estimate $2.80/month for a 40 GB volume. Modal lists volumes at $0.09/GiB-month with 1 TiB-month included; its Starter plan includes $30/month in compute and 1 TiB/month egress. Allowances are shared across a workspace and may change. Sources: [Runpod billing](https://docs.runpod.io/serverless/pricing), [Modal pricing](https://modal.com/pricing), [project README](../README.md).

## Startup and network

For `pi05_libero`, the repository measured nine first starts at 60.3–118.3 seconds (median 64.6), and three later starts from the filled volume at 61.4, 73.6 and 88.8 seconds. Worker initialization, checkpoint download, GPU compilation, scheduling and gateway delay all matter. These are small samples, not claims about all model families or providers. [BENCHMARKS.md](BENCHMARKS.md) describes the build and timing conditions.

Modal [memory snapshots](https://modal.com/docs/guide/memory-snapshots) may skip imports, compilation and warmup. GPU snapshots include CUDA state and model weights but are alpha, can require code changes, can conflict with `torch.compile`, and generally do not help when the bottleneck is reading model weights from storage; snapshot creation and restore must be measured. There is no evidence yet that the project's CUDA graphs, Triton kernels, self-tests and setup restore correctly, or that snapshots are available for every candidate deployment style. Compare fresh initialization and repeated restores, including actual billable time and correctness. Modal's [cold-start guide](https://modal.com/docs/guide/cold-start) also documents prebuilt images and the cost of retaining warm capacity.

Existing π0.5 measurements on an east-coast Wi-Fi client were 61.1 ms round-trip p50 and 68.8 ms p95 for a 4090 worker in North Carolina; the model itself took about 31–34 ms. A California Runpod worker measured 111.6 ms p50 direct versus 124.8 ms through the gateway, with `hot=True` also contributing to the difference. Different routes, GPU hosts and fast-path availability can easily outweigh provider-specific overhead. Modal's default Function ingress is Virginia, while execution may be elsewhere; restricting compute region raises cost. Benchmark from the robot's real network and log both proxy and container region ([project benchmark](BENCHMARKS.md), [Modal region selection](https://modal.com/docs/guide/region-selection)).

## Integration and reliability

The project currently creates Runpod endpoints and volumes and uses the Runpod load balancer and a websocket protocol; a Modal backend requires deployment, state, auth, volume, status and lifecycle integration ([USAGE.md](USAGE.md), [WIRE.md](WIRE.md)). Modal [Web Functions](https://modal.com/docs/guide/webhooks) support WebSockets, with one function call per connection and a 2 MiB WebSocket message limit. Confirm large camera observations fit, configure the Function timeout (default five minutes, maximum 24 hours), and verify reconnection behavior with long robot sessions ([timeouts](https://modal.com/docs/guide/timeouts)). Multiple clients per GPU require deliberate concurrency and correctness checks.

Modal [Servers](https://modal.com/docs/guide/servers) are another possible integration: on scale-from-zero their first requests receive 503 while triggering startup, so clients must retry. Server [sticky sessions](https://modal.com/docs/guide/sticky-sessions) can retain container affinity on reconnect while the session is alive, but lifetime and billing require explicit testing. Function and Server behavior should not be conflated.

[Modal Functions](https://modal.com/docs/guide/preemption) are preemptible; the nonpreemptible Function option is unavailable for GPU Functions. A robot connection can therefore be interrupted and must reconnect safely; establish model and action-state behavior after a restart. This source specifically describes Functions; check Server guarantees separately rather than assuming they match. Neither provider's runtime replaces robot-side safeguards.

## Proposed evaluation

1. Start with `pi05_libero`, same checkpoint, bf16 precision, flow steps, observations, client, JPEG encoding, and fast-path action validation. Add `molmoact2_droid` or `flex_pi_libero` if higher-memory deployments matter. Record actual GPU type and software versions.
2. Measure 10 or more fresh starts and 10 or more restarts per configuration, separating scheduling, image and checkpoint load, model compile, self-test, readiness and client first action. Compare Modal without and with snapshots; verify restored outputs and failure recovery.
3. From the same robot network, take at least 1,000 warm requests at back-to-back and 5 Hz, reporting p50/p95/p99 for end-to-end and model time. Alternate providers where practical; include connection failures, location, image sizes, and direct versus gateway paths.
4. Record actual CPU/RAM, region multiplier, storage, billed worker lifetime, concurrent sessions and invoice/usage, then compare per session and per robot-hour at real duty cycles. Include interruptions, reconnects, idle timeout and cold-start charges. Decide whether the measured improvement justifies maintaining a second provider.
