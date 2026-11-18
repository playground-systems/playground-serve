"""`pgs demo`: one dummy observation through a running endpoint, so a new user sees it work without a robot.

`example_obs()` is a random observation in the layout openpi's LIBERO policy takes (openpi's make_libero_example: two
224 x 224 RGB images, an 8-D state and a prompt), for LIBERO models such as pi05_libero. Like the client, this module
imports on Python 3.8.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

CALLS = 5  # timed calls after the first one


def example_obs(seed: int = 0) -> dict:
    """A random LIBERO observation (pi05_libero's input): the actions it gets back are meaningless, the shapes real."""
    rng = np.random.default_rng(seed)
    return {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.random(8),
        "prompt": "pick up the black bowl and place it on the plate",
    }


def run(target: str, key: str | None, out=print, connect=None, make_obs=None) -> None:
    """Connect to `target` (a model name or ws/wss URL, as `connect` takes it), send an example observation
    (make_obs(seed), default example_obs) once to wake the worker and CALLS more times, and print the actions' shape
    and the round trip."""
    if connect is None:
        from .resolve import connect
    make_obs = make_obs or example_obs
    out(f"Connecting to {target} (if no worker runs, one starts now: about 1 minute)...")
    t0 = time.monotonic()
    with connect(target, api_key=key) as policy:
        actions = policy.act(make_obs(0))
        first_s = time.monotonic() - t0
        rtt, infer = [], []
        for seed in range(1, CALLS + 1):
            policy.act(make_obs(seed))
            rtt.append(policy.timing["rtt_ms"])
            if policy.timing.get("infer_ms") is not None:
                infer.append(policy.timing["infer_ms"])
    out(f"Actions: shape {tuple(actions.shape)} (horizon x action size; the observation was random, so the values "
        "mean nothing).")
    out(f"First action after {first_s:.1f} s (includes connecting and any worker start).")
    out(f"Round trip over {CALLS} more calls: median {statistics.median(rtt):.0f} ms, max {max(rtt):.0f} ms"
        + (f"; model time on the server median {statistics.median(infer):.0f} ms." if infer else "."))
