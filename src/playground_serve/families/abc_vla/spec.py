"""Registry entries of the abc_vla family (ABC-VLA, Amazon FAR) and an example observation. serve.py holds the same
constants for the worker (tests/families/test_abc_vla.py checks that they agree); see serve.py for the request keys.

License: code Apache-2.0 (github.com/amazon-far/abc LICENSE). Weights: the checkpoint's sidecar says Apache-2.0 and
that the Gemma Terms of Use apply, its backbone being derived from Gemma 3; hosting it for yourself is allowed, and
Gemma's prohibited-use policy passes through to anyone the endpoint serves. Not gated; on the authors' own host.
"""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec

COMMIT = "d0832d12651d1b260a652861a14648dc5f3660c7"
CHECKPOINT = "https://abc-data.timehorizons.org/checkpoints/vla_abc130k_v2/200000.pt"
CAMERAS = ("top", "left", "right")
STATE_DIM, HORIZON, HZ = 14, 30, 30
LICENSE = "code Apache-2.0; weights Apache-2.0 with the Gemma Terms of Use (Gemma 3 backbone)"

# Volume: environment tar (torch 2.11 cu128 venv; NOTES.md has the measured size) + the 8.8 GB checkpoint, 30 GB.
# GPU: "inference needs a ~24 GB GPU" (deploy/README.md), so the 4090 pool first and the A40 second. RAM: the model is
# built in bf16 on the CPU (about 9 GB) and the checkpoint is memory-mapped.
SPECS = {
    "abc_vla_yam": ProposedSpec(name="abc_vla_yam", family="abc_vla", checkpoint=CHECKPOINT, volume_gb=30,
                                min_container_ram_gb=24, gpu_pools=("ADA_24", "AMPERE_48"),
                                action_shape=(HORIZON, STATE_DIM), robot="yam", env={"RS_FAST": "auto"},
                                gpu_validated="2026-11-16"),
}


def example_observation(name: str, rng: np.random.Generator | None = None) -> dict:
    """One bimanual YAM observation: three 480x640 RGB frames (the real cameras' size), joints, grippers, a prompt."""
    if name not in SPECS:
        raise KeyError(name)
    rng = rng or np.random.default_rng(0)
    images = {cam: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for cam in CAMERAS}
    state = np.concatenate([rng.normal(size=6) * 0.3, [0.5], rng.normal(size=6) * 0.3, [0.5]]).astype(np.float32)
    return {"images": images, "state": state, "prompt": "throw plastic bottles in bin"}
