"""Registry entries for the Cosmos 3 DROID policies (ProposedSpec) and an example observation (keys: serve.py).

License: OpenMDW-1.1 for code and weights (not gated; commercial use allowed), see NOTES.md.
"""

from __future__ import annotations

import pathlib

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec
from playground_serve.families.cosmos3_policy.fast import decode_observation
from playground_serve.families.cosmos3_policy.serve import FRAME_HW, IMAGE_KEYS, LICENSE, VARIANTS

FAMILY = "cosmos3_policy"
# Stated: repo sizes 9.15 GB (Edge) and 32.9 GB (Nano) (HF API usedStorage); parameters 3.86B / 15.75B; supported
# architectures Ampere, Hopper, Blackwell (model cards); Edge ran its [32, 8] setting on a 16 GB Jetson T2000; Nano
# tested on H100. Pools for Edge (served infer_ms, JPEG 360 x 640, NOTES.md): RTX 5090 (ADA_32_PRO, 1.89 USD/hr) first,
# fast path 439.5 ms median against 514.0 stock; A40 (AMPERE_48, 1.22) as the cheaper fallback, 839 / 853. The H100
# (5.89) served 235.3 / 274.1: `pgs up cosmos3_edge_droid --gpu H100`. Nano needs an 80 GB card: built and unit-tested,
# not run.
_POOLS = {"cosmos3_edge_droid": ("ADA_32_PRO", "AMPERE_48"),
          "cosmos3_nano_droid": ("AMPERE_80", "ADA_80_PRO")}
_VOLUME_GB = {"cosmos3_edge_droid": 50, "cosmos3_nano_droid": 80}
_RAM_GB = {"cosmos3_edge_droid": 32, "cosmos3_nano_droid": 64}
# The hosted service's ids for the same model and robot (docs, models page).
HOSTED_IDS = {"cosmos3_edge_droid": "cosmos3-edge-policy-droid", "cosmos3_nano_droid": "cosmos3-nano-policy-droid"}
GPU_VALIDATED = {"cosmos3_edge_droid": "2026-11-16", "cosmos3_nano_droid": ""}
# Owner decision 2026-11-16: both entries are experimental.
EXPERIMENTAL = {"cosmos3_edge_droid": True, "cosmos3_nano_droid": True}
# CUDA 13 wheels (torch 2.13+cu130): hosts must run driver >= 580.
MIN_CUDA = "13.0"

SPECS = {name: ProposedSpec(name=name, family=FAMILY, checkpoint=f"hf://{v['repo']}@{v['revision']}",
                            volume_gb=_VOLUME_GB[name], min_container_ram_gb=_RAM_GB[name], gpu_pools=_POOLS[name],
                            action_shape=(32, 8), robot="droid", env={"RS_FAST": "auto", "RS_SELF_FETCH": "1",
                                 "TORCHINDUCTOR_CACHE_DIR": "/runpod-volume/pgs/inductor_cache/cosmos3_policy"},
                            gpu_validated=GPU_VALIDATED[name], hosted_id=HOSTED_IDS[name],
                            experimental=EXPERIMENTAL[name], license=LICENSE, min_cuda=MIN_CUDA)
         for name, v in VARIANTS.items()}


# Recorded test data stays out of the worker bundle (the core ships every file of this folder): tests/families/data.
DROID_OBS = pathlib.Path(__file__).resolve().parents[4] / "tests" / "families" / "data" / "droid_obs.json"


def test_observations(name: str) -> list[dict]:
    """Two recorded DROID observations (tests/families/data/droid_obs.json: lerobot/droid_1.0.1 episode 1, frames 60
    and 120, the cameras at their recorded 180 x 320). Available in a checkout, not in an installed package."""
    import json

    assert name in VARIANTS
    data = json.loads(DROID_OBS.read_text())
    return [decode_observation(o) for o in data["observations"]]


def example_observation(name: str, rng: np.random.Generator | None = None) -> dict:
    """One DROID observation as RoboLab sends it: three 360x640 RGB frames, 7 joints, gripper, an instruction."""
    rng = rng or np.random.default_rng(0)
    assert name in VARIANTS
    obs = {key: rng.integers(0, 256, (*FRAME_HW, 3), dtype=np.uint8) for key in IMAGE_KEYS}
    return {**obs, "observation/joint_position": rng.normal(size=7).astype(np.float32),
            "observation/gripper_position": rng.uniform(size=1).astype(np.float32),
            "prompt": "Pick up the object and place it in the target container."}
