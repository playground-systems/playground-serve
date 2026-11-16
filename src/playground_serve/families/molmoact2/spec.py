"""Registry entries of the MolmoAct2 family (Ai2) and an example observation per entry for `pgs demo`."""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec
from playground_serve.families.molmoact2.molmoact2_variants import VARIANTS

# Volume: environment tar (Python 3.12, torch 2.11 cu128, transformers 4.57.1) about 7 GB (estimate) + one checkpoint,
# 21.78 GB of float32 safetensors (Hugging Face API) = about 29 GB; 50 GB leaves room. GPU: bfloat16 weights take about
# 11 GB (the card: "under 16GB"), so the RTX 4090 first and the A40 second. RAM: transformers loads the float32 shards
# one by one into bfloat16 (about 16 GB at the peak, estimate); 24 GB keeps the RTX 4090 hosts eligible.
LICENSE = ("Apache-2.0 (code and weights); Ai2: for research and educational use under its Responsible Use "
           "Guidelines")
VALIDATED = dict.fromkeys(VARIANTS, "2026-11-16")  # entry -> date it served real actions on a GPU worker (NOTES.md)

SPECS = {name: ProposedSpec(name=name, family="molmoact2", checkpoint=f"hf://{v['repo']}@{v['revision']}",
                            volume_gb=50, min_container_ram_gb=24, gpu_pools=("ADA_24", "AMPERE_48"),
                            action_shape=(v["horizon"], v["action_dim"]), robot=v["robot"],
                            gpu_validated=VALIDATED.get(name, ""), license=LICENSE)
         for name, v in VARIANTS.items()}


def example_observation(name: str, rng: np.random.Generator | None = None) -> dict:
    """Random camera frames at 256 x 256 (LIBERO's render size; the other robots' frames may be any size), a random
    state and a task: the actions mean nothing, the shapes are real."""
    rng = rng or np.random.default_rng(0)
    v = VARIANTS[name]
    obs = {f"{cam}_cam": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for cam in v["cameras"]}
    return {**obs, "state": rng.normal(size=v["state_dim"]).astype(np.float32), "instruction": "pick up the object"}
