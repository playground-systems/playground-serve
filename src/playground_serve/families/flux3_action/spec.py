"""Registry entry for FLUX 3 Action on DROID (ProposedSpec) and an example observation (keys: serve.py).

LICENSE, NON-COMMERCIAL: the weights are under the FLUX Kommunity License v1.0: non-commercial use; commercial use of
outputs only for a Qualifying User (gross annualized revenue with affiliates under US$5M). Both repositories are gated
on Hugging Face: the user accepts the license there and supplies HF_TOKEN. The text encoder is Qwen3-VL-4B-Instruct
(Apache-2.0); the code is Apache-2.0. Served and verified on an A40 with the official eager setting (NOTES.md).
"""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec
from playground_serve.families.flux3_action.serve import FRAME_HW, IMAGE_KEYS, LICENSE, VARIANTS

FAMILY = "flux3_action"
NON_COMMERCIAL = True
# Owner decision 2026-11-16: experimental.
EXPERIMENTAL = {"flux3_action_droid": True}
# Stated: "about 32 GB of GPU memory in bfloat16 on an NVIDIA H200" (model card); download sizes from the HF API:
# model.safetensors 13.9 GB + the base repo's text_encoder/ 8.9 GB + video_vae.safetensors 2.5 GB = 25.3 GB.
# Estimated: environment tar about 9 GB (torch 2.10 cu128) + 25.3 GB = about 35 GB, so 60 GB; RAM: the package loads on
# the CPU before moving to the GPU (DiT 13.9 GB, text encoder 8.9 GB), so 48 GB is a guess. Pool: the A40 (AMPERE_48,
# 1.22 USD/hr), the one card measured (33.6 GB of GPU memory in use, 3180 to 3228 ms served); the 32 GB RTX 5090 is
# too small, and no cheaper 48 GB card is listed.
HOSTED_IDS = {"flux3_action_droid": "flux-3-action-droid"}
# RS_COMPILE_DIT "0": the official eager setting (`serve-robolab --no-compile-dit`, the one the repo's replay-fixtures
# compares). The CLI's default, the compiled DiT with CUDA graphs, failed at every start on the A40 with this stack
# ("Inplace update to inference tensor outside InferenceMode" in CUDAGraph.capture_begin, during the official
# warm-up); serve.py falls back to eager if it is turned on and fails.
SPECS = {name: ProposedSpec(name=name, family=FAMILY, checkpoint=f"hf://{v['repo']}@{v['revision']}",
                            volume_gb=60, min_container_ram_gb=48, gpu_pools=("AMPERE_48",),
                            action_shape=(v["horizon"], v["action_dim"]), robot="droid",
                            env={"RS_COMPILE_DIT": "0", "RS_SELF_FETCH": "1"}, secrets=("HF_TOKEN",),
                            gpu_validated="2026-11-16", hosted_id=HOSTED_IDS[name], experimental=EXPERIMENTAL[name],
                            license=LICENSE)
         for name, v in VARIANTS.items()}


def test_observations(name: str) -> list[dict]:
    """The two recorded DROID observations of tests/families/data/droid_obs.json (verify_obs.json holds the first) at
    360 x 640, resized as the client does. Available in a checkout, not in an installed package."""
    import pathlib

    from playground_serve.families.flux3_action.serve import verify_observations

    assert name in VARIANTS
    return verify_observations(pathlib.Path(__file__).resolve().parents[4] / "tests" / "families" / "data"
                               / "droid_obs.json")


def example_observation(name: str, rng: np.random.Generator | None = None) -> dict:
    """One DROID observation as RoboLab sends it: three 360x640 RGB frames, 7 joints, gripper, an instruction."""
    rng = rng or np.random.default_rng(0)
    assert name in VARIANTS
    obs = {key: rng.integers(0, 256, (*FRAME_HW, 3), dtype=np.uint8) for key in IMAGE_KEYS}
    return {**obs, "observation/joint_position": rng.normal(size=7).astype(np.float32),
            "observation/gripper_position": rng.uniform(size=1).astype(np.float32),
            "prompt": "put the marker in the cup"}
