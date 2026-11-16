"""Registry entries of the gr00t_n17 family (NVIDIA GR00T N1.7 on LIBERO) and an example observation per entry.
serve.py holds the same constants for the worker (tests/families/test_gr00t_n17.py checks that they agree).

`gr00t_n17_libero` ran on an RTX 4090 (stock and fast path, NOTES.md); the other three suites share its code and were
not run. The backbone the code loads by name at every start, nvidia/Cosmos-Reason2-2B, is gated on Hugging Face
(approval after accepting its terms): the entries need the secret HF_TOKEN from an account that accepted them.

License: code Apache-2.0; weights nvidia/GR00T-N1.7-LIBERO under the NVIDIA Open Model License (commercial use and
self-hosting allowed), not gated; the backbone under its own terms, accepted per account on Hugging Face.
"""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec

COMMIT = "3b49067943eb1ab2eb2203b93d461c4390c78da3"
REPO, REVISION = "nvidia/GR00T-N1.7-LIBERO", "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"
SUITES = {"gr00t_n17_libero": "libero_spatial", "gr00t_n17_libero_goal": "libero_goal",
          "gr00t_n17_libero_object": "libero_object", "gr00t_n17_libero_10": "libero_10"}
VIDEO_KEYS = ("image", "wrist_image")
STATE_DIMS = {"x": 1, "y": 1, "z": 1, "roll": 1, "pitch": 1, "yaw": 1, "gripper": 2}
PROMPT_KEY = "annotation.human.action.task_description"
HORIZON, ACTION_DIM, IMAGE_SIZE = 16, 7, (256, 256)
LICENSE = "code Apache-2.0; weights NVIDIA Open Model License; backbone nvidia/Cosmos-Reason2-2B gated (its own terms)"

# Volume: environment tar (estimate 9 GB: the locked wheels without TensorRT and deepspeed) + one suite's inference
# files 6.91 GB (HF API) + the backbone 4.89 GB in HF_HOME = about 21 GB; 40 GB. GPU: "1 GPU with 16 GB+ VRAM" for
# inference (README), so the 4090 pool first and the A40 second. RAM: 32 GB (estimate).
# HF_HUB_ENABLE_HF_TRANSFER=0: the stock image sets it to 1, and the backbone download then fails for want of the
# hf_transfer package (measured on the first boot).
SPECS = {
    name: ProposedSpec(name=name, family="gr00t_n17", checkpoint=f"hf://{REPO}@{REVISION}/{suite}", volume_gb=40,
                       min_container_ram_gb=32, gpu_pools=("ADA_24", "AMPERE_48"), action_shape=(HORIZON, ACTION_DIM),
                       robot="libero", secrets=("HF_TOKEN",),
                       env={"HF_HUB_ENABLE_HF_TRANSFER": "0", "RS_FAST": "auto",
                            # boot.sh's stage.py fetch takes only the files the policy reads (serve.CHECKPOINT_FILES)
                            "RS_CKPT_FILES": "config.json,embodiment_id.json,model-*.safetensors,"
                                             "model.safetensors.index.json,processor_config.json,statistics.json"},
                       gpu_validated="2026-11-16" if name == "gr00t_n17_libero" else "")
    for name, suite in SUITES.items()
}


def example_observation(name: str, rng: np.random.Generator | None = None) -> dict:
    """One LIBERO step as the official env sends it (libero_env.py:144-158): two 256x256 RGB frames, the eef state
    (position, axis-angle, two gripper finger positions) and the task description."""
    if name not in SPECS:
        raise KeyError(name)
    rng = rng or np.random.default_rng(0)
    obs = {f"video.{k}": rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8) for k in VIDEO_KEYS}
    obs.update({f"state.{k}": rng.uniform(-1, 1, d).astype(np.float32) for k, d in STATE_DIMS.items()})
    return {**obs, PROMPT_KEY: "pick up the black bowl between the plate and the ramekin and place it on the plate"}
