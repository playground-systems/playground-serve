"""Registry entry of the xiaomi_r1 family (Xiaomi-Robotics-1 on RoboCasa365) and an example observation. serve.py holds
the same constants for the worker (tests/families/test_xiaomi_r1.py checks that they agree); see serve.py for the keys.

License: Apache-2.0 for the code (github.com/XiaomiRobotics/Xiaomi-Robotics-1) and the weights (the Hugging Face repo's
license tag and LICENSE file); not gated. Self-hosting is allowed.
"""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec

COMMIT = "0dd7aef8dc87296246aae812a1f59ccb708e5546"
REPO, REVISION = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365", "3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
IMAGE_SIZE, STATE_DIM, ACTION_DIM, HORIZON, HZ, HISTORY, INTERVAL = (256, 256), 14, 12, 16, 20, 4, 2
LICENSE = "Apache-2.0 (code and weights)"

# Volume: checkpoint 10.12 GB (3 bf16 safetensors shards and the tokenizer; HF API) + the environment tar (8.2 GB
# measured): 40 GB. GPU pools, measured served infer_ms with the fast path (NOTES.md): RTX 5090 (ADA_32_PRO, 1.89
# USD/hr) 125.6 ms first, RTX 4090 (ADA_24, 1.34) 148.6 ms second, A40 (AMPERE_48, 1.22; stock only, not measured)
# last. The H100 (5.89) was 121.4 ms, not worth 3x the price; the L40S host measured was CPU-bound (623 ms).
# RAM: from_pretrained builds the bf16 model on the CPU before .cuda() (about 10 GB; estimate).
SPECS = {
    "xiaomi_r1_robocasa365": ProposedSpec(
        name="xiaomi_r1_robocasa365", family="xiaomi_r1", checkpoint=f"hf://{REPO}@{REVISION}", volume_gb=40,
        min_container_ram_gb=32, gpu_pools=("ADA_24", "ADA_32_PRO", "AMPERE_48"), action_shape=(HORIZON, ACTION_DIM),
        robot="robocasa365", env={"TOKENIZERS_PARALLELISM": "false", "RS_FAST": "auto"},
        gpu_validated="2026-11-16"),
}


def sample_history(history, length: int = HISTORY, interval: int = INTERVAL) -> np.ndarray:
    """The frames or states a request carries, from the robot's own history (oldest first): serve.sample_history,
    eval_robocasa365/entry.py:81-89. Keep the last (length - 1) * interval + 1 items per camera and for the state; at an
    episode's start the oldest one stands in for ticks that do not exist yet."""
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [max(0, len(items) - 1 - (length - 1 - index) * interval) for index in range(length)]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


def as_frames_map(stack: np.ndarray) -> dict:
    """A (T, H, W, 3) stack as {"0": frame, ..., "T-1": frame}: the form whose frames the client's jpeg mode encodes
    (it encodes H x W x 3 arrays inside maps)."""
    return {str(i): np.ascontiguousarray(frame) for i, frame in enumerate(stack)}


def example_observation(name: str, rng: np.random.Generator | None = None, ticks: int = 9) -> dict:
    """One observation after `ticks` control ticks: each camera's 256 x 256 frames and the 14-D states sampled with
    sample_history (4 ticks, 2 apart), and the task description."""
    if name not in SPECS:
        raise KeyError(name)
    rng = rng or np.random.default_rng(0)
    frames = {key: [rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(ticks)] for key in CAMERAS}
    states = [rng.normal(size=STATE_DIM).astype(np.float32) for _ in range(ticks)]
    obs = {key: sample_history(history) for key, history in frames.items()}
    return {**obs, "state": sample_history(states), "instruction": "Close the blender lid."}
