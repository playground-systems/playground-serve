"""Registry entry for Flex-pi on LIBERO (ProposedSpec) and example observations (keys: serve.py).

License: MIT for code and weights; the DINOv3 License applies to the vision encoder the model loads (timm, not gated);
the VAE and T5 come from ModelScope (Wan2.2, Apache-2.0). See NOTES.md.
"""

from __future__ import annotations

import numpy as np

from playground_serve.families.common.proposed_spec import ProposedSpec
from playground_serve.families.flex_pi.fast import decode_observation
from playground_serve.families.flex_pi.serve import DEPTHS, IMAGE, LICENSE, STATE, VARIANTS, WRIST

FAMILY = "flex_pi"
# Volume: checkpoint 12.08 GB (torch pickle, HF API) + T5 umt5-xxl 11 GB + Wan2.2 VAE 1.4 GB + tokenizer (INSTALL.md
# §2.1) + DINOv3 ViT-B about 0.35 GB + environment tar 7.7 GB (measured) = about 33 GB; 60 GB leaves room. GPU: 25.4
# to 26.4 GB in use (measured), so the 24 GB RTX 4090 is out. Pools (served infer_ms, action-only, no depth, fast path;
# NOTES.md): A40 (AMPERE_48, 1.22 USD/hr) first, 131.9 ms (187.4 with depth); RTX 5090 (ADA_32_PRO, 1.89) second,
# 125.2 (136.8): its 6.7 ms less is smaller than the round trip a US user adds when it serves from Europe, the only
# place it had Serverless stock. The H100 (5.89) served 65.4 ms: `pgs up flex_pi_libero --gpu H100`. RAM: torch.load
# reads the 12 GB checkpoint on the CPU while T5's 11 GB loads (estimate).
HOSTED_IDS = {"flex_pi_libero": "flexpi-libero"}
GPU_VALIDATED = {"flex_pi_libero": "2026-11-16"}
EXPERIMENTAL = {"flex_pi_libero": False}

SPECS = {name: ProposedSpec(name=name, family=FAMILY, checkpoint=f"hf://{v['repo']}@{v['revision']}",
                            volume_gb=60, min_container_ram_gb=48, gpu_pools=("AMPERE_48", "ADA_32_PRO"),
                            action_shape=(v["horizon"], v["action_dim"]), robot="libero",
                            env={"DIFFSYNTH_MODEL_BASE_PATH": "/runpod-volume/diffsynth", "RS_FLEXPI_JOINT": "false",
                                 "TOKENIZERS_PARALLELISM": "false", "RS_FAST": "auto",
                                 "TORCHINDUCTOR_CACHE_DIR": "/runpod-volume/pgs/inductor_cache/flex_pi",
                                 "RS_SELF_FETCH": "1"},
                            gpu_validated=GPU_VALIDATED[name], hosted_id=HOSTED_IDS[name],
                            experimental=EXPERIMENTAL[name], license=LICENSE)
         for name, v in VARIANTS.items()}


def test_observations(name: str) -> list[dict]:
    """The recorded LIBERO observation of selftest_<name>.json (libero_spatial episode 0 frame 30), with depth and
    without."""
    import json
    import pathlib

    data = json.loads(pathlib.Path(__file__).with_name(f"selftest_{name}.json").read_text())
    return [decode_observation(data["observation"], depth=True), decode_observation(data["observation"], depth=False)]


def example_observation(name: str, rng: np.random.Generator | None = None, depth: bool = False) -> dict:
    """One observation as a LIBERO client would send it: 256 x 256 frames already rotated 180 degrees, the 8-D state,
    the task's language, and optionally both depth maps in uint16 millimetres."""
    rng = rng or np.random.default_rng(0)
    v = VARIANTS[name]
    obs = {IMAGE: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
           WRIST: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
           STATE: rng.normal(size=v["state_dim"]).astype(np.float32),
           "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate"}
    if depth:
        obs.update({key: rng.integers(300, 2000, (256, 256), dtype=np.uint16) for key in DEPTHS})
    return obs
