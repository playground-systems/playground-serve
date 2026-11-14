"""The models Playground Serve can serve: openpi configs with public checkpoints, plus any openpi config with your own
checkpoint. One worker (worker/) serves every entry from one environment. MODELS.md says how far each entry has been
checked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
# The environment tar on the volume holds a venv built on this image's Python 3.11; it works with this image only.
IMAGE = "runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404"
CHECKPOINTS = "gs://openpi-assets/checkpoints/"
# Runpod serverless pools, fastest first. pi05_libero measured 77 ms per inference on the RTX 4090 (ADA_24) against
# 146 ms on the A40 (AMPERE_48) (docs/BENCHMARKS.md). Every entry is a ~3B-parameter PaliGemma model whose
# bf16 weights take 6.7 GB of GPU memory, so the same pools fit all of them.
GPU_POOLS = ("ADA_24", "AMPERE_48")
# Runpod serverless list prices (USD per hour per running worker) and GPUs, read 2026-11-13 from the catalog
# (`/catalog/gpus`); `pgs models` shows these when it has no key to read the live ones.
POOL_PRICES = {"ADA_24": 1.34, "AMPERE_48": 1.22, "ADA_48_PRO": 1.89, "AMPERE_80": 2.99, "ADA_80_PRO": 5.89}
POOL_GPUS = {"ADA_24": "RTX 4090", "AMPERE_48": "A40", "ADA_48_PRO": "L40S", "AMPERE_80": "A100", "ADA_80_PRO": "H100"}
PRICES_READ = "2026-11-13"
# Environment tar 8.2 GB + one checkpoint (10.1 to 11.6 GiB, read from the public bucket) + tokenizer and JAX caches
# (a few MB) = about 21 GB. 40 GB is what pi05_libero ran with, and leaves room for a second model on the same volume.
VOLUME_GB = 40
# Not measured: an estimate for restoring a ~12 GB checkpoint into bf16 next to the jax and torch imports.
MIN_CONTAINER_RAM_GB = 16
OPENPI_LICENSE = "Apache-2.0 (openpi); PaliGemma base: Gemma terms"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    openpi_config: str
    checkpoint: str  # gs:// URI or an absolute path on the worker
    volume_gb: int
    min_container_ram_gb: int
    gpu_pools: list[str]
    image: str
    robot: str = ""  # libero, droid, aloha; "" for custom entries
    gpu_validated: str = ""  # the date real weights served a real action on a GPU worker; "" = untested
    license: str = ""


def _entry(name: str, robot: str, checkpoint: str | None = None, gpu_validated: str = "") -> ModelSpec:
    return ModelSpec(name=name, openpi_config=name, checkpoint=checkpoint or CHECKPOINTS + name, volume_gb=VOLUME_GB,
                     min_container_ram_gb=MIN_CONTAINER_RAM_GB, gpu_pools=list(GPU_POOLS), image=IMAGE, robot=robot,
                     gpu_validated=gpu_validated, license=OPENPI_LICENSE)


# Config names and checkpoint URIs from openpi at OPENPI_COMMIT: README.md "Fine-Tuned Models", the "Inference"
# configs in src/openpi/training/config.py, and DEFAULT_CHECKPOINT in scripts/serve_policy.py.
_REGISTRY = {spec.name: spec for spec in (
    _entry("pi05_libero", "libero", gpu_validated="2026-11-13"),
    _entry("pi05_droid", "droid"),
    _entry("pi0_fast_droid", "droid"),
    _entry("pi0_droid", "droid"),
    _entry("pi05_aloha", "aloha", checkpoint=CHECKPOINTS + "pi05_base"),  # serve_policy's --env ALOHA default
    _entry("pi0_aloha_towel", "aloha"),
    _entry("pi0_aloha_tupperware", "aloha"),
    _entry("pi0_aloha_pen_uncap", "aloha"),
    _entry("pi0_aloha_sim", "aloha"),
)}


def names() -> list[str]:
    return list(_REGISTRY)


def get(name: str) -> ModelSpec:
    """A registered model, or a custom one written CONFIG@CHECKPOINT (see custom)."""
    if name in _REGISTRY:
        spec = _REGISTRY[name]
        return dataclasses.replace(spec, gpu_pools=list(spec.gpu_pools))
    if "@" in name:
        return custom(*name.split("@", 1))
    raise KeyError(f"unknown model {name!r}: use one of {', '.join(_REGISTRY)}, or CONFIG@CHECKPOINT for your own")


def custom(openpi_config: str, checkpoint: str) -> ModelSpec:
    """Any openpi config name at OPENPI_COMMIT with your own checkpoint.

    `checkpoint` is a gs:// URI (the worker downloads it once into the volume) or an absolute path on the worker, for
    example /runpod-volume/checkpoints/my_run/30000. Either way it must hold openpi's checkpoint layout: params/ and
    assets/<asset id>/norm_stats.json. The config name is not checked here (this side has no openpi); a wrong one
    fails at the worker's policy load. The name holds a hash of both, so each pair gets its own JAX cache.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", openpi_config):
        raise ValueError(f"not an openpi config name: {openpi_config!r}")
    if not (re.match(r"[a-z][a-z0-9+.-]*://.", checkpoint) or checkpoint.startswith("/")):
        raise ValueError(f"checkpoint must be a URI such as gs://bucket/dir or an absolute path on the worker: {checkpoint!r}")
    digest = hashlib.sha256(f"{openpi_config}@{checkpoint}".encode()).hexdigest()[:8]
    return ModelSpec(name=f"custom-{openpi_config}-{digest}", openpi_config=openpi_config, checkpoint=checkpoint,
                     volume_gb=VOLUME_GB, min_container_ram_gb=MIN_CONTAINER_RAM_GB, gpu_pools=list(GPU_POOLS),
                     image=IMAGE, license=OPENPI_LICENSE)


def table(live_prices: dict | None = None) -> str:
    """`pgs models`: every registered entry with its first GPU pool and hourly price, whether it ran on a GPU, and
    its license note. `live_prices` (pool -> USD/hr, from the catalog) replaces the list prices of PRICES_READ."""
    rows = [("Model", "Robot", "GPU (pool)", "USD/hr", "Validated", "License")]
    for name in names():
        spec = get(name)
        pool = spec.gpu_pools[0]
        price = (live_prices or {}).get(pool, POOL_PRICES.get(pool))
        rows.append((name, spec.robot, f"{POOL_GPUS.get(pool, '?')} ({pool})",
                     "?" if price is None else f"{price:.2f}",
                     f"ran on GPU {spec.gpu_validated}" if spec.gpu_validated else "untested", spec.license))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1] for r in rows]
    note = ("Prices: live Runpod serverless prices per running worker." if live_prices else
            f"Prices: Runpod list prices read {PRICES_READ} (no Runpod key, or the live read failed; with a key "
            "`pgs models` reads the live ones).")
    return "\n".join(lines + ["", note, "A worker bills only while it runs; volumes bill per GB-month until "
                                      "`pgs delete MODEL --volume`. Details: MODELS.md."])
