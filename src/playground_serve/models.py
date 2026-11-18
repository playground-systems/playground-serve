"""The models Playground Serve can serve: openpi configs with public checkpoints (plus any openpi config with your own
checkpoint), and the entries of the other model families.

A family is the code an endpoint runs. "openpi" (worker/) serves every openpi entry from one environment. Every other
family is a folder families/<family>/ with spec.py (its entries: SPECS, see families/common/proposed_spec.py),
serve.py and build_env.sh; it builds its own environment and shares families/common/. The registry finds the folders
itself, so adding a family changes nothing here. MODELS.md says how far each entry has been checked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import pathlib
import re
import warnings

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
# The environment tar on the volume holds a venv built on this image's Python 3.11; it works with this image only.
IMAGE = "runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404"
CHECKPOINTS = "gs://openpi-assets/checkpoints/"
# Runpod serverless pools, fastest first. pi05_libero measured 77 ms per inference on the RTX 4090 (ADA_24) against
# 146 ms on the A40 (AMPERE_48) (docs/BENCHMARKS.md). Every entry is a ~3B-parameter PaliGemma model whose
# bf16 weights take 6.7 GB of GPU memory, so the same pools fit all of them.
GPU_POOLS = ("ADA_24", "AMPERE_48")
# Runpod serverless list prices (USD per hour per running worker) and GPUs, read 2026-11-13 from the catalog
# (`/catalog/gpus`; ADA_32_PRO read 2026-11-16); `pgs models` shows these when it has no key to read the live ones.
# A family's spec lists its pools in order of preference; `pgs up` takes the first with stock.
POOL_PRICES = {"ADA_24": 1.34, "AMPERE_48": 1.22, "ADA_48_PRO": 1.89, "ADA_32_PRO": 1.89, "AMPERE_80": 2.99,
               "ADA_80_PRO": 5.89, "HOPPER_141": 6.99}
POOL_GPUS = {"ADA_24": "RTX 4090", "AMPERE_48": "A40", "ADA_48_PRO": "L40S", "ADA_32_PRO": "RTX 5090",
             "AMPERE_80": "A100", "ADA_80_PRO": "H100", "HOPPER_141": "H200"}
PRICES_READ = "2026-11-13"
# Runpod serverless CPU prices, USD per vCPU-hour (region.PROBE_CPU's, read 2026-11-16): the LLM agents' workers.
CPU_PRICES = {"cpu3c": 0.036}
# Environment tar 8.2 GB + one checkpoint (10.1 to 11.6 GiB, read from the public bucket) + tokenizer and JAX caches
# (a few MB) = about 21 GB. 40 GB is what pi05_libero ran with, and leaves room for a second model on the same volume.
VOLUME_GB = 40
# Not measured: an estimate for restoring a ~12 GB checkpoint into bf16 next to the jax and torch imports.
MIN_CONTAINER_RAM_GB = 16
OPENPI_LICENSE = "Apache-2.0 (openpi); PaliGemma base: Gemma terms"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    openpi_config: str  # "" outside the openpi family
    checkpoint: str  # gs:// URI or an absolute path on the worker; hf://<repo>@<revision> or https:// for families
    volume_gb: int
    min_container_ram_gb: int
    gpu_pools: list[str]
    image: str
    robot: str = ""  # libero, droid, aloha, so101, yam, ...; "" for custom entries
    gpu_validated: str = ""  # the date real weights served a real action on a GPU worker; "" = untested
    license: str = ""
    family: str = "openpi"  # the worker code: openpi (worker/) or a folder of families/
    env: dict = dataclasses.field(default_factory=dict)  # extra endpoint env the family's serve.py reads
    secrets: tuple = ()  # env names filled from the user's Runpod secrets of the same name
    action_shape: tuple = ()  # (horizon, action_dim) of one reply, where the family states it
    experimental: bool = False  # slow or not fully verified: marked in `pgs models` and MODELS.md, noted at `pgs up`
    min_cuda: str = ""  # lowest host CUDA the family's environment needs, if above the image's (launcher.cuda_floor)
    cpu: str = ""  # a CPU worker (cpu3c-2) in place of gpu_pools: an LLM agent, whose model runs at its provider
    agent: dict = dataclasses.field(default_factory=dict)  # an LLM agent's route: provider, slug, title, prices


def _entry(name: str, robot: str, checkpoint: str | None = None, gpu_validated: str = "") -> ModelSpec:
    return ModelSpec(name=name, openpi_config=name, checkpoint=checkpoint or CHECKPOINTS + name, volume_gb=VOLUME_GB,
                     min_container_ram_gb=MIN_CONTAINER_RAM_GB, gpu_pools=list(GPU_POOLS), image=IMAGE, robot=robot,
                     gpu_validated=gpu_validated, license=OPENPI_LICENSE)


# Config names and checkpoint URIs from openpi at OPENPI_COMMIT: README.md "Fine-Tuned Models", the "Inference"
# configs in src/openpi/training/config.py, and DEFAULT_CHECKPOINT in scripts/serve_policy.py.
_REGISTRY = {spec.name: spec for spec in (
    _entry("pi05_libero", "libero", gpu_validated="2026-11-13"),
    _entry("pi05_droid", "droid", gpu_validated="2026-11-16"),  # stock path, RTX 4090 (MODELS.md)
    _entry("pi0_fast_droid", "droid"),
    _entry("pi0_droid", "droid"),
    _entry("pi05_aloha", "aloha", checkpoint=CHECKPOINTS + "pi05_base"),  # serve_policy's --env ALOHA default
    _entry("pi0_aloha_towel", "aloha"),
    _entry("pi0_aloha_tupperware", "aloha"),
    _entry("pi0_aloha_pen_uncap", "aloha"),
    _entry("pi0_aloha_sim", "aloha"),
)}


def pool(gpu: str) -> str:
    """The Runpod pool for `pgs up --gpu`: a pool id (HOPPER_141) or its GPU's name (H200), in any case."""
    for pool_id, name in POOL_GPUS.items():
        if gpu.strip().upper().replace(" ", "") in (pool_id, name.upper().replace(" ", "")):
            return pool_id
    raise KeyError(f"unknown GPU {gpu!r}: use one of {', '.join(f'{n} ({p})' for p, n in POOL_GPUS.items())}")


FAMILIES = pathlib.Path(__file__).resolve().parent / "families"
_FAMILY_SPECS: dict[str, ModelSpec] | None = None  # name -> entry of every family other than openpi, found once
_SPEC_MODULES: dict[str, object] = {}  # name -> its family's spec module (example_observation)


def _validated(value) -> str:
    """ProposedSpec.gpu_validated (a date, or a bool from before dates) as ModelSpec's date string."""
    if isinstance(value, str):
        return value
    return "yes" if value else ""


def _family_specs() -> dict[str, ModelSpec]:
    """Every entry of families/<family>/spec.py (SPECS: dict name -> ProposedSpec, or a list). A spec.py that does not
    import, or an entry whose name is taken, is left out with a warning; the rest of the registry still works."""
    global _FAMILY_SPECS
    if _FAMILY_SPECS is not None:
        return _FAMILY_SPECS
    found: dict[str, ModelSpec] = {}
    for path in sorted(FAMILIES.glob("*/spec.py")):
        family = path.parent.name
        if family == "common":
            continue
        try:
            module = importlib.import_module(f"playground_serve.families.{family}.spec")
            specs = module.SPECS
            entries = list(specs.values() if isinstance(specs, dict) else specs)
        except Exception as e:  # noqa: BLE001  (one broken family must not stop `pgs` for the others)
            warnings.warn(f"model family {family}: spec.py not loaded: {e!r}", stacklevel=2)
            continue
        for p in entries:
            if p.name in _REGISTRY or p.name in found:
                warnings.warn(f"model family {family}: {p.name} is already registered; left out", stacklevel=2)
                continue
            found[p.name] = ModelSpec(
                name=p.name, openpi_config="", checkpoint=p.checkpoint, volume_gb=int(p.volume_gb),
                min_container_ram_gb=int(p.min_container_ram_gb), gpu_pools=list(p.gpu_pools), image=p.image,
                robot=p.robot, gpu_validated=_validated(p.gpu_validated),
                license=getattr(p, "license", "") or getattr(module, "LICENSE", ""),
                family=family, env=dict(p.env), secrets=tuple(p.secrets), action_shape=tuple(p.action_shape),
                experimental=bool(getattr(p, "experimental", False)), min_cuda=getattr(p, "min_cuda", ""),
                cpu=getattr(p, "cpu", ""), agent=dict(getattr(p, "agent", {}) or {}))
            _SPEC_MODULES[p.name] = module
    _FAMILY_SPECS = found
    return found


def names(family: str | None = None) -> list[str]:
    """Registered model names, openpi's first; only those of `family` if one is given."""
    every = {**_REGISTRY, **_family_specs()}
    return [n for n, spec in every.items() if family is None or spec.family == family]


def get(name: str) -> ModelSpec:
    """A registered model, or a custom one written CONFIG@CHECKPOINT (see custom)."""
    spec = _REGISTRY.get(name) or _family_specs().get(name)
    if spec is not None:
        return dataclasses.replace(spec, gpu_pools=list(spec.gpu_pools), env=dict(spec.env), agent=dict(spec.agent))
    if "@" in name:
        return custom(*name.split("@", 1))
    raise KeyError(f"unknown model {name!r}: use one of {', '.join(names())}, or CONFIG@CHECKPOINT for your own")


def example_observation(name: str):
    """A dummy observation in the model's layout for `pgs demo`, from its family's spec.py
    (example_observation(name)), or None where the family has none. The openpi LIBERO one is demo.example_obs."""
    get(name)
    make = getattr(_SPEC_MODULES.get(name), "example_observation", None)
    return make(name) if make else None


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
    rows = [("Model", "Family", "Robot", "GPU (pool)", "USD/hr", "Validated", "License")]
    for name in names():
        spec = get(name)
        if spec.cpu:  # an LLM agent: a CPU worker; its tokens are billed by its provider
            flavor, vcpu = spec.cpu.split("-")[:2]
            rows.append((name, spec.family, spec.robot, f"CPU ({flavor}, {vcpu} vCPU)", f"{CPU_PRICES.get(flavor, 0) * int(vcpu):.3f}",
                         ("answered a tool call " + spec.gpu_validated) if spec.gpu_validated else "untested",
                         f"{spec.agent.get('title', '')} via OpenRouter ({spec.agent.get('slug', '')}); " + spec.license))
            continue
        pool = spec.gpu_pools[0]
        price = (live_prices or {}).get(pool, POOL_PRICES.get(pool))
        rows.append((name, spec.family, spec.robot, f"{POOL_GPUS.get(pool, '?')} ({pool})",
                     "?" if price is None else f"{price:.2f}",
                     (("ran on GPU" + ("" if spec.gpu_validated == "yes" else f" {spec.gpu_validated}"))
                      if spec.gpu_validated else "untested") + ("; experimental" if spec.experimental else ""),
                     spec.license))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1] for r in rows]
    note = ("Prices: live Runpod serverless prices per running worker." if live_prices else
            f"Prices: Runpod list prices read {PRICES_READ} (no Runpod key, or the live read failed; with a key "
            "`pgs models` reads the live ones).")
    return "\n".join(lines + ["", note, "A worker bills only while it runs; volumes bill per GB-month until "
                                      "`pgs delete MODEL --volume`. LLM agents (family llm_agent) run on a CPU worker "
                                      "and bill their tokens to the OPENROUTER_API_KEY they were started with. "
                                      "Details: MODELS.md."])
