"""Registry lookups and custom entries (no openpi needed; see test_models_openpi.py for the checks against openpi)."""

import dataclasses
import pathlib

import pytest

from playground_serve import models

WORKER = pathlib.Path(models.__file__).parent / "worker"


def test_names_and_get():
    names = models.names()
    assert names[0] == "pi05_libero"
    assert len(names) == len(set(names))
    for name in names:
        spec = models.get(name)
        assert spec.name == name
        assert spec.checkpoint.startswith("gs://openpi-assets/checkpoints/")
        assert spec.gpu_pools and all(isinstance(p, str) for p in spec.gpu_pools)
        assert spec.robot in ("libero", "droid", "aloha")
        assert spec.image == models.IMAGE


def test_contract_fields():
    fields = [f.name for f in dataclasses.fields(models.ModelSpec)]
    assert fields[:7] == ["name", "openpi_config", "checkpoint", "volume_gb", "min_container_ram_gb", "gpu_pools", "image"]
    spec = models.get("pi05_droid")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "x"
    spec.gpu_pools.append("X")  # a copy: the registry is unchanged
    assert models.get("pi05_droid").gpu_pools == ["ADA_24", "AMPERE_48"]


def test_pi05_libero_is_todays_model():
    spec = models.get("pi05_libero")
    assert (spec.openpi_config, spec.checkpoint) == ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero")
    assert spec.volume_gb == 40 and spec.gpu_validated
    assert spec.image == "runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404"
    assert set(spec.gpu_pools) == {"ADA_24", "AMPERE_48"}  # the two pools it has run on (docs/BENCHMARKS.md)
    assert [n for n in models.names() if models.get(n).gpu_validated] == ["pi05_libero"]


def test_unknown_name():
    with pytest.raises(KeyError, match="CONFIG@CHECKPOINT"):
        models.get("pi05_nothing")


def test_custom():
    spec = models.custom("pi05_droid", "gs://my-bucket/runs/droid/30000")
    assert spec.openpi_config == "pi05_droid" and spec.checkpoint == "gs://my-bucket/runs/droid/30000"
    assert spec.name.startswith("custom-pi05_droid-") and spec.robot == "" and not spec.gpu_validated
    assert spec.gpu_pools == list(models.GPU_POOLS) and spec.image == models.IMAGE
    assert models.get("pi05_droid@gs://my-bucket/runs/droid/30000") == spec
    assert models.custom("pi05_droid", "gs://my-bucket/runs/droid/20000").name != spec.name
    path = models.get("pi0_aloha_towel@/runpod-volume/checkpoints/towel/9999")
    assert path.checkpoint == "/runpod-volume/checkpoints/towel/9999"


@pytest.mark.parametrize("config,checkpoint", [
    ("pi05 droid", "gs://b/c"), ("", "gs://b/c"), ("pi05_droid", "relative/dir"), ("pi05_droid", "gs://"),
    ("pi05_droid", "~/ckpt"),
])
def test_custom_rejects(config, checkpoint):
    with pytest.raises(ValueError):
        models.custom(config, checkpoint)


def test_build_env_pins_the_same_openpi_commit():
    assert f"OPENPI_COMMIT=${{OPENPI_COMMIT:-{models.OPENPI_COMMIT}}}" in (WORKER / "build_env.sh").read_text()


def test_gpu_option_names_a_pool():
    assert models.pool("H200") == models.pool("hopper_141") == "HOPPER_141" and models.pool("rtx 4090") == "ADA_24"
    assert models.POOL_PRICES["HOPPER_141"] == 6.99
    with pytest.raises(KeyError, match="H200 \\(HOPPER_141\\)"):
        models.pool("H900")
