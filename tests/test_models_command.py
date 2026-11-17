"""`pgs models`: every registered entry with robot, GPU and hourly price, validated state and license note; read only,
works without a key (list prices, with a note) and reads live prices with one."""

from __future__ import annotations

import pytest

from playground_serve import cli, models
from playground_serve.runpod_api import ApiError


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no ~/.config/playground-serve/key either

    def no_network(*a, **kw):
        raise AssertionError("pgs models without a key must not call Runpod")

    monkeypatch.setattr(cli, "RunpodApi", no_network)


def rows(text: str) -> dict:
    return {line.split()[0]: line for line in text.splitlines()[1:] if line and line.split()[0] in models.names()}


def test_without_a_key(no_key, capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    table = rows(out)
    assert set(table) == set(models.names())
    assert "ran on GPU 2026-11-13" in table["pi05_libero"] and "untested" in table["pi0_droid"]
    assert "RTX 4090 (ADA_24)" in table["pi05_droid"] and "1.34" in table["pi05_droid"]
    assert "Gemma" in table["pi05_libero"]
    assert "list prices read 2026-11-13" in out
    for name in models.names():
        spec = models.get(name)
        assert spec.robot in table[name]
        status = ("answered a tool call " if spec.cpu else "ran on GPU ") + spec.gpu_validated if spec.gpu_validated else "untested"
        assert status in table[name]


class FakeCatalog:
    def __init__(self, key):
        assert key == "test-key"

    def serverless_gpus(self, min_cuda=None):
        return [{"pool": "ADA_24", "price": {"serverless": 1.11}}, {"pool": "ADA_24", "price": {"serverless": 1.5}},
                {"pool": "AMPERE_48", "price": {"serverless": 0.99}}]


def test_live_prices_with_a_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(cli, "RunpodApi", FakeCatalog)
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "1.50" in rows(out)["pi05_libero"] and "live Runpod serverless prices" in out


def test_live_read_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

    class Refusing(FakeCatalog):
        def serverless_gpus(self, min_cuda=None):
            raise ApiError(403, "GET", "/catalog/gpus", "forbidden")

    monkeypatch.setattr(cli, "RunpodApi", Refusing)
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "1.34" in rows(out)["pi05_libero"] and "list prices" in out
