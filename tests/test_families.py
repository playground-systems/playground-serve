"""Model families as first-class entries: the registry finds every families/<family>/spec.py, `pgs up` creates a
family's endpoint from its spec (pools, volume size, env map), and a broken family never stops the others."""

from __future__ import annotations

import ast
import dataclasses
import json
import importlib.util
import io
import subprocess
import sys
import tokenize
import warnings

import pytest

from playground_serve import bundle, models
from playground_serve.launcher import (DIRECT_KEY, DIRECT_PORT, MAX_REQUEST_BYTES, RELEASE, RELEASE_KEY, Launcher,
                                       LauncherError, release_id)
from test_launcher import DCS, GPUS, START, FakeApi

FAMILIES = sorted(p.parent.name for p in models.FAMILIES.glob("*/spec.py") if p.parent.name != "common")


def test_every_family_spec_imports_in_the_pgs_process():
    """Each spec.py imports with the package's own dependencies (no worker modules on sys.path), without warnings."""
    code = ("import warnings; warnings.simplefilter('error'); from playground_serve import models; "
            "print(' '.join(sorted({models.get(n).family for n in models.names()})))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == sorted({"openpi", *FAMILIES})


def test_openpi_first_and_names_by_family():
    names = models.names()
    assert names[:len(models.names("openpi"))] == models.names("openpi") and names[0] == "pi05_libero"
    assert sorted(names) == sorted(n for f in ["openpi", *FAMILIES] for n in models.names(f))
    assert len(names) == len(set(names))


@pytest.mark.parametrize("family", FAMILIES)
def test_family_bundle_and_env(family):
    module = importlib.import_module(f"playground_serve.families.{family}.spec")
    files = bundle.worker_files(family)
    cpu = any(getattr(s, "cpu", "") for s in (module.SPECS.values() if isinstance(module.SPECS, dict) else module.SPECS))
    # A CPU family (an LLM agent) installs its packages at start (start.sh) in place of a built environment.
    assert set(bundle.SHARED_FILES) | set(bundle.COMMON_FILES) | {"serve.py", "start.sh" if cpu else "build_env.sh"} <= set(files)
    assert not set(bundle.NOT_SHIPPED) & set(files)
    for proposed in (module.SPECS.values() if isinstance(module.SPECS, dict) else module.SPECS):
        spec = models.get(proposed.name)
        assert spec.family == family and spec.gpu_pools == list(proposed.gpu_pools)
        env = bundle.endpoint_env(spec, "r1")
        assert proposed.endpoint_env().items() <= env.items()  # the contract families/common/proposed_spec.py states
        assert env["RS_FAMILY"] == family and env["RS_FAST"] in ("auto", "off")
        assert not set(spec.secrets) & set(env)  # `pgs up` adds them from the user's machine (secret_env)
        assert bundle.unpack(env["RS_FILES"]) == bundle.worker_files(family, entry=spec.name)
        assert {n: d for n, d in files.items() if not n.startswith("selftest_")}.items() <= bundle.unpack(
            env["RS_FILES"]).items()


def test_unknown_family_files():
    with pytest.raises(ValueError, match="no worker family"):
        bundle.worker_files("nothing")
    with pytest.raises(ValueError, match="no worker family"):
        bundle.worker_files("common")


def test_a_broken_or_duplicate_family_is_left_out(tmp_path, monkeypatch):
    for family, text in {
        "aaa_broken": "raise RuntimeError('boom')\n",
        "bbb_dup": "from playground_serve.families.common.proposed_spec import ProposedSpec\n"
                   "SPECS = [ProposedSpec('pi05_libero', 'bbb_dup', 'hf://x/y@1', 10, 8, ('ADA_24',), (1, 1)),\n"
                   "         ProposedSpec('bbb_ok', 'wrong_folder_name', 'hf://x/y@1', 10, 8, ('ADA_24',), (2, 3),\n"
                   "                      gpu_validated=True)]\n",
    }.items():
        (tmp_path / family).mkdir()
        (tmp_path / family / "spec.py").write_text(text)

    def import_module(name):
        family = name.split(".")[2]
        spec = importlib.util.spec_from_file_location(name, tmp_path / family / "spec.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setattr(models, "FAMILIES", tmp_path)
    monkeypatch.setattr(models, "_FAMILY_SPECS", None)
    monkeypatch.setattr(models.importlib, "import_module", import_module)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        names = models.names()
    text = " ".join(str(w.message) for w in caught)
    assert "aaa_broken" in text and "boom" in text and "pi05_libero is already registered" in text
    assert names == models.names("openpi") + ["bbb_ok"]
    ok = models.get("bbb_ok")
    assert (ok.family, ok.action_shape, ok.gpu_validated, ok.openpi_config) == ("bbb_dup", (2, 3), "yes", "")
    assert models.get("pi05_libero").family == "openpi"
    monkeypatch.setattr(models, "_FAMILY_SPECS", None)  # the real families again for the tests after this one


def family_launcher(tmp_path, gpus=("ADA_24", "AMPERE_48")):
    api, out = FakeApi(), []
    launcher = Launcher(api, tmp_path / "state.json", start_command=START, out=out.append, sleep=lambda s: None,
                        clock=iter(range(0, 10**6)).__next__, poll=0, measure=api.measure, now=lambda: 1.8e9,
                        here=lambda: None)
    api.gpus = [g for g in GPUS if g["pool"] in gpus]
    return api, launcher, out


def test_every_pool_a_family_may_list_has_a_name_and_price():
    from playground_serve.families.common.proposed_spec import POOL_PRICES

    assert set(POOL_PRICES) <= set(models.POOL_GPUS) and set(models.POOL_GPUS) == set(models.POOL_PRICES)
    for name in models.names():
        assert all(p in models.POOL_GPUS for p in models.get(name).gpu_pools), name
    assert models.pool("RTX 5090") == "ADA_32_PRO" and models.pool("l40s") == "ADA_48_PRO"


@pytest.mark.skipif("molmoact2" not in FAMILIES, reason="no molmoact2 family")
def test_pgs_models_shows_each_family_entry_with_its_first_pool_and_price():
    table = models.table()
    row = next(line for line in table.splitlines() if line.startswith("molmoact2_droid "))
    assert "molmoact2" in row and "droid" in row and "RTX 4090 (ADA_24)" in row and "1.34" in row
    assert "Apache-2.0" in row and ("untested" in row or "ran on GPU" in row)
    live = models.table({"ADA_24": 1.5})
    assert "1.50" in next(line for line in live.splitlines() if line.startswith("molmoact2_droid "))


@pytest.mark.skipif("molmoact2" not in FAMILIES, reason="no molmoact2 family")
def test_up_with_gpu_takes_that_pool_for_a_family_entry(tmp_path):
    api, launcher, out = family_launcher(tmp_path, gpus=("ADA_24", "AMPERE_48", "ADA_80_PRO"))
    ep = launcher.up("molmoact2_droid", gpu=models.pool("H100"), dc="US-IL-1")["endpoint"]
    assert ep["gpu"]["pools"] == ["ADA_80_PRO"] and ep["dataCenterIds"] == ["US-IL-1"]
    assert "H100" in "\n".join(out) and "5.89 USD/hr" in "\n".join(out)
    # a later `pgs up` without --gpu keeps the endpoint's pool
    launcher.up("molmoact2_droid")
    assert api.endpoints[ep["id"]]["gpu"]["pools"] == ["ADA_80_PRO"]


@pytest.mark.skipif("molmoact2" not in FAMILIES, reason="no molmoact2 family")
def test_up_creates_a_family_endpoint_from_its_spec(tmp_path):
    api, launcher, out = family_launcher(tmp_path)
    api.dcs = DCS
    result = launcher.up("molmoact2_libero")
    ep = result["endpoint"]
    vol = result["volume"]
    assert "molmoact2-libero" in ep["name"] and ep["gpu"]["pools"] == ["ADA_24"]
    assert vol["size"] == 50 and ep["env"]["RS_FAMILY"] == "molmoact2"
    assert ep["env"]["RS_ENV_TAR"] == "/runpod-volume/pgs/molmoact2/env-secure-v2.tar"
    text = "\n".join(out)
    assert "model's Python environment" in text and "pgs demo molmoact2_libero" in text
    assert "example_obs" not in text  # openpi's LIBERO example is not this model's observation


def test_secrets_come_from_this_machine_at_up_and_are_never_saved(tmp_path, monkeypatch):
    secret = next((n for n in models.names() if models.get(n).secrets), None)
    if secret is None:
        pytest.skip("no entry with secrets")
    names = models.get(secret).secrets
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "no-token"))
    api, launcher, out = family_launcher(tmp_path)
    with pytest.raises(LauncherError, match=f"needs the secret {names[0]}"):
        launcher.up(secret)
    assert not [c for c in api.calls if c[0].startswith("create")]  # refused before anything was created
    (tmp_path / "no-token").write_text("hf-file-token-value\n")
    for name in names[1:]:
        monkeypatch.setenv(name, "env-secret-value")
    ep = launcher.up(secret)["endpoint"]
    if "HF_TOKEN" in names:
        assert ep["env"]["HF_TOKEN"] == "hf-file-token-value"
    text = "\n".join(out) + (tmp_path / "state.json").read_text()
    assert "hf-file-token-value" not in text and "env-secret-value" not in text


def test_an_experimental_entry_is_marked_in_models_and_at_up(tmp_path, monkeypatch):
    name = "molmoact2_libero" if "molmoact2" in FAMILIES else models.names()[-1]
    real = models._family_specs()
    monkeypatch.setattr(models, "_FAMILY_SPECS", {**real, name: dataclasses.replace(real[name], experimental=True)})
    row = next(line for line in models.table().splitlines() if line.startswith(name + " "))
    assert "; experimental" in row and not any("experimental" in line for line in models.table().splitlines()
                                                 if line.startswith("pi05_libero "))
    api, launcher, out = family_launcher(tmp_path)
    launcher.up(name)
    assert f"{name}: experimental: slow or not fully verified, see MODELS.md" in out
    from playground_serve.families.common.proposed_spec import ProposedSpec

    assert ProposedSpec("x", "f", "hf://a/b@c", 1, 1, ("ADA_24",), (1, 1)).experimental is False  # older specs


def test_a_start_over_the_limit_says_the_limit_and_how_to_retry(tmp_path):
    api, launcher, out = family_launcher(tmp_path)
    launcher.up("molmoact2_libero" if "molmoact2" in FAMILIES else models.names()[-1])
    ep = next(iter(api.endpoints.values()))
    with pytest.raises(LauncherError, match=r"the limit of `pgs up` for this start .* `pgs up molmoact2_libero` starts"):
        launcher._wait_ready(ep, timeout=-1, t0=0, want={"min": 0, "max": 1, "idleTimeout": 300}, release="other")


# Runpod refuses a create or update request it expands past 102400 bytes (HTTP 413; launcher.MAX_REQUEST_BYTES, measured
# 100,300 accepted and 100,400 refused). Every entry's largest `pgs up` request, with a 64-character value for each of
# its secrets, the direct port and the release id, and its bring-up verify run's (RS_VERIFY=1), stays this far under it.
MARGIN = 0.15


@pytest.mark.parametrize("name", models.names())
def test_every_entry_fits_runpods_request_limit_with_a_margin(name, tmp_path):
    spec = models.get(name)
    launcher = Launcher(FakeApi(), tmp_path / "state.json", start_command=list(bundle.START_COMMAND), out=print)
    runs = [spec] + ([dataclasses.replace(spec, env={**spec.env, "RS_VERIFY": "1"})] if spec.family != "openpi" else [])
    for run in runs:
        env = {**bundle.endpoint_env(run, RELEASE), **{k: "x" * 64 for k in run.secrets}, DIRECT_KEY: str(DIRECT_PORT)}
        env[RELEASE_KEY] = release_id(env)
        want = {"min": 1, "max": 1, "idleTimeout": 300}
        size = (len(json.dumps(launcher._agent_body(run, env, want)).encode()) if run.cpu else
                launcher._largest_request(name, run, env, GPUS, DCS, want))
        assert size <= (1 - MARGIN) * MAX_REQUEST_BYTES, (name, run.env.get("RS_VERIFY"), size)


def test_up_stops_before_anything_is_created_when_the_request_is_too_large(tmp_path):
    api, launcher, out = family_launcher(tmp_path)
    launcher.make_env = lambda spec, release: {**bundle.endpoint_env(spec, release), "RS_FILES": "A" * MAX_REQUEST_BYTES}
    name = models.names()[-1]
    with pytest.raises(LauncherError, match=rf"{name}: the endpoint request would be [0-9,]+ bytes, over the 100,300 "
                                            r"that Runpod accepts .* Nothing was created"):
        launcher.up(name)
    assert not [c for c in api.calls if c[0] in ("create_volume", "create_endpoint", "update_endpoint")]


def test_an_entry_ships_only_its_own_self_test_reference():
    for name in models.names():
        spec = models.get(name)
        if spec.family == "openpi":
            continue
        refs = {f for f in bundle.unpack(bundle.endpoint_env(spec, "r1")["RS_FILES"]) if f.startswith("selftest_")}
        assert refs <= {f"selftest_{name}.json"}, name
        on_disk = models.FAMILIES / spec.family / f"selftest_{name}.json"
        assert refs == ({on_disk.name} if on_disk.is_file() else set()), name


def _blank_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body and isinstance(
                body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body[0].value.value = ""
    return ast.dump(tree)


@pytest.mark.parametrize("family", FAMILIES)
def test_slim_files_are_the_same_code_line_for_line(family):
    """A family's shipped Python files (bundle.slim) parse to the source's AST but for the docstrings, keep every line
    number, and hold no comment; a file that reads __doc__ keeps its module docstring."""
    root = models.FAMILIES.parent
    for name, data in bundle.worker_files(family, verify=True).items():
        if not name.endswith(".py"):
            continue
        source = next(p for p in (models.FAMILIES / family / name, models.FAMILIES / "common" / name,
                                  root / "worker" / name) if p.is_file()).read_bytes()
        assert data.count(b"\n") == source.count(b"\n"), name
        assert _blank_docstrings(ast.parse(data)) == _blank_docstrings(ast.parse(source)), name
        assert not [t for t in tokenize.tokenize(io.BytesIO(data).readline) if t.type == tokenize.COMMENT], name
        if b"__doc__" in source:
            assert ast.get_docstring(ast.parse(data)) == ast.get_docstring(ast.parse(source)), name
    assert bundle.slim(b"x = (\n") == b"x = (\n"  # does not parse here: shipped as it is


def test_verify_files_ship_only_for_bring_up_runs(tmp_path, monkeypatch):
    """Files named verify* (bring-up checks) ship only for RS_VERIFY=1; dev tools never."""
    import pathlib

    real = pathlib.Path(bundle.__file__).parent
    root = tmp_path / "pkg"
    (root / "families" / "fam").mkdir(parents=True)
    (root / "worker").symlink_to(real / "worker")
    (root / "families" / "common").symlink_to(real / "families" / "common")
    for name in ("serve.py", "build_env.sh", "fast.py", "verify.py", "verify_obs.json", "bringup.py", "spec.py"):
        (root / "families" / "fam" / name).write_text(name)
    monkeypatch.setattr(bundle.resources, "files", lambda package: root)
    names = set(bundle.worker_files("fam"))
    assert {"serve.py", "fast.py"} <= names and not {"verify.py", "verify_obs.json", "bringup.py", "spec.py"} & names
    assert {"verify.py", "verify_obs.json"} <= set(bundle.worker_files("fam", verify=True)) - {"bringup.py"}
    spec = models.ModelSpec("x", "", "hf://a/b@c", 10, 8, ["ADA_24"], "img", family="fam", env={"RS_VERIFY": "1"})
    assert "verify.py" in bundle.unpack(bundle.family_env(spec, "r1")["RS_FILES"])


def test_min_cuda_raises_the_host_cuda_floor(tmp_path):
    from playground_serve.launcher import cuda_floor, spec_cuda

    image = models.get("pi05_libero").image
    assert cuda_floor(image) == "12.8" and cuda_floor(image, "13.0") == "13.0" and cuda_floor(image, "12.4") == "12.8"
    assert spec_cuda(models.get("pi05_libero")) == "12.8"
    needs = [n for n in models.names() if models.get(n).min_cuda]
    if not needs:
        pytest.skip("no entry with min_cuda")
    api, launcher, out = family_launcher(tmp_path, gpus=("ADA_24", "AMPERE_48", "ADA_80_PRO"))
    api.dcs = DCS
    spec = models.get(needs[0])
    ep = launcher.up(needs[0], gpu="AMPERE_48")["endpoint"]
    assert ep["gpu"]["minCudaVersion"] == spec.min_cuda == spec_cuda(spec) != "12.8"


def run_checkpoint_step(tmp_path, env):
    """The checkpoint part of families/common/boot.sh, with python3 replaced by a stub that logs its arguments and
    answers `stage.py check` with "not complete"."""
    text = (models.FAMILIES / "common" / "boot.sh").read_text()
    step = text[text.index("KEY=${RS_CHECKPOINT"):text.index("if [ $ENV_FROM = local ]")]
    calls = tmp_path / "calls"
    script = (f'python3() {{ echo "$*" >> {calls}; [ "$2" != check ]; }}\nENV_FROM=build\n' + step
              + 'wait\necho "dir=${RS_CKPT_DIR:-} volume=${RS_CKPT_VOLUME:-}"\n')
    out = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True,
                         check=True).stdout
    return out.strip(), (calls.read_text() if calls.exists() else "")


def test_self_fetch_skips_the_checkpoint_download(tmp_path):
    env = {"RS_CHECKPOINT": "hf://org/model@abc", "RS_WEIGHTS": "/vol/w"}
    out, calls = run_checkpoint_step(tmp_path, env)
    assert "stage.py fetch hf://org/model@abc /rs_local/weights/org/model@abc" in calls
    assert out == "dir=/rs_local/weights/org/model@abc volume=/vol/w/org/model@abc"
    (tmp_path / "calls").unlink()
    out, calls = run_checkpoint_step(tmp_path, {**env, "RS_SELF_FETCH": "1"})
    assert calls == "" and out == "dir= volume="  # no fetch, and neither stage.py fill nor run_worker sees a copy


@pytest.mark.skipif("llm_agent" not in FAMILIES, reason="no llm_agent family")
def test_up_serves_an_llm_agent_on_a_cpu_worker(tmp_path, monkeypatch):
    """`pgs up opus_5_5`: a load-balancing endpoint on a CPU worker with no volume, the family's start.sh, the model's
    name and the OpenRouter key (from this machine) in its env; `pgs models` lists the nine and nothing else."""
    api, launcher, out = family_launcher(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LauncherError, match="OPENROUTER_API_KEY"):
        launcher.up("opus_5_5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    result = launcher.up("opus_5_5")
    ep = result["endpoint"]
    assert result["volume"] is None and not api.volumes and ep["cpu"] == [{"id": "cpu3c", "vcpuCount": 2}]
    assert "gpu" not in ep and ep["image"] == "python:3.12-slim" and "/rs/start.sh" in " ".join(ep["cmd"])
    assert ep["env"]["RS_MODEL"] == ep["env"]["PGS_AGENT_MODEL"] == "opus_5_5" and ep["env"]["OPENROUTER_API_KEY"] == "sk-or-test"
    text = "\n".join(out)
    assert "anthropic/claude-opus-5.5" in text and "0.072 USD/hr" in text and "sk-or-test" not in text
    assert sorted(models.names("llm_agent")) == sorted(["fable_5_2", "opus_5_5", "gemini_4_carbon", "gemini_4_argon",
                                                        "gemini_3_8_flash_high", "gpt_6_astra", "gpt_6_1_sol",
                                                        "muse_wonder_1_0", "kimi_3_1"])
    row = next(line for line in models.table().splitlines() if line.startswith("kimi_3_1 "))
    assert "moonshotai/kimi-k3.1-pro" in row and "CPU (cpu3c, 2 vCPU)" in row
