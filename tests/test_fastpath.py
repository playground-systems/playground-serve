"""The fast model path's safety pieces, without a GPU: which path a worker picks, the boot self-test gate, why the stock
path serves after a fallback, and the prompt slots the CUDA graph reads (zeroed on every call)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from playground_serve.worker import fast_policy, fastpath, selftest

GPU = "NVIDIA GeForce RTX 4090"


@pytest.mark.parametrize("mode,model,gpu,ref,want", [
    ("auto", "pi05_libero", GPU, True, "fast"),
    ("off", "pi05_libero", GPU, True, "stock"),  # pgs up --stock-path
    ("auto", "pi05_droid", GPU, True, "stock"),  # no fast path for other models
    ("auto", "custom-pi05_libero-1a2b3c4d", GPU, True, "stock"),  # own checkpoint: the stored actions do not apply
    ("auto", "pi05_libero", "NVIDIA H200", True, "fast"),  # pgs up --gpu H200
    ("auto", "pi05_libero", "NVIDIA A40", True, "stock"),  # the second pool's card: not measured
    ("auto", "pi05_libero", "NVIDIA GeForce RTX 5090", True, "stock"),  # needs torch cu128: left out
    ("auto", "pi05_libero", "", True, "stock"),  # nvidia-smi said nothing
    ("auto", "pi05_libero", GPU, False, "stock"),  # no stored reference: never unchecked
])
def test_path_decision(mode, model, gpu, ref, want):
    path, reason = fastpath.decide(mode, model, gpu, ref)
    assert path == want and reason
    if want == "stock" and mode == "off":
        assert "--stock-path" in reason


def test_only_pi05_libero_on_the_measured_gpus_and_it_has_a_reference():
    assert fastpath.FAST == {"pi05_libero": (GPU, "NVIDIA H200")}
    assert fastpath.reference("pi05_libero").is_file()


def test_decide_cli_logs_and_leaves_the_reason(tmp_path):
    code = (f"import sys; sys.path.insert(0, {str(fastpath.HERE)!r}); import fastpath; "
            f"fastpath.PATH_REASON = {str(tmp_path / 'reason')!r}; sys.exit(fastpath.main(sys.argv[1:]))")
    run = lambda gpu, **env: subprocess.run([sys.executable, "-c", code, "decide", gpu], capture_output=True, text=True,
                                            env={"RS_MODEL": "pi05_libero", **env})
    fast = run(GPU)
    assert fast.returncode == 0 and json.loads(fast.stdout)["path"] == "fast"
    stock = run(GPU, RS_FAST="off")
    line = json.loads(stock.stdout)
    assert stock.returncode == 1 and (line["rs"], line["path"]) == ("path", "stock")
    assert (tmp_path / "reason").read_text() == line["reason"]


def test_fallback_reason_comes_from_the_fast_server_or_its_exit(tmp_path):
    reason = tmp_path / "failed"
    assert "exited with code 139" in fastpath.fallback_reason("139", str(reason))
    assert "could not fetch realtime-vla" in fastpath.fallback_reason("clone", str(reason))
    reason.write_text("fast path failed: self-test against stock openpi's stored actions failed: {...}")
    assert fastpath.fallback_reason("3", str(reason)).startswith("fast path failed: self-test")


# ----- the self-test gate: the stored stock actions, up to rounding, and nothing else

def test_selftest_reference_is_a_real_frame_with_stock_actions():
    ref = selftest.load("pi05_libero")
    obs = selftest.observation(ref)
    assert np.asarray(ref["actions"]).shape == (10, 7) and obs["rs_seed"] == ref["seed"]
    for key in ("observation/image", "observation/wrist_image"):
        assert obs[key][b"__jpeg__"] is True and obs[key][b"data"][:2] == b"\xff\xd8"  # JPEG bytes, as the client sends


def test_selftest_gate():
    ref = selftest.load("pi05_libero")
    stock = np.asarray(ref["actions"])
    rng = np.random.default_rng(0)
    assert selftest.check(stock, ref)[0]
    assert selftest.check(stock + rng.uniform(-0.01, 0.01, stock.shape), ref)[0]  # rounding (measured: max 0.011 to 0.018)
    assert not selftest.check(stock + 0.15, ref)[0]  # beyond rounding
    off = stock.copy()
    off[3, 6] += 2.0  # a flipped gripper
    assert not selftest.check(off, ref)[0]
    bad = stock.copy()
    bad[0, 0] = np.nan  # the slot bug's symptom
    assert selftest.check(bad, ref) == (False, {"error": "non-finite actions"})
    assert not selftest.check(stock[:, :6], ref)[0]


# ----- prompt slots: unused ones are zeroed on every call (masked, but 0 x NaN = NaN)

def test_unused_prompt_slots_are_zeroed_every_call():
    start, slots = 512, 32
    x = np.full((start + slots + 10, 4), np.nan, np.float32)  # whatever torch.empty left there
    long = np.ones((25, 4), np.float32)
    fast_policy.write_prompt(x, start, slots, long, 25)
    assert np.isfinite(x[start:start + slots]).all() and (x[start + 25:start + slots] == 0).all()
    x[start + 20:start + slots] = np.nan  # NaN written into the slots later (or left by an earlier process)
    short = np.full((20, 4), 2.0, np.float32)
    fast_policy.write_prompt(x, start, slots, short, 20)
    assert (x[start:start + 20] == 2.0).all() and (x[start + 20:start + slots] == 0).all()
    assert np.isnan(x[:start]).all()  # image token rows are not the prompt's business


def test_a_prompt_longer_than_the_slots_is_refused_not_cut():
    with pytest.raises(ValueError, match="--stock-path"):
        fast_policy.write_prompt(np.zeros((600, 4)), 512, 32, np.ones((33, 4)), 33)
