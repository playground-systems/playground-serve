"""Bring-up check of the Cosmos 3 policy worker (RS_VERIFY=1 only; bundle.py ships files named verify* only then):
the official RobolabPolicyService.infer against serve.py's wrapper on the stored observation, the noise floor, stock
against the fast path and the official inference knobs, per-phase times, and the self-test against every stored
reference. serve.load() calls run() and serves its Switch (stock, or the fast path per request with "rs_path")."""

from __future__ import annotations

import os
import time

import numpy as np


def free_cuda() -> None:
    import gc

    import torch

    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def run(name: str, log, fast, s):
    """Bring-up check (RS_VERIFY=1), logged: the official RobolabPolicyService.infer against this wrapper on the stored
    observation; stock against itself at a fixed seed and at a new seed (the noise floor); stock against the fast
    options; times. Serves the stock service afterwards. `s` is the serve module (serve.py runs as __main__)."""
    ref = fast.load_reference(name)
    obs = fast.decode_observation(ref["observation"])
    service = s.build_service(name)
    stock = s.Policy(name, service, "stock", "verify run (RS_VERIFY=1)")
    service._rng = np.random.default_rng(1234)
    official = np.asarray(service.infer(dict(obs))["action"], np.float32)  # robolab.py infer, seed from its RNG
    official_seed = int(np.random.default_rng(1234).integers(0, 2**31))
    ours = stock.infer({**obs, "rs_seed": official_seed})["actions"]

    def diff(a, b):
        d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
        return {"mean": round(float(d.mean()), 6), "max": round(float(d.max()), 5)}

    def timed(policy, seed, n=5):
        out, ms = None, []
        for _ in range(n):
            t = time.perf_counter()
            out = policy.infer({**obs, "rs_seed": seed})["actions"]
            ms.append(round((time.perf_counter() - t) * 1000, 1))
        return out, ms

    seed = int(ref["seed"])
    a_stock, ms_stock = timed(stock, seed)
    a_seed2, _ = timed(stock, seed + 1, 1)
    log("verify_official", official_seed=official_seed, official_vs_wrapper=diff(official, ours),
        official_actions=np.round(official, 6).tolist())
    log("verify_stock", seed=seed, ms=ms_stock, new_seed_vs_stock=diff(a_seed2, a_stock),
        stock_actions=np.round(a_stock, 6).tolist())
    log("verify_breakdown", **breakdown(stock, service, {**obs, "rs_seed": seed}))
    same, times = fast.sample_check(service, obs)
    quick = s.Policy(name, service, "fast", "", zero_tail=True)
    a_fast, ms_fast = timed(quick, seed, 8)
    log("verify_fast", variant="zero_tail", sample_equal=same, build_sample=times, ms=ms_fast,
        vs_stock=diff(a_fast, a_stock), breakdown=breakdown(quick, service, {**obs, "rs_seed": seed}))
    variants = () if os.environ.get("RS_VERIFY_QUICK") == "1" else (("frames_360x640", (360, 640)),)
    for label, frames in variants:  # the client's size for this model: no mosaic resize
        big = {**obs, **{k: np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(obs[k]).resize(
            frames[::-1])) for k in s.IMAGE_KEYS}}
        same_big, times_big = fast.sample_check(service, big)
        log("verify_fast", variant=f"zero_tail_{label}", sample_equal=same_big, build_sample=times_big,
            stock=breakdown(stock, service, {**big, "rs_seed": seed}),
            fast=breakdown(quick, service, {**big, "rs_seed": seed}),
            vs_stock=diff(quick.infer({**big, "rs_seed": seed})["actions"], stock.infer({**big, "rs_seed": seed})["actions"]))
    # The hosted service states "batched guidance, compiled language part" for this model: both are official knobs.
    knobs = (("batched_cfg", None, {"use_batched_cfg": True}),
             ("compiled_language", {"compiled_region": "language"}, {}))
    for label, overrides, kwargs in knobs if variants else ():
        try:
            svc = service if overrides is None else s.build_service(name, overrides)
            p = s.Policy(name, svc, "fast", "", zero_tail=True, generate_kwargs=kwargs)
            a_v, ms_v = timed(p, seed, 6)
            log("verify_variant", variant=label, ms=ms_v, vs_stock=diff(a_v, a_stock),
                breakdown=breakdown(p, svc, {**obs, "rs_seed": seed}))
            if overrides is not None:
                del svc, p
                free_cuda()
        except Exception as e:  # noqa: BLE001
            log("verify_variant_failed", variant=label, error=repr(e)[:600])
    try:  # against every stored reference: same GPU type, and how far other GPU types are
        for ref_gpu in ref["references"]:
            ok, detail = fast.selftest(quick, ref, ref_gpu)
            log("verify_selftest", gpu=s.gpu_name(), reference_gpu=ref_gpu, passed=ok, **detail)
    except Exception as e:  # noqa: BLE001
        log("verify_selftest", error=repr(e)[:400])
    return Switch(stock, quick)


class Switch:
    """Verify runs only (RS_VERIFY=1): serves the stock path, or the fast path when a request carries
    "rs_path": "fast", so that one worker measures both through the gateway."""

    def __init__(self, stock, quick):
        self.paths = {"stock": stock, "fast": quick}
        self.metadata = {**stock.metadata, "path": "stock", "path_reason": "verify run (RS_VERIFY=1): rs_path picks"}

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        return self.paths[str(obs.pop("rs_path", "stock"))].infer(obs)


def breakdown(policy, service, obs, n: int = 3) -> dict:
    """Median ms of the request's phases (CUDA synchronized): _build_sample, the model's _prepare_inference_data
    (vision encode, tokenization), the rest of generate_samples_from_batch (the sampling loop), _format_outputs."""
    import torch

    model, times = service.model, {}

    def wrap(obj, attr, key):
        fn = getattr(obj, attr)

        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            times.setdefault(key, []).append((time.perf_counter() - t) * 1000)
            return out
        setattr(obj, attr, timed)  # an instance attribute; popped again below

    for obj, attr, key in ((service, "_build_sample", "build_sample"), (model, "_prepare_inference_data", "prepare"),
                           (model, "generate_samples_from_batch", "generate"), (service, "_format_outputs", "format")):
        wrap(obj, attr, key)
    try:
        for _ in range(n):
            t = time.perf_counter()
            policy.infer(dict(obs))
            times.setdefault("total", []).append((time.perf_counter() - t) * 1000)
    finally:
        for name in ("_build_sample", "_format_outputs"):
            vars(service).pop(name, None)
        for name in ("_prepare_inference_data", "generate_samples_from_batch"):
            vars(model).pop(name, None)
    out = {k: round(float(np.median(v)), 1) for k, v in times.items()}
    out["sampling"] = round(out.get("generate", 0) - out.get("prepare", 0), 1)
    return out


def ready_now(pgs_server) -> None:
    """Verify runs only: health turns 200 now. A flex_pi verify worker whose health stayed 204 for about 9.5 minutes
    was restarted by Runpod in the middle of its checks; bring-up sends no request before the ready line anyway."""
    import pathlib

    pathlib.Path(pgs_server.READY_FILE).touch()
