"""Bring-up check for Flex-pi (RS_VERIFY=1; /rs/verify.py on the worker): the evaluator's own functions against
serve.py's Policy on the same model and the same recorded LIBERO observation, then the fast options against stock.

The official side is experiments/libero/eval_libero_single.py (EVAL) at the pinned commit, imported from the checkout
with the simulator stubbed: LIBERO and robosuite are not installed on the worker. Only EVAL's three simulator reads
are replaced: get_libero_image (the client already sends the 180-degree-rotated frames it returns),
_extract_sim_state (the client sends the 8-D state it builds) and get_libero_per_cam_depth_uint16_mm (the client sends
the uint16 mm depth it returns). Everything else is EVAL's code: its hydra config (sim_libero.yaml, the launcher's
regime overrides, the saved-config swap), _obs_to_model_input, the infer_action keywords, _denormalize_action and the
gripper post-processing in _predict_action_chunk.
"""

from __future__ import annotations

import os
import sys
import time
import types

import numpy as np

REPO = os.environ.get("RS_REPO_DIR", "/opt/pgs/repo")  # the checkout build_env.sh makes


def ready_now(pgs_server) -> None:
    """Health turns 200 now: a verify worker whose health stayed 204 for about 9.5 minutes was restarted by Runpod
    in the middle of its checks. Bring-up sends no request before the ready line anyway."""
    import pathlib

    pathlib.Path(pgs_server.READY_FILE).touch()


def import_eval():
    """EVAL as a module, with the simulator imports stubbed."""
    import flexpi.utils.libero_setup as setup

    setup.prepare_libero = setup.patch_torch_load  # keep its torch.load patch, skip the LIBERO config binding
    setup.assert_mujoco_pin = lambda *a, **k: None
    for mod in ("libero", "libero.libero", "libero.libero.envs", "libero.libero.envs.bddl_utils",
                "libero.libero.benchmark"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    sys.modules["libero"].libero = sys.modules["libero.libero"]
    sys.modules["libero.libero"].envs = sys.modules["libero.libero.envs"]
    sys.modules["libero.libero"].get_libero_path = lambda *a, **k: ""
    sys.modules["libero.libero"].benchmark = sys.modules["libero.libero.benchmark"]
    sys.modules["libero.libero.envs"].OffScreenRenderEnv = object
    sys.modules["libero.libero.envs"].SubprocVectorEnv = object
    sys.modules["libero.libero.envs"].bddl_utils = sys.modules["libero.libero.envs.bddl_utils"]
    for path in (REPO, os.path.join(REPO, "experiments", "libero")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import experiments.libero.eval_libero_single as EVAL

    return EVAL


def eval_cfg(EVAL, name: str, joint: bool, depth: bool):
    """EVAL's config as scripts/eval_flexpi_libero_single.sh launches it for this checkpoint."""
    from huggingface_hub import hf_hub_download
    from hydra import compose, initialize_config_dir

    import serve

    v = serve.VARIANTS[name]
    ckpt = hf_hub_download(v["repo"], v["weights"], revision=v["revision"])
    stats = hf_hub_download(v["repo"], "dataset_stats.json", revision=v["revision"])
    intr = hf_hub_download(v["intrinsics"][0], v["intrinsics"][2], revision=v["intrinsics"][1], repo_type="dataset")
    flag = lambda b: "true" if b else "false"  # noqa: E731
    overrides = [f"ckpt={ckpt}", f"EVALUATION.dataset_stats_path={stats}", f"+EVALUATION.camera_intrinsics_path={intr}",
                 "EVALUATION.tshape_eval_resize=stretch", "+EVALUATION.infer_present_video=true",
                 "+EVALUATION.infer_present_dino=true", f"+EVALUATION.infer_present_pointmap={flag(depth)}",
                 f"+EVALUATION.infer_joint_video={flag(joint)}", f"+EVALUATION.infer_joint_dino={flag(joint)}",
                 f"+EVALUATION.infer_joint_pointmap={flag(joint)}"]
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    return EVAL._maybe_swap_eval_cfg_with_saved(cfg)


def official_actions(EVAL, cfg, model, processor, obs: dict, depth_hw) -> np.ndarray:
    """EVAL._predict_action_chunk on one request (the three simulator reads replaced, see the module docstring)."""
    EVAL.get_libero_image = lambda o: {"image": o["image"], "wrist_image": o["wrist_image"]}
    EVAL._extract_sim_state = lambda o: np.asarray(o["state"], np.float32)
    EVAL.get_libero_per_cam_depth_uint16_mm = lambda o, env: {"image": o["image_depth"],
                                                              "wrist_image": o["wrist_image_depth"]}
    sim_obs = {"image": obs["observation/image"], "wrist_image": obs["observation/wrist_image"],
               "state": obs["observation/state"]}
    if "observation/image_depth" in obs:
        sim_obs["image_depth"] = obs["observation/image_depth"]
        sim_obs["wrist_image_depth"] = obs["observation/wrist_image_depth"]
    else:  # EVAL always reads depth on this 3D config; with present_pointmap false it is not used
        sim_obs["image_depth"] = np.zeros((256, 256), np.uint16)
        sim_obs["wrist_image_depth"] = np.zeros((256, 256), np.uint16)
    h, w = (int(x) for x in cfg.data.train.video_size)
    action, _, _ = EVAL._predict_action_chunk(sim_obs, obs["prompt"], model, processor, cfg,
                                              action_horizon=int(cfg.data.train.num_frames) - 1, input_w=w,
                                              input_h=h, model_device="cuda", env=object())
    return np.asarray(action, np.float32)


def other_seed(policy, obs: dict, seed: int) -> np.ndarray:
    """The same call with another noise seed (EVAL uses 42 on every call): the size of a new noise draw."""
    import torch

    import serve

    kwargs = dict(policy.kwargs(dict(obs)), seed=seed)
    with torch.no_grad():
        pred = policy.model.infer_action(**kwargs)
    return serve.finish_actions(policy.ops.denormalize(pred["action"])[0])


_MISSING = object()
STAGES = (("text", None, "encode_prompt"), ("vae", None, "_encode_input_image_latents_tensor"),
          ("dino", None, "_encode_first_frame_dino"), ("pointmap", None, "_encode_first_frame_pointmap_raw"),
          ("pointmap_embed", None, "_embed_pointmap"), ("prefill", "mot", "prefill_video_cache"),
          ("denoise_step", None, "_predict_action_noise_with_cache"),
          ("prefill_and_loop", None, "_run_action_prefill_denoise_loop"))


def stage_profile(policy, obs: dict, n: int = 3) -> dict:
    """Median ms per stage of one call (each stage CUDA-synchronized before and after, so the sum may exceed the
    unprofiled time a little): CPU inputs, the model's stages (STAGES; a stage's calls per request are added up, and
    "calls" says how many), the rest of infer_action ("glue"), post-processing."""
    import torch

    import serve

    model = policy.model
    per_call: dict = {}
    counts: dict = {}
    wrapped = []
    for label, sub, attr in STAGES:
        owner = getattr(model, sub) if sub else model
        if not hasattr(owner, attr):
            continue
        fn = getattr(owner, attr)

        def timed(*a, _fn=fn, _label=label, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = _fn(*a, **k)
            torch.cuda.synchronize()
            per_call[_label] = per_call.get(_label, 0.0) + (time.perf_counter() - t) * 1000
            counts[_label] = counts.get(_label, 0) + 1
            return out

        wrapped.append((owner, attr, vars(owner).get(attr, _MISSING)))  # compile_encoders may have set one
        setattr(owner, attr, timed)
    rows: dict = {}
    try:
        for _ in range(n):
            per_call.clear()
            counts.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            kwargs = policy.kwargs(dict(obs))
            if policy.text_cache is not None:  # as Policy.infer does on the fast path
                prompt = kwargs.pop("prompt")
                if prompt not in policy.text_cache:
                    policy.text_cache[prompt] = model.encode_prompt(prompt)
                context, mask = policy.text_cache[prompt]
                kwargs.update(context=context.clone(), context_mask=mask.clone())
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                pred = model.infer_action(**kwargs)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            serve.finish_actions(policy.ops.denormalize(pred["action"])[0])
            t3 = time.perf_counter()
            stage = dict(per_call)
            stage["glue"] = (t2 - t1) * 1000 - sum(per_call.values())
            stage.update(inputs=(t1 - t0) * 1000, post=(t3 - t2) * 1000, total=(t3 - t0) * 1000)
            for k, v in stage.items():
                rows.setdefault(k, []).append(v)
    finally:
        for owner, attr, before in wrapped:
            if before is _MISSING:
                vars(owner).pop(attr, None)
            else:
                setattr(owner, attr, before)
    out = {k: round(float(np.median(v)), 1) for k, v in rows.items()}
    out["calls"] = dict(counts)
    return out


def reset_compiled(model) -> None:
    """Forget the lazily compiled inference methods, so that new prepare_for_inference options trace anew."""
    import torch

    for attr in list(vars(model)):
        if attr.endswith("_is_compiled"):
            setattr(model, attr, False)
    torch._dynamo.reset()


def run(name: str, policy, model, processor, cfg, num_video_frames: int, *, log, finish, path_options) -> dict:
    import fast

    ref = __import__("json").loads(fast.reference_path(name).read_text())
    joint = policy.joint
    diff = lambda a, b: {"mean": round(float(np.abs(a - b).mean()), 6), "max": round(float(np.abs(a - b).max()), 5)}  # noqa: E731

    def timed(p, obs, n):
        out, ms = None, []
        for _ in range(n):
            t = time.perf_counter()
            out = p.infer(dict(obs))["actions"]
            ms.append(round((time.perf_counter() - t) * 1000, 1))
        return out, ms

    EVAL = import_eval()
    stock = {}
    for depth in (True, False):
        obs = fast.decode_observation(ref["observation"], depth=depth)
        ecfg = eval_cfg(EVAL, name, joint, depth)
        same_cfg = ecfg.model == cfg.model and ecfg.data.train.processor == cfg.data.train.processor
        off = official_actions(EVAL, ecfg, model, processor, obs, None)
        ours, ms = timed(policy, obs, 4)
        stock[depth] = ours
        log("verify_official", depth=depth, joint=joint, model_cfg_equal=bool(same_cfg), ms=ms,
            official_vs_wrapper=diff(off, ours), official_actions=np.round(off, 6).tolist(),
            stock_actions=np.round(ours, 6).tolist())
        log("verify_seed", depth=depth, seed43_vs_seed42=diff(other_seed(policy, obs, 43), ours))
        log("verify_stages", path="stock", depth=depth, joint=joint, **stage_profile(policy, obs))

        other = type(policy)(policy.name, policy.model, policy.ops, joint=not joint,
                             num_video_frames=policy.num_video_frames, no_grad=policy._no_grad)
        log("verify_other_regime", depth=depth, joint=not joint, ms=timed(other, obs, 3)[1])
    # Variants of the repo's prepare_for_inference options (each documented as the same math). Measured on an A40 in
    # the full joint regime and dropped there: attn_backend="auto" (no gain), scope "loop" (no gain, 6 to 12 min of
    # compile), glue_cache + encoder_cuda_graph (failed). In the action-only regime "auto" drops the all-True
    # attention masks of the KV-cache path, so SDPA may run flash (the repo: about 4x on that attention).
    fast_opts = dict(path_options["fast"])
    variants = {"compile_step": fast_opts,
                "compile_step_auto_attn": dict(fast_opts, attn_backend="auto"),
                "compile_step_auto_attn_glue_encoders": dict(fast_opts, attn_backend="auto", glue_cache=True,
                                                             compile_encoders=True),
                "compile_loop_auto_attn": dict(fast_opts, attn_backend="auto", torch_compile_scope="loop"),
                "compile_loop_auto_attn_glue_encoders": dict(fast_opts, attn_backend="auto", torch_compile_scope="loop",
                                                             glue_cache=True, compile_encoders=True)}
    fast_policy = type(policy)(policy.name, policy.model, policy.ops, joint=joint, num_video_frames=num_video_frames,
                               no_grad=policy._no_grad, text_cache=True)
    results = {}
    keep = [v for v in os.environ.get("RS_VERIFY_VARIANTS", "").split(",") if v]  # e.g. compile_step only
    for label, options in variants.items():
        if keep and label not in keep:
            continue
        try:
            reset_compiled(model)
            finish(options)
            t = time.time()
            for depth in (True, False):
                obs = fast.decode_observation(ref["observation"], depth=depth)
                first, _ = timed(fast_policy, obs, 1)
                a, ms = timed(fast_policy, obs, 6)
                d = diff(a, stock[depth])
                log("verify_fast", variant=label, depth=depth, first_call_s=round(time.time() - t, 1), ms=ms,
                    vs_stock=d, repeat=diff(a, first))
                results.setdefault(label, []).append((float(np.median(ms)), d["max"]))
                log("verify_stages", path="fast", variant=label, depth=depth, joint=joint,
                    **stage_profile(fast_policy, obs))
        except Exception as e:  # noqa: BLE001
            log("verify_fast_failed", variant=label, error=repr(e)[:600])
    reset_compiled(model)
    finish(path_options["stock"])
    ok = {k: max(m for m, _ in v) for k, v in results.items() if len(v) == 2 and all(d <= fast.MAX_ABS for _, d in v)}
    best = min(ok, key=ok.get) if ok else None
    log("verify_choice", best=best, median_ms={k: round(v, 1) for k, v in ok.items()})
    return stock, (variants[best] if best else None)


def selftests(switch, ref: dict, stock: dict, gpu: str, log) -> None:
    """The fast path's boot self-test through the switch: against this run's own stock actions (what a reference
    stored for this GPU would hold) and against every stored reference (other GPU types)."""
    import fast

    class Fast:
        def infer(self, obs):
            return switch.infer({**obs, "rs_path": "fast"})

    joint = switch.paths["stock"].joint
    key = fast.regime(joint)
    own = {**ref, "references": {gpu: {key: {"actions": stock[True].tolist(),
                                              "actions_no_depth": stock[False].tolist()}}}}
    stored = [(f"stored_{g}", ref, g) for g, by_regime in ref["references"].items() if key in by_regime]
    for label, r, g in [("this_run", own, gpu)] + stored:
        ok, detail = fast.selftest(Fast(), r, g, joint)
        log("verify_selftest", gpu=gpu, regime=key, reference=label, passed=ok, **detail)


class Switch:
    """Verify runs only (RS_VERIFY=1): serves the stock setting, or the fast one when a request carries
    "rs_path": "fast" (prepare_for_inference flags switch per request; the compiled graphs stay cached), so one worker
    measures both through the gateway."""

    def __init__(self, stock, quick, finish, options: dict, warm_obs: dict):
        self.paths, self.finish, self.options, self.current = {"stock": stock, "fast": quick}, finish, options, None
        self.metadata = {**stock.metadata, "path": "stock", "path_reason": "verify run (RS_VERIFY=1): rs_path picks"}
        for path in ("fast", "stock"):  # compile the fast path now, not in the first measured request
            for depth_obs in warm_obs:
                self.infer({**depth_obs, "rs_path": path})

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        path = str(obs.pop("rs_path", "stock"))
        if path != self.current:
            self.finish(self.options[path])
            self.current = path
        return self.paths[path].infer(obs)
