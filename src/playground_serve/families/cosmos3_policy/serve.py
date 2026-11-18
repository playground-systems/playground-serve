"""Cosmos 3 policy (NVIDIA) worker: one DROID checkpoint behind Playground Serve's websocket protocol (pgs_server.py).

Model calls follow the repo's RoboLab server at COMMIT (cosmos_framework/scripts/action_policy_server_robolab.py, cited
as robolab.py): build its RobolabServerArgs with the flags of the docs' launch lines (docs/action_policy_droid_server.md;
Edge adds --format-prompt-as-json True --guidance-interval 960 1001), construct RobolabPolicyService (the object the
script hands to openpi's WebsocketPolicyServer) and run one request the way its infer(obs) does on one rank:
_build_sample, then _generate with the request's seed, then _format_outputs. The service downloads the checkpoint at the
pinned revision into HF_HOME, tiles the cameras, inverts the gripper on the way in and out, and runs
model.generate_samples_from_batch with 4 UniPC steps, guidance 3.0 and shift 5.0.

Observation (msgpack-numpy dict), the official request keys:
    "observation/wrist_image_left", "observation/exterior_image_1_left", "observation/exterior_image_2_left":
        HxWx3 uint8 RGB. The service puts the wrist frame on top and the two exterior frames at half its size below,
        then resizes that mosaic to 540x640; 360x640 frames give exactly 540x640 (the metadata's image_size, so the
        client's jpeg mode sends them at that size). One pre-tiled "observation/image" works instead of the three.
    "observation/joint_position": (7,) radians (or (T, 7): the last row is used),
    "observation/gripper_position": scalar or (1,), DROID's convention (closed fraction), "prompt": str.
    "rs_seed" (int, optional): a measurement hook, as in the openpi worker: the request's noise seed. Without it the
        seed comes from the service's own RNG, which advances per request (robolab.py _next_seed).
Reply: {"actions": float32 (32, 8)}: absolute joint positions and gripper for the next 32 steps at 15 Hz (the official
reply names it "action").

Model paths (metadata "path", "path_reason"): "stock" is the service above, unchanged. "fast" (fast.py) is the same
service and weights with inference options the repo ships (CUDA graphs, static shapes, batched CFG); it serves only
after its boot self-test reproduced the stock actions stored for this GPU (selftest_<model>.json), else stock serves.

Endpoint env: RS_MODEL (a VARIANTS key), HF_HOME on the volume, RS_FAST ("auto" or "off"), RS_VERIFY=1 (bring-up only,
verify.py: log the official service's actions next to this wrapper's, stock against the fast options, and the noise
floor).
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

COMMIT = "1f1737931214166d5e7b253def319c562cf7a1eb"  # github.com/NVIDIA/cosmos-framework, 2026-11-13
HORIZON, ACTION_DIM, ACTION_HZ = 32, 8, 15  # robolab.py defaults: chunk 32, action_dim 8, conditioning fps 15
IMAGE_KEYS = ("observation/wrist_image_left", "observation/exterior_image_1_left", "observation/exterior_image_2_left")
STATE_KEYS = ("observation/joint_position", "observation/gripper_position")
FRAME_HW = (360, 640)  # per camera: the mosaic is then exactly the service's 540x640
LICENSE = "OpenMDW-1.1 (code and weights; commercial use allowed)"
# Hugging Face revisions read 2026-11-13 (both repos public, not gated). `flags` are RobolabServerArgs fields.
VARIANTS = {
    "cosmos3_edge_droid": dict(repo="nvidia/Cosmos3-Edge-Policy-DROID",
                               revision="a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14",
                               flags=dict(format_prompt_as_json=True, guidance_interval=(960.0, 1001.0))),
    "cosmos3_nano_droid": dict(repo="nvidia/Cosmos3-Nano-Policy-DROID",
                               revision="5e9dd43c3c99d813c512960594bfd7b40678b180", flags={}),
}


def server_args(name: str) -> dict:
    """RobolabServerArgs fields for one variant: the docs' launch line, revision pinned (the default is main)."""
    v = VARIANTS[name]
    return {"checkpoint_path": v["repo"], "hf_revision": v["revision"], **v["flags"]}


class Policy:
    """One request as RobolabPolicyService.infer runs it on a single rank, with the seed from "rs_seed" when the
    request carries one. `zero_tail` (the fast path, fast.py) builds the sample with fast.zero_tail()."""

    def __init__(self, name: str, service, path: str = "stock", path_reason: str = "", zero_tail: bool = False,
                 generate_kwargs: dict | None = None):
        self.name, self.variant, self.service, self.zero_tail = name, VARIANTS[name], service, zero_tail
        self.generate_kwargs = dict(generate_kwargs or {})  # verify variants: extra generate_samples_from_batch args
        self._lock = getattr(service, "_lock", None) or threading.Lock()
        v = self.variant
        self.metadata = {"model": name, "family": "cosmos3_policy", "checkpoint": f"{v['repo']}@{v['revision']}",
                         "action_shape": [HORIZON, ACTION_DIM], "action_hz": ACTION_HZ,
                         "keys": list(IMAGE_KEYS + STATE_KEYS), "state_dim": 8, "prompt_key": "prompt",
                         "image_size": list(FRAME_HW), "history": 0, "license": LICENSE, "experimental": True,
                         "path": path, "path_reason": path_reason}

    def _generate_with(self, sample: dict, seed: int) -> dict:
        """robolab.py _generate with extra keywords for generate_samples_from_batch (verify variants only)."""
        import torch
        from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample

        s = self.service
        with torch.inference_mode():
            return s.model.generate_samples_from_batch(
                _build_data_batch_from_sample(sample), guidance=s.cfg.guidance,
                guidance_interval=list(s.cfg.guidance_interval) if s.cfg.guidance_interval is not None else None,
                seed=[seed], num_steps=s.cfg.num_steps, shift=s.cfg.shift, **self.generate_kwargs)

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        seed = obs.pop("rs_seed", None)
        s = self.service
        with self._lock:
            start = time.monotonic()
            seed = s._next_seed() if seed is None else int(seed)
            if self.zero_tail:
                import fast  # /rs/fast.py

                with fast.zero_tail():
                    sample = s._build_sample(obs)
            else:
                sample = s._build_sample(obs)
            samples = s._generate(sample, seed) if not self.generate_kwargs else self._generate_with(sample, seed)
            out = s._format_outputs(obs, samples, start_time=start)
        return {"actions": np.asarray(out["action"], dtype=np.float32)}


def build_service(name: str, setup_overrides: dict | None = None):
    """RobolabPolicyService for one variant; `setup_overrides` are extra OmniSetupOverrides fields (verify runs)."""
    # The service downloads with `uv run --project cosmos_framework/utils/hf_cli ... hf download` (utils/checkpoint_db.py
    # _hf_download). uv is a dependency of the repo, so it sits next to this Python; build_env.sh prebuilt the hf CLI
    # environment, so a boot needs no PyPI access for it.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("IMAGINAIRE_HF_CLI_ENVIRONMENT", os.path.join(os.path.dirname(sys.prefix), "hf_cli"))
    # Importing the script runs its init_script(), as `python -m` does.
    from cosmos_framework.scripts import action_policy_server_robolab as robolab

    extra = dict(setup_overrides or {})

    class Service(robolab.RobolabPolicyService):
        def _build_setup_args(self, args, parallelism_overrides):  # spread into OmniSetupOverrides' fields
            return super()._build_setup_args(args, {**parallelism_overrides, **extra})

    return (Service if extra else robolab.RobolabPolicyService)(robolab.RobolabServerArgs(**server_args(name)))


DEFAULT_THREADS = 16


def set_threads(log) -> None:
    """torch's CPU thread pool, set once before anything compiles (changing it later recompiles until dynamo's cache
    limit stops the process): RS_TORCH_THREADS, default 16, never more than torch's own default (one per core).
    On a 224-core H100 host torch's default of 112 threads made the CPU steps (resize, pad, tokenization) 246 ms;
    16 threads made them 45 ms with the same actions (NOTES.md). "0" keeps torch's default."""
    import torch

    want = int(os.environ.get("RS_TORCH_THREADS") or DEFAULT_THREADS)
    if want > 0:
        torch.set_num_threads(min(want, torch.get_num_threads()))
    log("cpu", host_cores=os.cpu_count(), torch_threads=torch.get_num_threads())


def gpu_name() -> str:
    import pgs_server

    return pgs_server.gpu_info("name").splitlines()[0].strip() if pgs_server.gpu_info("name") else ""


def load() -> Policy:
    import pgs_server

    import fast  # /rs/fast.py, next to this file

    set_threads(pgs_server.log)

    name = os.environ["RS_MODEL"]
    if os.environ.get("RS_VERIFY") == "1":  # bring-up check (verify.py ships only with RS_VERIFY=1)
        import verify

        verify.ready_now(pgs_server)
        return verify.run(name, pgs_server.log, fast, sys.modules[__name__])
    t = time.time()
    service = build_service(name)
    pgs_server.log("service_built", s=round(time.time() - t, 1))
    gpu = gpu_name()
    use, reason = fast.decide(os.environ.get("RS_FAST", "auto"), name, gpu)
    pgs_server.log("path", path="fast" if use else "stock", reason=reason)
    if use:
        t = time.time()
        try:
            ref = fast.load_reference(name)
            same, times = fast.sample_check(service, fast.decode_observation(ref["observation"]))
            policy = Policy(name, service, "fast", reason, zero_tail=True)
            ok, detail = fast.selftest(policy, ref, gpu) if same else (False, {"error": "fast sample differs from stock"})
            pgs_server.log("fast_selftest", passed=ok, s=round(time.time() - t, 1), sample_equal=same, **times,
                           **detail)
            if ok:
                return policy
            reason = f"fast path failed its self-test against stock: {detail}"
        except Exception as e:  # noqa: BLE001  (any failure: serve stock)
            reason = f"fast path failed: {e!r}"[:400]
        pgs_server.log("fallback", reason=reason)
    return Policy(name, service, "stock", reason)


def make_example() -> dict:
    """Warm-up input: zero frames at DROID's size, zero state. (The official server takes no warm-up request.)"""
    obs = {key: np.zeros((*FRAME_HW, 3), np.uint8) for key in IMAGE_KEYS}
    return {**obs, "observation/joint_position": np.zeros(7, np.float32),
            "observation/gripper_position": np.zeros(1, np.float32), "prompt": "warmup", "rs_seed": 0}


def main() -> None:
    import pgs_server

    # The service downloads the checkpoint itself (into HF_HOME on the volume): the spec's RS_SELF_FETCH=1 tells
    # boot.sh to skip stage.py fetch, so run_worker waits for no checkpoint copy.
    pgs_server.run_worker(load, make_example)


if __name__ == "__main__":
    main()
