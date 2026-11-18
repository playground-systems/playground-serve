"""FLUX 3 Action (Black Forest Labs) worker: the DROID policy behind Playground Serve's websocket protocol.

LICENSE: the weights are under the FLUX Kommunity License v1.0 (gated on Hugging Face): non-commercial use, and
commercial use of outputs only for a "Qualifying User" (under US$5M annual revenue with affiliates).

Model calls follow the repo's `flux-action serve-robolab` at COMMIT (src/flux_action/cli.py:185-205, :316-349):
load_serving_policy(checkpoint, revision=..., subfolder=..., device="cuda", settings={}, dtype="bfloat16",
compile_dit=True, offload_text_encoder=False, warmup=1) (serving/robolab.py:159-224), wrapped in RoboLabPolicy, whose
infer(obs) is what the official server calls (robolab.py:113-131, :227-251). It composes the cameras, flips the gripper
and calls FluxActionPolicy.predict_from_composite (policy.py:1219-1250) with the package's sampler settings (4
cosmos_unipc steps, guidance 4.0, shift 5.0; docs/setup.md:429-442).

Observation (msgpack-numpy dict), the official request keys (robolab.py:16-22, docs/setup.md:461-473):
    "observation/wrist_image_left", "observation/exterior_image_1_left", "observation/exterior_image_2_left":
        360x640x3 uint8 RGB; the wrist frame goes on top and the exterior frames at half size below, and the result
        must be exactly 540x640 (robolab.py:72-87). Or one pre-tiled "observation/image" (540, 640, 3).
    "observation/joint_position": (7,) radians (or (T, 7): the last row is used),
    "observation/gripper_position": scalar or (1,), closed fraction as in DROID, "prompt": str.
Reply: {"actions": float32 (32, 8)}: absolute joint commands and gripper closed fraction, 15 Hz (README model card;
the official reply names it "action").

Served and verified on an A40 (NOTES.md) with the official eager setting: the spec sets RS_COMPILE_DIT=0, because the
CLI's default compiled setting failed at CUDA graph capture on this stack; load() falls back to eager if it is turned
on and fails. No separate fast path.

Endpoint env: RS_MODEL (a VARIANTS key), HF_HOME on the volume, HF_TOKEN (an account that accepted the license of
both flux-3-action-droid and flux-3-action-base), optional RS_SUBFOLDER (a package under variants/, e.g. variants/gd;
the fp8r ones need SM89 or newer), RS_OFFLOAD_TEXT_ENCODER=1 (the server's --offload-text-encoder, about 9 GB less GPU
memory) and RS_COMPILE_DIT ("0" = the server's --no-compile-dit, the spec's default; "1" = its compiled default).
"""

from __future__ import annotations

import os

import numpy as np

COMMIT = "aea7863fbbbda1e282fa7bde2054e290bf09b3f8"  # github.com/black-forest-labs/flux-action, 2026-11-10
IMAGE_KEYS = ("observation/wrist_image_left", "observation/exterior_image_1_left", "observation/exterior_image_2_left")
STATE_KEYS = ("observation/joint_position", "observation/gripper_position")
FRAME_HW = (360, 640)  # per camera, so that the composite is the required 540x640 (robolab.py:46, :72-87)
# Hugging Face revision read 2026-11-13. The package's config pins its text encoder and video VAE to
# black-forest-labs/flux-3-action-base@62878e2925e59b7a89ec14463ce89932624c490d.
VARIANTS = {
    "flux3_action_droid": dict(repo="black-forest-labs/flux-3-action-droid",
                                revision="d84c80adc0721780245aabb75e45d4164ead7b97", horizon=32, action_dim=8, hz=15),
}
LICENSE = ("FLUX Kommunity License v1.0: non-commercial; commercial use of outputs only for Qualifying Users "
           "(under US$5M annual revenue)")


def serving_kwargs(name: str, env=os.environ) -> dict:
    """load_serving_policy keywords as `flux-action serve-robolab` passes them with its defaults (cli.py:322-335)."""
    return dict(revision=VARIANTS[name]["revision"], subfolder=env.get("RS_SUBFOLDER") or None, device="cuda",
                settings={}, dtype="bfloat16", compile_dit=env.get("RS_COMPILE_DIT", "1") == "1",
                offload_text_encoder=env.get("RS_OFFLOAD_TEXT_ENCODER") == "1", warmup=1)


class Policy:
    """Wraps the official RoboLabPolicy; renames its "action" to Playground Serve's "actions"."""

    def __init__(self, name: str, adapter, subfolder: str | None = None):
        self.name, self.variant, self.adapter = name, VARIANTS[name], adapter
        v = self.variant
        checkpoint = f"{v['repo']}@{v['revision']}" + (f"/{subfolder}" if subfolder else "")
        self.metadata = {"model": name, "family": "flux3_action", "checkpoint": checkpoint,
                         "action_shape": [v["horizon"], v["action_dim"]], "action_hz": v["hz"],
                         "keys": list(IMAGE_KEYS + STATE_KEYS), "state_dim": v["action_dim"], "prompt_key": "prompt",
                         "image_size": list(FRAME_HW), "history": 0, "license": LICENSE, "experimental": True,
                         "non_commercial": True}

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        seed = obs.pop("rs_seed", None)  # measurement hook, as on the openpi worker: RoboLabPolicy.infer's own seed
        out = self.adapter.infer(obs) if seed is None else self.adapter.infer(obs, seed=int(seed))
        return {"actions": np.asarray(out["action"], dtype=np.float32)}


def load() -> Policy:
    import pgs_server
    from flux_action.serving.robolab import RoboLabPolicy, load_serving_policy

    name = os.environ["RS_MODEL"]
    kwargs = serving_kwargs(name)
    prime_cuda_graph_rng()
    if os.environ.get("RS_VERIFY") == "1":
        verify_ready(pgs_server)
        return verify(name, kwargs, pgs_server.log)
    try:
        policy = load_serving_policy(VARIANTS[name]["repo"], **kwargs)
        reason = "official serve-robolab settings (compiled DiT and VAE encoder)"
    except Exception:  # noqa: BLE001  (the compiled setting failed: serve the official eager one)
        import gc
        import traceback

        import torch

        pgs_server.log("compiled_failed", error=traceback.format_exc()[-1500:])
        gc.collect()
        torch.cuda.empty_cache()
        kwargs = {**kwargs, "compile_dit": False}
        policy = load_serving_policy(VARIANTS[name]["repo"], **kwargs)
        reason = "compiled load failed at this start: official eager setting (--no-compile-dit)"
    pgs_server.log("serving_setup", **(policy.serving_setup or {}))
    served = Policy(name, RoboLabPolicy(policy), subfolder=kwargs["subfolder"])
    served.metadata.update(path="stock", path_reason=reason)
    return served


def prime_cuda_graph_rng() -> None:
    """Capture one trivial CUDA graph before anything else runs. The first capture registers the CUDA generator for
    graphs and allocates its graph-safe seed and offset tensors; when that first happened under torch.inference_mode
    (an encoder call), the DiT's later capture outside it failed ("Inplace update to inference tensor outside
    InferenceMode", in CUDAGraph.capture_begin). The trivial graph is never replayed and touches no model state."""
    import torch

    x = torch.zeros(1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1)
    torch.cuda.synchronize()


def verify_observations(path=None) -> list[dict]:
    """verify_obs.json (bring-up runs only), or another file of the same form: recorded DROID frames, resized to
    360 x 640 as the client does."""
    import base64
    import io
    import json
    import pathlib

    from PIL import Image

    def frame(b32: str) -> np.ndarray:
        with Image.open(io.BytesIO(base64.b32decode(b32))) as im:
            return np.asarray(im.convert("RGB").resize(FRAME_HW[::-1], resample=Image.BILINEAR))

    data = json.loads(pathlib.Path(path or pathlib.Path(__file__).with_name("verify_obs.json")).read_text())
    return [{**{f"observation/{k}": frame(o[k]) for k in ("wrist_image_left", "exterior_image_1_left",
                                                          "exterior_image_2_left")},
             "observation/joint_position": np.asarray(o["joint_position"], np.float32),
             "observation/gripper_position": np.asarray(o["gripper_position"], np.float32), "prompt": o["prompt"]}
            for o in data["observations"]]


def verify(name: str, kwargs: dict, log) -> Policy:
    """Bring-up check (RS_VERIFY=1), logged: the official eager setting (load_serving_policy with compile_dit=False,
    what the repo's replay-fixtures command compares) in its own process against the served setting at the same seed;
    the official RoboLabPolicy.infer against this wrapper; a new seed; times. Serves the served setting afterwards."""
    import time

    from flux_action.serving.robolab import RoboLabPolicy, load_serving_policy

    obs_list, seed = verify_observations(), 7

    def diff(a, b):
        d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
        return {"mean": round(float(d.mean()), 6), "max": round(float(d.max()), 5)}

    def timed(p, obs, n, s=seed):
        out, ms = None, []
        for _ in range(n):
            t = time.perf_counter()
            out = p.infer({**obs, "rs_seed": s})["actions"]
            ms.append(round((time.perf_counter() - t) * 1000, 1))
        return out, ms

    import json
    import subprocess
    import sys
    import tempfile

    # The official eager setting runs in its own process: CUDA graph capture after eager inference in the same
    # process failed ("Inplace update to inference tensor outside InferenceMode" in capture_begin), and both settings
    # do not fit one 48 GB card at once.
    out = tempfile.mktemp(suffix=".json")
    t = time.time()
    subprocess.run([sys.executable, __file__, "eager", out], check=True)
    eager = json.loads(open(out).read())
    a_eager = [np.asarray(a, np.float32) for a in eager["actions"]]
    log("verify_eager", s=round(time.time() - t, 1), ms=eager["ms"], repeat=eager["repeat"],
        actions=[np.round(a, 6).tolist() for a in a_eager])
    t = time.time()
    try:
        policy = load_serving_policy(VARIANTS[name]["repo"], **kwargs)
    except Exception:  # noqa: BLE001  (as load(): the official eager setting instead)
        import gc
        import traceback

        import torch

        log("verify_compiled_failed", error=traceback.format_exc()[-3000:])
        gc.collect()
        torch.cuda.empty_cache()
        kwargs = {**kwargs, "compile_dit": False}
        policy = load_serving_policy(VARIANTS[name]["repo"], **kwargs)
    adapter = RoboLabPolicy(policy)
    log("verify_served_ready", s=round(time.time() - t, 1), serving_setup=policy.serving_setup or {})
    served = Policy(name, adapter, subfolder=kwargs["subfolder"])
    for i, obs in enumerate(obs_list):
        a, ms = timed(served, obs, 5)
        official = adapter.infer(dict(obs), seed=seed)["action"]
        log("verify_served", obs=i, ms=ms, vs_eager_process=diff(a, a_eager[i]), official_vs_wrapper=diff(official, a),
            new_seed_vs_seed=diff(timed(served, obs, 1, seed + 1)[0], a), actions=np.round(a, 6).tolist())
    return served


def verify_ready(pgs_server) -> None:
    """Verify runs only: health turns 200 now. A flex_pi verify worker whose health stayed 204 for about 9.5 minutes
    was restarted by Runpod in the middle of its checks; bring-up sends no request before the ready line anyway."""
    import pathlib

    pathlib.Path(pgs_server.READY_FILE).touch()


def make_example() -> dict:
    """Warm-up input like the server's own (zero image and state, robolab.py:197-200), as the three views."""
    obs = {key: np.zeros((*FRAME_HW, 3), np.uint8) for key in IMAGE_KEYS}
    return {**obs, "observation/joint_position": np.zeros(7, np.float32),
            "observation/gripper_position": np.zeros(1, np.float32), "prompt": "warm-up request"}


def main() -> None:
    import pgs_server

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set: both black-forest-labs/flux-3-action-droid and -base are gated; the "
                         "user accepts their license on Hugging Face and supplies a token (NOTES.md)")
    # The official loader downloads into HF_HOME itself: the spec's RS_SELF_FETCH=1 tells boot.sh to skip stage.py
    # fetch, so run_worker waits for no checkpoint copy.
    pgs_server.run_worker(load, make_example)


def eager_main(out: str) -> None:
    """`serve.py eager OUT` (verify runs): the official eager setting (load_serving_policy with compile_dit=False, as
    the repo's replay-fixtures command) on the verify observations at seed 7; actions and times to OUT (json)."""
    import json
    import time

    from flux_action.serving.robolab import RoboLabPolicy, load_serving_policy

    name = os.environ["RS_MODEL"]
    kwargs = {**serving_kwargs(name), "compile_dit": False, "warmup": 0}
    adapter = RoboLabPolicy(load_serving_policy(VARIANTS[name]["repo"], **kwargs))
    actions, ms = [], []
    for obs in verify_observations():
        times = []
        for _ in range(3):
            t = time.perf_counter()
            a = np.asarray(adapter.infer(dict(obs), seed=7)["action"], np.float32)
            times.append(round((time.perf_counter() - t) * 1000, 1))
        actions.append(a)
        ms.append(times)
    again = np.asarray(adapter.infer(dict(verify_observations()[0]), seed=7)["action"], np.float32)
    d = np.abs(again.astype(np.float64) - actions[0])
    with open(out, "w") as f:
        json.dump({"actions": [a.tolist() for a in actions], "ms": ms,
                   "repeat": {"mean": float(d.mean()), "max": float(d.max())}}, f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "eager":
        eager_main(sys.argv[2])
    else:
        main()
