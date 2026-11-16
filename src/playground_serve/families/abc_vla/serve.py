"""ABC-VLA (Amazon FAR) worker: the released VLA parent behind Playground Serve's websocket protocol (pgs_server.py).

Two paths; the metadata's "path" and "path_reason" and the log line {"rs": "path"} say which one serves and why.

stock  The repo's own server at COMMIT, as `deploy/serve_policy.py --policy.checkpoint-path=<ckpt>` builds it:
       deploy.policy.vla_policy.Policy(Args(...).vla_config()) (serve_policy.py:15-23), a DeployPolicy
       (deploy/policy/base.py) around abc_minimal.policy.VLAInferencePolicy (Gemma 3 4B + SigLIP at 224 in bf16, an
       fp32 attention pool and diffusion head, 10 Euler steps, z-score stats embedded in the checkpoint). Each request
       goes through the server's own infer_observation (deploy/websocket_server.py:20-32): it decodes JPEG-bytes
       images and pops action_prefix / prefix_length / noise before policy.infer. With no "noise" the policy uses
       zeros (base.py:55-56), so the same observation always gives the same actions.
fast   abc_fast.py: the same modules and weights, the same dtypes, the same 10 steps and the same inputs, called with
       the per-prompt work done once (tokens, text embeddings, attention masks, rotary tables) and the model run as one
       CUDA graph per prompt and prefix length. The sequence is cut at the prompt's last token instead of padded to
       256: the padding comes after every real token, so under the causal mask no real token attends to it, and the
       attention pool masks it out; it got exactly zero weight in the stock path. Only for GPUs in FAST, and only after
       a boot self-test: the stored observation (selftest_abc_vla_yam.json) must give the stored stock actions on the
       stock path and the same actions on the fast path, twice; else the stock path serves. A start that crashed while
       it built the fast path leaves FAST_FAILED, and the next start in that container serves stock.

Observation (msgpack-numpy dict), the repo's request keys (deploy/README.md, abc_sim/README.md:94-98):
    "images": {"top", "left", "right"}: uint8 RGB, HxWx3 (what Playground Serve's jpeg mode delivers; transposed
              here), (3, H, W) as the repo's client sends, or JPEG bytes (the repo's client compression); any size,
              the model letterboxes to 224x224,
    "state": (14,) left joints 1-6, left gripper, right joints 1-6, right gripper (grippers 0-1),
    "prompt": str (default "throw plastic bottles in bin", InferenceConfig.prompt),
    optional "action_prefix" (n, 14) + "prefix_length" (real-time chunking; the checkpoint was trained on up to 7)
    and "noise" (30, 14).
Reply: {"actions": float32 (30, 14), "policy_timing": {"policy_ms"}}, absolute joint-position targets at 30 Hz
(abc_sim/config.py:54); policy_ms is the policy call without the server's JPEG decode.

Weights: the checkpoint is one file on the authors' host (not Hugging Face). boot.sh points RS_CKPT_DIR at the volume's
complete copy or at a container-disk directory; until stage.py fetch takes https:// URIs (NOTES.md, requests for the
core), prepare() downloads it there with stage.py's parallel ranges and checks its sha256 once, and stage.py fill saves
it to the volume after ready.

Endpoint env: RS_MODEL, RS_CHECKPOINT, RS_CKPT_DIR (boot.sh), RS_FAST ("auto" default, "off" = stock path),
RS_FAST_COMPILE ("1" default: the graph body compiled with inductor; "0": the CUDA graph alone, bit-identical),
RS_COMPILE_CACHE (inductor's cache, per GPU type, on the volume).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

import pgs_server

COMMIT = "d0832d12651d1b260a652861a14648dc5f3660c7"  # github.com/amazon-far/abc, 2026-09-25
# The released VLA parent prepare.py --vla-pretrained downloads by default (family abc130k, step 200000). Values from
# its sidecar https://abc-data.timehorizons.org/checkpoints/vla_abc130k_v2/200000.json, read 2026-11-13 and 2026-11-16.
VARIANTS = {
    "abc_vla_yam": dict(uri="https://abc-data.timehorizons.org/checkpoints/vla_abc130k_v2/200000.pt",
                        file="200000.pt", bytes=8802367911,
                        sha256="4838acf69279497b917c998039089b5b6a310dfe65fc0b5c438c3339ee25c753",
                        cameras=("top", "left", "right"), state_dim=14, horizon=30, hz=30),
}
LICENSE = "code Apache-2.0; weights Apache-2.0 with the Gemma Terms of Use (Gemma 3 backbone)"
# GPU names (nvidia-smi) where the fast path was measured served against stock (NOTES.md).
FAST = {"abc_vla_yam": ("NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3")}
FAST_FAILED = "/tmp/rs_fast_failed"
HERE = pathlib.Path(__file__).resolve().parent
log = pgs_server.log


def channels_first(image):
    """HxWx3 arrays become 3xHxW, the layout the repo's policy reads; 3xHxW arrays and JPEG bytes pass unchanged."""
    if isinstance(image, (bytes, bytearray)):
        return image
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 3 and image.shape[0] != 3:
        return np.transpose(image, (2, 0, 1))
    return image


class Policy:
    def __init__(self, name: str, deploy_policy, infer_observation, path: str = "stock", reason: str = ""):
        """`deploy_policy`: deploy.policy.vla_policy.Policy or abc_fast.FastDeployPolicy (the same infer contract);
        `infer_observation`: deploy.websocket_server's."""
        self.name, self.variant = name, VARIANTS[name]
        self.deploy_policy, self.infer_observation = deploy_policy, infer_observation
        v = self.variant
        self.metadata = {"model": name, "family": "abc_vla", "checkpoint": v["uri"], "commit": COMMIT,
                         "action_shape": [v["horizon"], v["state_dim"]], "action_hz": v["hz"],
                         "keys": ["images", "state", "prompt"], "cameras": [f"images.{c}" for c in v["cameras"]],
                         "state_dim": v["state_dim"], "prompt_key": "prompt", "image_size": [224, 224],
                         "history": 0, "license": LICENSE, "path": path, "path_reason": reason}

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)  # infer_observation pops keys; leave the caller's dict alone
        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError(f"'images' must be a map with the cameras {list(self.variant['cameras'])}")
        missing = set(self.variant["cameras"]) - set(images)
        if missing:
            raise ValueError(f"images is missing {sorted(missing)}")
        obs["images"] = {k: channels_first(v) for k, v in images.items()}
        t = time.perf_counter()
        actions = np.asarray(self.infer_observation(self.deploy_policy, obs)["actions"], dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("the policy returned non-finite actions")
        # the policy call alone; the server's infer_ms adds the JPEG decode before it
        return {"actions": actions, "policy_timing": {"policy_ms": (time.perf_counter() - t) * 1e3}}


# ----- weights
def weights_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("RS_CKPT_DIR") or "/runpod-volume/pgs/weights/abc_vla")


def prepare() -> pathlib.Path:
    """The checkpoint file, downloaded into RS_CKPT_DIR unless that holds a complete copy (stage.py's marker)."""
    import stage  # /rs/stage.py

    v = VARIANTS[os.environ["RS_MODEL"]]
    dest = weights_dir()
    if os.environ.get("RS_CKPT_DIR"):  # boot.sh's stage.py fetch may be writing it: let it end first
        pgs_server.wait_for_fetch(dest)
    if not stage.complete(dest):
        (dest / ".rs_failed").unlink(missing_ok=True)  # boot.sh's stage.py fetch takes gs:// URIs only
        t = time.time()
        log("ckpt_fetch_start", uri=v["uri"], dest=str(dest))
        stage.listing = lambda uri: [(v["file"], v["bytes"], uri)]
        stage.fetch(v["uri"], dest)
        got = sha256(dest / v["file"])
        if got != v["sha256"]:
            (dest / stage.MARKER).unlink(missing_ok=True)
            raise RuntimeError(f"sha256 of {v['file']} is {got}, expected {v['sha256']}")
        log("ckpt_fetch_done", bytes=v["bytes"], s=round(time.time() - t, 1),
            mb_s=round(v["bytes"] / (time.time() - t) / 1e6))
    return dest / v["file"]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


# ----- policy
def load_stock(path: pathlib.Path):
    """deploy/serve_policy.py's create_policy for a VLA checkpoint, with the server's defaults."""
    sys.path.insert(0, os.getcwd())  # boot.sh runs this from the repo checkout
    from abc_minimal.policy import InferenceConfig
    from deploy.policy.vla_policy import Policy as VLAPolicy
    from deploy.serve_policy_config import Args

    args = Args(policy=InferenceConfig(checkpoint_path=str(path)), policy_type="vla")
    return VLAPolicy(args.vla_config())


def decide(name: str, gpu: str) -> tuple[str, str]:
    mode = os.environ.get("RS_FAST", "auto")
    if mode != "auto":
        return "stock", f"fast path turned off (RS_FAST={mode})"
    if name not in FAST:
        return "stock", f"no fast path for {name}"
    if gpu not in FAST[name]:
        return "stock", f"fast path not measured on {gpu or 'this GPU'}, only on {', '.join(FAST[name])}"
    if not selftest_file(name).is_file():
        return "stock", f"no self-test reference for {name}"
    if os.path.exists(FAST_FAILED):
        return "stock", f"an earlier start in this container crashed while it built the fast path: " \
                        f"{pathlib.Path(FAST_FAILED).read_text()[:200]}"
    return "fast", f"CUDA graph path on {gpu}, if its boot self-test passes"  # load() updates it after the test


def selftest_file(name: str) -> pathlib.Path:
    return HERE / f"selftest_{name}.json"


def selftest_cases(name: str) -> tuple[list[dict], dict]:
    """The stored observations (images as the client's JPEG bytes, decoded as server_decode does) with the actions
    the stock path gave for them."""
    from server_decode import decode_jpeg

    ref = json.loads(selftest_file(name).read_text())
    out = []
    for case in ref["cases"]:
        obs = {"images": {k: decode_jpeg(base64.b64decode(b)) for k, b in ref["images_jpeg"].items()},
               "state": np.asarray(ref["state"], np.float32), "prompt": ref["prompt"]}
        if case.get("prefix_length"):
            obs["action_prefix"] = np.asarray(case["action_prefix"], np.float32)
            obs["prefix_length"] = case["prefix_length"]
        out.append({"obs": obs, "actions": np.asarray(case["actions"], np.float64), "name": case["name"]})
    return out, ref["bounds"]


def compare(a, b) -> dict:
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"mean_abs": round(float(d.mean()), 6), "max_abs": round(float(d.max()), 5),
            "finite": bool(np.isfinite(a).all())}


def selftest(name: str, stock: Policy, fast: Policy) -> tuple[bool, dict]:
    """Each stored case on the stock path (against the stored actions) and twice on the fast path (against stock)."""
    cases, bounds = selftest_cases(name)
    detail, ok = {}, True
    for case in cases:
        s = stock.infer(dict(case["obs"]))["actions"]
        vs_stored = compare(s, case["actions"])
        runs = [compare(fast.infer(dict(case["obs"]))["actions"], s) for _ in range(2)]
        worst = max(runs, key=lambda r: r["max_abs"])
        passed = worst["finite"] and worst["max_abs"] <= bounds["fast_max"] and worst["mean_abs"] <= bounds["fast_mean"]
        detail[case["name"]] = {"stock_vs_stored": vs_stored, "fast_vs_stock": worst, "passed": passed}
        if vs_stored["max_abs"] > bounds["stored_max"]:
            detail[case["name"]]["warning"] = "the stock path differs from the stored stock actions"
        ok = ok and passed
    return ok, detail


def build_fast(name, stock, deploy_policy, infer_observation, gpu, reason, label, compiled):
    """(fast Policy, "") when the boot self-test passes, else (None, why); the stock path is untouched either way."""
    import torch

    t = time.time()
    pathlib.Path(FAST_FAILED).write_text(f"fast path build started ({label})")
    fast = None
    try:
        import abc_fast

        fast = Policy(name, abc_fast.FastDeployPolicy(deploy_policy, compile_body=compiled), infer_observation,
                      "fast", reason)
        ok, detail = selftest(name, stock, fast)
        log("fast_selftest", variant=label, passed=ok, s=round(time.time() - t, 1), **detail)
        if ok:
            worst = max(d["fast_vs_stock"]["max_abs"] for d in detail.values())
            fast.metadata["path_reason"] = why = (f"{label} on {gpu}; boot self-test passed (max difference from "
                                                  f"stock {worst:g}, built in {time.time() - t:.0f} s)")
            log("path", path="fast", variant=label, reason=why)
            return fast, ""
        why = f"{label}: self-test failed: {json.dumps(detail)[:300]}"
    except Exception as e:  # noqa: BLE001  (any failure: try the next variant, then stock)
        why = f"{label} failed: {e!r}"[:300]
    finally:
        pathlib.Path(FAST_FAILED).unlink(missing_ok=True)
    del fast
    torch.cuda.empty_cache()
    return None, why


def load() -> Policy:
    from deploy.websocket_server import infer_observation

    name = os.environ["RS_MODEL"]
    t = time.time()
    deploy_policy = load_stock(weights_dir() / VARIANTS[name]["file"])
    log("stock_loaded", s=round(time.time() - t, 1))
    stock = Policy(name, deploy_policy, infer_observation)
    gpu = pgs_server.gpu_info("name").splitlines()[0].strip() if pgs_server.gpu_info("name") else ""
    path_, reason = decide(name, gpu)
    if path_ == "fast":
        compile_body = os.environ.get("RS_FAST_COMPILE", "1") == "1"
        if compile_body:  # inductor's compiled kernels on the volume: only a GPU type's first start compiles
            cache = pathlib.Path(os.environ.get("RS_COMPILE_CACHE", "/runpod-volume/pgs/abc_vla/inductor"))
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / gpu.replace(" ", "_")))
        failures = []
        for label, compiled in ((("compiled graph body", True),) if compile_body else ()) + (("CUDA graph", False),):
            fast, why = build_fast(name, stock, deploy_policy, infer_observation, gpu, reason, label, compiled)
            if fast is not None:
                return fast
            failures.append(why)
        reason = "; ".join(failures)[:600]
        log("fallback", reason=reason)
    stock.metadata.update(path="stock", path_reason=reason)
    log("path", path="stock", reason=reason)
    return stock


def make_example() -> dict:
    """Warm-up input: black frames at the client's jpeg size (224x224), zero state, the default prompt."""
    v = VARIANTS[os.environ["RS_MODEL"]]
    images = {cam: np.zeros((224, 224, 3), np.uint8) for cam in v["cameras"]}
    return {"images": images, "state": np.zeros(v["state_dim"], np.float32), "prompt": "throw plastic bottles in bin"}


def main() -> None:
    prepare()  # before run_worker, which waits for RS_CKPT_DIR's marker
    pgs_server.run_worker(load, make_example)


if __name__ == "__main__":
    main()
