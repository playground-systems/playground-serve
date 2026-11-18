"""GR00T N1.7 (NVIDIA) worker: the LIBERO checkpoints behind Playground Serve's websocket protocol (pgs_server.py).

Two paths; the metadata's "path" and "path_reason" and the log line {"rs": "path"} say which one serves and why.
stock  as below, unchanged.
fast   gr00t_fast.py: the same call with the action head's 4 flow steps run as one CUDA graph, from the same noise
       draw. Only for models and GPUs in FAST, and only after a boot self-test against the stock path
       (selftest_<model>.json); else stock serves.

Model calls follow the repo's own LIBERO server at COMMIT (examples/LIBERO/README.md:126-129):
gr00t/eval/run_gr00t_server.py builds Gr00tPolicy(embodiment_tag, model_path, device, strict) (:105-110, defaults
device="cuda" :64 and strict=True :84) and, for --use-sim-policy-wrapper, wraps it in Gr00tSimPolicyWrapper (:162-165);
its PolicyServer answers each request with policy.get_action(observation) (gr00t/policy/server_client.py:176, 262-265).
Gr00tPolicy loads the checkpoint with AutoModel.from_pretrained and casts it to bf16 (gr00t/policy/gr00t_policy.py);
the 4 denoising steps come from the checkpoint's config.json.

Observation (msgpack-numpy dict): the official request keys (gr00t_policy.py; the LIBERO env's observation,
gr00t/eval/sim/LIBERO/libero_env.py:144-158) without the batch and time dims, which infer() adds back:
    "video.image", "video.wrist_image": HxWx3 uint8 RGB (LIBERO renders 256x256, libero_env.py:104-109),
    "state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw": 1 value each, "state.gripper": 2 values,
    "annotation.human.action.task_description": str;
    optional "rs_seed": int, the flow noise from torch.manual_seed(rs_seed) for this call only (checks).
Reply: {"actions": float32 (16, 7)}, columns x, y, z, roll, pitch, yaw, gripper (the order libero_env.py:171-182
concatenates them in), plus "action.<key>": (16, dim) each, the official reply without its batch dim.

The server does what the official server does and no more. Left to the robot client, as the official LIBERO env does
it: rotating both camera frames 180 degrees (libero_env.py:148-149), the eef state as position + axis-angle + gripper
qpos (:144-146), mapping the gripper action from [0, 1] to a binary -1/+1 and inverting it (:183-184), and executing 8
of the 16 actions per call (README.md:142).

Weights: hf://nvidia/GR00T-N1.7-LIBERO@<rev>/<suite>, only the files the policy reads (CHECKPOINT_FILES, 6.91 GB of
the suite's 36 GB), into RS_CKPT_DIR; until stage.py fetch takes hf:// URIs, prepare() lists them through the Hugging
Face API and runs stage.fetch. The gated backbone nvidia/Cosmos-Reason2-2B (processor and initial weights) is loaded by
name from the Hub at every start (gr00t/model/modules/qwen3_backbone.py:183, processing_gr00t_n1d7.py:129) into
HF_HOME on the volume: the endpoint needs HF_TOKEN from an account that accepted its terms.
Endpoint env: RS_MODEL (a VARIANTS key), RS_CHECKPOINT, RS_CKPT_DIR (boot.sh), HF_HOME, HF_TOKEN (secret).
"""

from __future__ import annotations

import base64
import contextlib
import fnmatch
import json
import os
import pathlib
import time
import urllib.parse
import urllib.request

import numpy as np

import pgs_server

COMMIT = "3b49067943eb1ab2eb2203b93d461c4390c78da3"  # github.com/NVIDIA/Isaac-GR00T, 2026-11-10
REPO, REVISION = "nvidia/GR00T-N1.7-LIBERO", "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"  # HF API, read 2026-11-16
BACKBONE = "nvidia/Cosmos-Reason2-2B"  # gated (automatic approval on Hugging Face), loaded by name at every start
EMBODIMENT_TAG = "LIBERO_PANDA"  # value "libero_sim" (gr00t/data/embodiment_tags.py)
LICENSE = "code Apache-2.0; weights NVIDIA Open Model License; backbone nvidia/Cosmos-Reason2-2B gated (its own terms)"
# The files the policy reads from one suite folder, as examples/LIBERO/README.md downloads them; the folder's optimizer
# and trainer states are not needed.
CHECKPOINT_FILES = ("config.json", "embodiment_id.json", "model-*.safetensors", "model.safetensors.index.json",
                    "processor_config.json", "statistics.json")
VIDEO_KEYS = ("image", "wrist_image")
STATE_DIMS = {"x": 1, "y": 1, "z": 1, "roll": 1, "pitch": 1, "yaw": 1, "gripper": 2}
ACTION_DIMS = {"x": 1, "y": 1, "z": 1, "roll": 1, "pitch": 1, "yaw": 1, "gripper": 1}
PROMPT_KEY = "annotation.human.action.task_description"
HORIZON, HZ, IMAGE_SIZE = 16, 20, (256, 256)
# One checkpoint folder per LIBERO suite, all four in the same repo with identical configs.
VARIANTS = {
    "gr00t_n17_libero": dict(suite="libero_spatial"),
    "gr00t_n17_libero_goal": dict(suite="libero_goal"),
    "gr00t_n17_libero_object": dict(suite="libero_object"),
    "gr00t_n17_libero_10": dict(suite="libero_10"),
}
KEYS = (*(f"video.{k}" for k in VIDEO_KEYS), *(f"state.{k}" for k in STATE_DIMS), PROMPT_KEY)
HF = "https://huggingface.co"
# Model -> GPU names (nvidia-smi) where the fast path was measured served against stock (NOTES.md).
FAST = {"gr00t_n17_libero": ("NVIDIA GeForce RTX 4090",)}
FAST_FAILED = "/tmp/rs_fast_failed"
HERE = pathlib.Path(__file__).resolve().parent
log = pgs_server.log


class Policy:
    def __init__(self, name: str, policy, fast=None, path: str = "stock", reason: str = ""):
        """`policy`: Gr00tSimPolicyWrapper(Gr00tPolicy(...)), or anything with its get_action(observation). `fast`:
        gr00t_fast.FastActionHead installed on that policy's action head; its active() turns it on for a call."""
        self.name, self.variant, self.policy, self.fast = name, VARIANTS[name], policy, fast
        self.metadata = {"model": name, "family": "gr00t_n17", "checkpoint": f"hf://{REPO}@{REVISION}",
                         "subfolder": self.variant["suite"], "commit": COMMIT,
                         "action_shape": [HORIZON, sum(ACTION_DIMS.values())], "action_hz": HZ, "keys": list(KEYS),
                         "action_keys": list(ACTION_DIMS), "state_dim": sum(STATE_DIMS.values()),
                         "prompt_key": PROMPT_KEY, "image_size": list(IMAGE_SIZE), "history": 0, "license": LICENSE,
                         "path": path, "path_reason": reason}

    def infer(self, obs: dict) -> dict:
        # The batch the official LIBERO client sends: video (B, T, H, W, 3) uint8, state (B, T, D) float32, language
        # a (B,) sequence of str, with B = T = 1.
        batch = {}
        for key in VIDEO_KEYS:
            image = np.asarray(obs[f"video.{key}"])
            if image.dtype != np.uint8 or image.shape[-1] != 3:
                raise ValueError(f"video.{key} must be H x W x 3 uint8, got {image.dtype} {image.shape}")
            batch[f"video.{key}"] = image.reshape(1, 1, *image.shape[-3:])
        for key, dim in STATE_DIMS.items():
            state = np.asarray(obs[f"state.{key}"], dtype=np.float32)
            if state.size != dim:
                raise ValueError(f"state.{key} must hold {dim} value(s), got shape {state.shape}")
            batch[f"state.{key}"] = state.reshape(1, 1, dim)
        batch[PROMPT_KEY] = (str(obs[PROMPT_KEY]),)
        t = time.perf_counter()
        seed = obs.get("rs_seed")
        with self.fast.active() if self.fast is not None else contextlib.nullcontext():
            if seed is None:
                action, _info = self.policy.get_action(batch)
            else:  # this call's flow noise from torch.manual_seed(rs_seed); torch's own stream is left where it was
                import torch

                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(int(seed))
                    action, _info = self.policy.get_action(batch)
        per_key = {f"action.{k}": np.asarray(action[f"action.{k}"], dtype=np.float32)[0] for k in ACTION_DIMS}
        actions = np.concatenate(list(per_key.values()), axis=-1)
        if not np.isfinite(actions).all():
            raise ValueError("the policy returned non-finite actions")
        # the policy call alone; the server's infer_ms adds the JPEG decode before it
        return {"actions": actions, **per_key, "policy_timing": {"policy_ms": (time.perf_counter() - t) * 1e3}}


# ----- weights
def weights_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("RS_CKPT_DIR") or "/runpod-volume/pgs/weights/gr00t_n17")


def parse_checkpoint(uri: str) -> tuple[str, str, str]:
    """'hf://<org>/<repo>@<revision>/<subfolder>' -> (repo, revision, subfolder)."""
    repo, _, rest = uri.removeprefix("hf://").partition("@")
    revision, _, subfolder = rest.partition("/")
    return repo, revision, subfolder


def checkpoint_uri(name: str) -> str:
    return f"hf://{REPO}@{REVISION}/{VARIANTS[name]['suite']}"


def hf_listing(repo: str, revision: str, subfolder: str) -> list[tuple[str, int, str]]:
    """(path under the subfolder, size, download URL) of the CHECKPOINT_FILES in one folder of a Hugging Face repo."""
    url = f"{HF}/api/models/{repo}/tree/{revision}/{subfolder}"
    headers = {"User-Agent": "playground-serve-worker"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        entries = json.load(r)
    out = []
    for e in entries:
        rel = e["path"][len(subfolder) + 1:]
        if e.get("type") == "file" and any(fnmatch.fnmatch(rel, p) for p in CHECKPOINT_FILES):
            out.append((rel, int(e["size"]), f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(e['path'])}"))
    return out


def prepare() -> pathlib.Path:
    import stage  # /rs/stage.py

    name = os.environ["RS_MODEL"]
    dest = weights_dir()
    if os.environ.get("RS_CKPT_DIR"):  # boot.sh's stage.py fetch may be writing it: let it end first
        pgs_server.wait_for_fetch(dest)
    if not stage.complete(dest):
        (dest / ".rs_failed").unlink(missing_ok=True)  # boot.sh's stage.py fetch takes gs:// URIs only
        t = time.time()
        uri = checkpoint_uri(name)
        log("ckpt_fetch_start", uri=uri, dest=str(dest))
        stage.listing = lambda _uri: hf_listing(*parse_checkpoint(uri))
        files = stage.fetch(uri, dest)
        n = sum(files.values())
        log("ckpt_fetch_done", bytes=n, files=len(files), s=round(time.time() - t, 1),
            mb_s=round(n / (time.time() - t) / 1e6))
    return dest


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
        return "stock", ("an earlier start in this container crashed while it built the fast path: "
                         f"{pathlib.Path(FAST_FAILED).read_text()[:200]}")
    return "fast", f"CUDA graph path on {gpu}, if its boot self-test passes"  # load() updates it after the test


def selftest_file(name: str) -> pathlib.Path:
    return HERE / f"selftest_{name}.json"


def selftest_cases(name: str) -> tuple[list[dict], dict]:
    """The stored observation (frames as JPEG, decoded as server_decode does) with the stock actions per seed."""
    from server_decode import decode_jpeg

    ref = json.loads(selftest_file(name).read_text())
    obs = {f"video.{k}": decode_jpeg(base64.b64decode(b)) for k, b in ref["frames_jpeg"].items()}
    state, start = np.asarray(ref["state"], np.float32), 0
    for key, dim in STATE_DIMS.items():
        obs[f"state.{key}"] = state[start:start + dim]
        start += dim
    obs[PROMPT_KEY] = ref["task"]
    out = [{"obs": dict(obs, rs_seed=case["seed"]), "actions": np.asarray(case["actions"], np.float64),
            "name": case["name"]} for case in ref["cases"]]
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


def load_stock(path: pathlib.Path):
    """run_gr00t_server.py's policy: the tag as a string (resolved case-insensitively), device "cuda", strict True,
    wrapped for --use-sim-policy-wrapper."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    return Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EMBODIMENT_TAG, model_path=str(path), device="cuda",
                                             strict=True))


def load() -> Policy:
    name = os.environ["RS_MODEL"]
    if not os.environ.get("HF_TOKEN"):
        log("hf_token_missing", backbone=BACKBONE)
    t = time.time()
    wrapper = load_stock(weights_dir())
    log("stock_loaded", s=round(time.time() - t, 1))
    stock = Policy(name, wrapper)
    gpu = (pgs_server.gpu_info("name").splitlines() or [""])[0].strip()
    path, reason = decide(name, gpu)
    if path == "fast":
        t = time.time()
        pathlib.Path(FAST_FAILED).write_text("fast path build started")
        head = None
        try:
            import gr00t_fast

            head = wrapper.policy.model.action_head
            fast = Policy(name, wrapper, gr00t_fast.FastActionHead(head), "fast", reason)
            ok, detail = selftest(name, stock, fast)
            log("fast_selftest", passed=ok, s=round(time.time() - t, 1), **detail)
            if ok:
                worst = max(d["fast_vs_stock"]["max_abs"] for d in detail.values())
                fast.metadata["path_reason"] = reason = (f"CUDA graph path on {gpu}; boot self-test passed "
                                                         f"(max difference from stock {worst:g})")
                log("path", path="fast", reason=reason)
                return fast
            reason = f"fast path self-test failed: {json.dumps(detail)[:300]}"
        except Exception as e:  # noqa: BLE001  (any failure: serve stock, say why)
            reason = f"fast path failed: {e!r}"[:300]
        finally:
            pathlib.Path(FAST_FAILED).unlink(missing_ok=True)
        if head is not None:
            head.__dict__.pop("get_action_with_features", None)  # the stock method again
        import torch

        torch.cuda.empty_cache()
        log("fallback", reason=reason)
    stock.metadata.update(path="stock", path_reason=reason)
    log("path", path="stock", reason=reason)
    return stock


def make_example() -> dict:
    """Warm-up input: black 256x256 frames, zero state."""
    obs = {f"video.{k}": np.zeros((*IMAGE_SIZE, 3), np.uint8) for k in VIDEO_KEYS}
    obs.update({f"state.{k}": np.zeros(d, np.float32) for k, d in STATE_DIMS.items()})
    return {**obs, PROMPT_KEY: "pick up the black bowl between the plate and the ramekin and place it on the plate"}


def main() -> None:
    prepare()  # before run_worker, which waits for RS_CKPT_DIR's marker
    pgs_server.run_worker(load, make_example)


if __name__ == "__main__":
    main()
