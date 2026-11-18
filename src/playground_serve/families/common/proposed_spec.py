"""The registry entry a family's spec.py declares: playground_serve.models.ModelSpec's fields plus what a model outside
openpi needs. playground_serve.models finds every families/<family>/spec.py, imports it as
playground_serve.families.<family>.spec and reads its SPECS (a dict name -> ProposedSpec, or a list of them), so a new
family needs no change outside its own folder. spec.py runs in the `pgs` process: it may import the standard library,
numpy and this module, but not the worker side (pgs_server, torch, the model's code).

- family: the folder (families/<family>/); its files (all but spec.py, NOTES.md, __init__.py and the dev tools
  bringup.py and devhook.py) ship to the worker next to families/common/ and the openpi worker's health.py,
  server_decode.py and stage.py; files named verify* ship only for bring-up runs (env RS_VERIFY=1), and
  selftest_<name>.json only for the entry <name>. Python files ship without comments and docstrings (bundle.slim; line
  numbers kept). The folder name wins. Runpod refuses an endpoint request over about 100.3 kB (measured), and RS_FILES
  is most of it: tests/test_families.py keeps every entry 15 % under that.
- checkpoint: "hf://<repo>@<revision>", a gs:// directory or one https:// file; boot.sh downloads it with stage.py
  fetch into RS_CKPT_DIR (the volume's complete copy once it has one).
- env: extra endpoint env the family's serve.py reads (bundle.family_env adds RS_MODEL, RS_FAMILY, RS_CHECKPOINT,
  RS_ENV_TAR, RS_WEIGHTS, HF_HOME, HF_HUB_ENABLE_HF_TRANSFER=0, RS_FAST, PORT and PORT_HEALTH). RS_CKPT_FILES
  (comma-separated patterns) limits the checkpoint download to the files that match; RS_SELF_FETCH=1 skips it (the
  model's own loader downloads the checkpoint into HF_HOME; boot.sh sets no RS_CKPT_DIR).
- secrets: env names whose values `pgs up` takes from the user's machine (the environment variable, and for HF_TOKEN
  also the token file of `hf auth login`) into the endpoint's env; never in a spec, the state file or the output.
- action_shape: (horizon, action_dim) of one reply.
- robot: the embodiment (so101, yam, droid, libero, ...). Names are model plus robot (molmoact2_libero), as in the
  openpi registry (pi05_libero), because the robot changes the checkpoint and the observation format.
- gpu_validated: the date (YYYY-MM-DD) real weights served a real action on a GPU worker; "" or False = untested.
- license: the model's license note, shown by `pgs models` (a spec module's LICENSE constant serves when it is "").
- hosted_id: the same model and robot's id on a hosted service that serves it ("" where none); information only.
- experimental: slow or not fully verified (owner's label): `pgs models` and MODELS.md mark it, `pgs up` says so.
- min_cuda: the lowest host CUDA version the family's environment needs ("13.0" for CUDA 13 wheels, which need driver
  580 or newer), when it is above the image's (12.8); `pgs up` asks Runpod for hosts with at least that.
- cpu: a CPU worker in place of a GPU pool ("cpu3c-2": Runpod's flavor and vCPUs), for a family that runs no model on
  the worker (llm_agent); gpu_pools is then empty and the endpoint has no volume.
- agent: an llm_agent entry's route (provider, slug, title, reasoning effort, output limit, prices per million tokens).
"""

from __future__ import annotations

import dataclasses

# Same stock image as the openpi worker: the families bring their own Python and torch in the environment tar, so the
# image only provides python3 (health.py, stage.py), git, tar and the host's NVIDIA driver.
IMAGE = "runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404"
VOLUME = "/runpod-volume"
# Runpod serverless GPU pools and list prices (USD per hour per worker, read 2026-11-13 from the catalog; ADA_32_PRO
# 2026-11-16). gpu_pools lists them in order of preference: `pgs up` takes the first with stock.
POOL_PRICES = {
    "ADA_24": 1.34,       # RTX 4090, 24 GB
    "AMPERE_48": 1.22,    # A40 / A6000, 48 GB
    "ADA_48_PRO": 1.89,   # L40 / L40S / 6000 Ada, 48 GB
    "ADA_32_PRO": 1.89,   # RTX 5090, 32 GB
    "AMPERE_80": 2.99,    # A100, 80 GB
    "ADA_80_PRO": 5.89,   # H100, 80 GB
    "HOPPER_141": 6.99,   # H200, 141 GB
}


@dataclasses.dataclass(frozen=True)
class ProposedSpec:
    name: str
    family: str
    checkpoint: str
    volume_gb: int
    min_container_ram_gb: int
    gpu_pools: tuple[str, ...]
    action_shape: tuple[int, int]
    robot: str = ""
    image: str = IMAGE
    env: dict = dataclasses.field(default_factory=dict)
    secrets: tuple[str, ...] = ()
    gpu_validated: str | bool = ""
    license: str = ""
    hosted_id: str = ""
    experimental: bool = False
    min_cuda: str = ""
    cpu: str = ""  # a CPU worker (Runpod flavor and vCPUs, such as cpu3c-2) in place of gpu_pools, which is then empty
    agent: dict = dataclasses.field(default_factory=dict)  # an llm_agent entry's route: provider, slug, title, prices

    def endpoint_env(self) -> dict[str, str]:
        """The env the family's boot.sh and serve.py read, as playground_serve.bundle.family_env adds it (that map has
        these keys and values, plus RS_FILES, RS_RELEASE, RS_FAMILY, RS_WEIGHTS and RS_FAST; `pgs up` adds the
        secrets)."""
        return {"RS_MODEL": self.name, "RS_CHECKPOINT": self.checkpoint, "PORT": "8000", "PORT_HEALTH": "8001",
                "RS_ENV_TAR": f"{VOLUME}/pgs/{self.family}/env-secure-v2.tar", "HF_HOME": f"{VOLUME}/hf", **self.env}
