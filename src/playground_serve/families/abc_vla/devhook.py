"""Bring-up only: bringup.py --dev ships this file as /rs/serve.py and the family's serve.py as /rs/family_serve.py.
`pgs up` never ships it. The worker starts as the family's main() does (prepare, load, warm up, serve) and also answers
{"rs_dev_exec": "<python source>"} with {"actions": [[0]], "out": <what the code printed>}. The code runs on the serving
thread (the one that loaded the model and captured any CUDA graph), with `policy` (the loaded Policy), `dev` (this
wrapper; set dev.policy to swap the served policy), `fs` (family_serve) and `np` in scope; globals persist
between calls.
The endpoint is reachable only with the account's Runpod key (the gateway checks it)."""

from __future__ import annotations

import contextlib
import io
import traceback

import numpy as np

import family_serve as fs
import pgs_server


class Dev:
    def __init__(self, policy):
        self.policy = policy
        self.metadata = {**(getattr(policy, "metadata", None) or {}), "dev": True}
        self.scope = {"policy": policy, "dev": self, "fs": fs, "np": np}

    def infer(self, obs: dict) -> dict:
        code = obs.get("rs_dev_exec") if isinstance(obs, dict) else None
        if code is None:
            return self.policy.infer(obs)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                exec(compile(code, "<rs_dev_exec>", "exec"), self.scope)
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
        return {"actions": np.zeros((1, 1), np.float32), "out": out.getvalue()}


if __name__ == "__main__":
    fs.prepare()
    pgs_server.run_worker(lambda: Dev(fs.load()), fs.make_example)
