"""The fast model path: realtime-vla (github.com/dexmal/realtime-vla, MIT, commit fastpath.RTVLA_COMMIT, cloned by
boot.sh to SRC) behind openpi's own tokenizer and output transforms, mapped onto openpi's pi05_libero.
f32_state_graph is adapted from realtime-vla's Pi05Inference.record_run, Copyright (c) 2025 Dexmal, MIT License (the
license text is at the top of rtvla_opt/pi05_infer.py and in NOTICE).

Same checkpoint, same bf16 weights (realtime-vla's converter, no quantization), same 10 flow steps, two real views
(openpi's third LIBERO image is all zero and masked, so its tokens never count), PROMPT_SLOTS prompt slots (LIBERO's
prompts have at most 21 tokens; a longer prompt raises an error instead of being cut). Two changes from realtime-vla as
shipped: the flow state stays float32 between the steps as openpi keeps it (f32_state_graph), and the overlapped graph
of its patched pi05_infer.py (rtvla_opt/, two marked edits) runs the prefix on a side stream. The unused prompt slots are
zeroed on every call (write_prompt): they are masked, but 0 x NaN = NaN, and an early version that left them as
torch.empty gave NaN or garbage actions in 1 of 4 processes.

The weights are realtime-vla's converter output (convert_from_jax_pi05.py: a dict of bf16 tensors, "the checkpoint"
below): `restore` and `convert` make it from openpi's Orbax checkpoint in the serving process, `load` reads the pickle a
first start saved (`save`) to the volume.

    python fast_policy.py warm            compile the kernels into $RS_FAST_DIR/triton_cache (boot.sh, first start)
    python fast_policy.py save CKPT PKL   realtime-vla's converter writing its pickle (serve.py, after ready)
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import sys
import time

SRC = os.environ.get("RS_RTVLA_SRC", "/opt/rs/realtime-vla")
OPT = pathlib.Path(__file__).resolve().parent / "rtvla_opt"  # the patched pi05_infer.py, found before SRC's
VIEWS, CHUNK, PROMPT_SLOTS = 2, 10, 32
MAX_PROMPTS = 256  # embedded prompts kept on the GPU (one per distinct prompt, about 128 kB each)
LLM = "PaliGemma/llm/layers/"
VIT = "PaliGemma/img/Transformer/encoderblock/"


def check_prompt(n: int, slots: int) -> None:
    if n > slots:
        raise ValueError(f"the prompt has {n} tokens; the fast path serves up to {slots} "
                         "(`pgs up MODEL --stock-path` serves longer prompts)")


def write_prompt(encoder_x, start: int, slots: int, embeds, n: int) -> None:
    """Put a prompt's n embeddings into the graph's prompt slots and zero the unused ones, on every call."""
    check_prompt(n, slots)
    encoder_x[start:start + n] = embeds
    encoder_x[start + n:start + slots] = 0


def converter():
    """realtime-vla's convert_from_jax_pi05 module (cloned by boot.sh); importing it imports orbax and transformers."""
    sys.path.insert(0, SRC)
    import convert_from_jax_pi05  # noqa: E402

    return convert_from_jax_pi05


def restore(ckpt_dir: str) -> dict:
    """openpi's Orbax checkpoint as float32 arrays on the host, by the converter's own load_jax_weights (JAX on the
    CPU); no GPU use."""
    return converter().load_jax_weights(ckpt_dir)


def convert(p: dict, log=lambda phase, **kw: None) -> dict:
    """realtime-vla's convert_from_jax_pi05.py at RTVLA_COMMIT on `restore`'s arrays, with its float32 arithmetic and
    bf16 rounding done on the GPU instead of in numpy: convert_weights_pi05 op for op (the same transposes and
    reshapes; the RMSNorm scales folded in float32 as w * (1 + scale); float32 to bf16 round-to-nearest-even, as torch
    does on the CPU) and its own prepare_adarms_cond. Correctly rounded float32 multiplies and the same rounding give
    the same bf16 weights (all 45 tensors compared with the converter's output: identical). Returns
    the converter's dict with the tensors on the GPU; the converter's "language_embeds" (one baked prompt) is a
    placeholder here: Rtvla replaces it with its prompt slots either way. Empties `p`."""
    import numpy as np
    import torch

    C = converter()
    t0, took = time.time(), {"to_numpy": 0.0, "to_gpu": 0.0}

    def f32(path: str):
        node = p
        for key in path.split("/"):
            node = node[key]
        t = time.time()
        a = np.asarray(node["value"])
        took["to_numpy"] += time.time() - t
        if a.dtype != np.float32:  # the converter's torch.tensor(..., dtype=bfloat16) path assumes float32 leaves
            raise TypeError(f"{path} is {a.dtype}, not float32")
        t = time.time()
        x = torch.as_tensor(a).cuda()
        took["to_gpu"] += time.time() - t
        return x

    def bf(x):
        return x.to(torch.bfloat16).contiguous()

    w = {}
    w["vision_patch_embedding_w"] = bf(f32("PaliGemma/img/embedding/kernel"))
    w["vision_patch_embedding_b"] = bf(f32("PaliGemma/img/embedding/bias"))
    w["vision_position_embedding"] = bf(f32("PaliGemma/img/pos_embedding")).squeeze()
    att = VIT + "MultiHeadDotProductAttention_0/"
    w["vision_attn_qkv_w"] = torch.cat([bf(f32(att + f"{n}/kernel")).flatten(2) for n in ("query", "key", "value")], 2)
    w["vision_attn_qkv_b"] = torch.cat([bf(f32(att + f"{n}/bias")).flatten(1) for n in ("query", "key", "value")], 1)
    w["vision_attn_o_w"] = bf(f32(att + "out/kernel").flatten(1, -2))
    w["vision_attn_o_b"] = bf(f32(att + "out/bias").flatten(1))
    for name, layer in (("up", "Dense_0"), ("down", "Dense_1")):
        w[f"vision_ffn_{name}_w"] = bf(f32(VIT + f"MlpBlock_0/{layer}/kernel"))
        w[f"vision_ffn_{name}_b"] = bf(f32(VIT + f"MlpBlock_0/{layer}/bias"))
    for name, layer in (("pre_attn", "LayerNorm_0"), ("pre_ffn", "LayerNorm_1")):
        w[f"vision_{name}_norm_w"] = bf(f32(VIT + f"{layer}/scale"))
        w[f"vision_{name}_norm_b"] = bf(f32(VIT + f"{layer}/bias"))
    w["vision_final_norm_w"] = bf(f32("PaliGemma/img/Transformer/encoder_norm/scale"))
    w["vision_final_norm_b"] = bf(f32("PaliGemma/img/Transformer/encoder_norm/bias"))
    w["encoder_multi_modal_projector_w"] = bf(f32("PaliGemma/img/head/kernel"))
    w["encoder_multi_modal_projector_b"] = bf(f32("PaliGemma/img/head/bias"))
    for i, dim in (("", 2048), ("_1", 1024)):  # the PaliGemma expert ("encoder"), then the action expert ("decoder")
        part = "encoder" if not i else "decoder"
        q = f32(LLM + f"attn/q_einsum{i}/w").permute(0, 2, 1, 3).reshape(18, dim, 8 * 256)
        kv = f32(LLM + f"attn/kv_einsum{i}/w")
        k, v = kv[:, 0, 0], kv[:, 1, 0]
        if not i:
            scale = 1 + f32(LLM + "pre_attention_norm/scale")[:, :, None]
            q, k, v = q * scale, k * scale, v * scale
        q = q.reshape(18, dim, 8, 2, 128).permute(0, 1, 2, 4, 3).reshape(18, dim, 2048)
        k = k.reshape(18, dim, 2, 128).permute(0, 1, 3, 2).reshape(18, dim, 256)
        w[f"{part}_attn_qkv_w"] = bf(torch.cat([q, k, v], 2))
        del q, kv, k, v
        w[f"{part}_attn_o_w"] = bf(f32(LLM + f"attn/attn_vec_einsum{i}/w").reshape(18, 8 * 256, dim))
        gating = f32(LLM + f"mlp{i}/gating_einsum")
        scale = 1 + f32(LLM + "pre_ffw_norm/scale")[:, :, None] if not i else 1
        w[f"{part}_ffn_gate_w"] = bf(gating[:, 0] * scale)
        w[f"{part}_ffn_up_w"] = bf(gating[:, 1] * scale)
        del gating
        w[f"{part}_ffn_down_w"] = bf(f32(LLM + f"mlp{i}/linear"))
    for name, layer in (("pre_attn", "pre_attention_norm_1"), ("pre_ffn", "pre_ffw_norm_1")):
        w[f"decoder_{name}_norm_mod_w"] = bf(f32(LLM + f"{layer}/Dense_0/kernel"))
        w[f"decoder_{name}_norm_mod_b"] = bf(f32(LLM + f"{layer}/Dense_0/bias"))
    for name in ("time_mlp_in", "time_mlp_out", "action_in_proj", "action_out_proj"):
        w[f"decoder_{name}_w"] = bf(f32(f"{name}/kernel"))
        w[f"decoder_{name}_b"] = bf(f32(f"{name}/bias"))
    w["decoder_final_norm_mod_w"] = bf(f32("PaliGemma/llm/final_norm_1/Dense_0/kernel"))
    w["decoder_final_norm_mod_b"] = bf(f32("PaliGemma/llm/final_norm_1/Dense_0/bias"))
    w["decoder_time_embeds"] = C.prepare_adarms_cond(num_steps=10).contiguous()
    w["embedding_weight"] = bf(f32("PaliGemma/llm/embedder/input_embedding"))
    w["language_embeds"] = torch.zeros(1, 2048, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    took["ops"] = time.time() - t0 - took["to_numpy"] - took["to_gpu"]
    t = time.time()
    p.clear()  # the float32 arrays (12.4 GB on the host); no gc.collect(): they hold no cycles
    took["free"] = time.time() - t
    log("fast_gpu_convert", **{f"{k}_s": round(v, 1) for k, v in took.items()})
    return w


def load(pkl: str) -> dict:
    with open(pkl, "rb") as f:
        return pickle.load(f)


def save(ckpt_dir: str, pkl: str) -> None:
    """realtime-vla's converter itself (its __main__, numpy on the CPU, 35 to 44 s) writing its pickle: what `load`
    reads on later starts. serve.py runs it as its own low-priority process once the fast path serves (`fast_policy.py
    save`): written by the serving process, the 6.7 GB of copies and pickling held its GIL and the GPU's default stream
    and stalled requests for up to 1.7 s. Same bf16 weights as `convert`'s: all 45 tensors
    compared identical. The converter bakes one prompt in; its tokenizer is replaced (only the shape is used)."""
    import runpy

    import torch
    import transformers

    class Tok:
        def __call__(self, *a, **k):
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

    transformers.AutoTokenizer.from_pretrained = lambda *a, **k: Tok()
    sys.argv = ["convert_from_jax_pi05.py", "--jax_path", ckpt_dir, "--output", pkl + ".tmp", "--prompt", "x",
                "--tokenizer_path", "unused"]
    runpy.run_path(os.path.join(SRC, "convert_from_jax_pi05.py"), run_name="__main__")
    with open(pkl + ".tmp", "rb") as f:
        os.fsync(f.fileno())
    os.replace(pkl + ".tmp", pkl)


def warm_up() -> None:
    """Compile realtime-vla's Triton kernels into TRITON_CACHE_DIR and load the tokenizer, by building the graph once
    with placeholder weights (values do not matter for that; the weights stay uninitialized except the few the graph
    code reads from the checkpoint). boot.sh runs it as its own process (`fast_policy.py warm`) on a first start, while
    serve.py imports and the checkpoint downloads; serve.py's real build then finds the kernels compiled (12 s of
    compiling otherwise)."""
    import torch

    z = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16)  # noqa: E731
    Rtvla({"embedding_weight": z(1, 2048), "decoder_action_in_proj_w": z(32, 1024), "decoder_action_in_proj_b": z(1024),
           "decoder_action_out_proj_w": z(1024, 32), "decoder_action_out_proj_b": z(32)}, None)



def f32_state_graph(inf, ckpt):
    """realtime-vla's graph with the flow state x_t in float32 between the 10 steps, as in openpi's sample_actions:
    x_t float32 (initial noise not rounded), action_in_proj in float32, action_out_proj with the unfolded bf16 weights,
    dt * v_t in bf16, the sum in float32. Every other kernel is realtime-vla's. Returns (state buffer, CUDA graph,
    every tensor the graph uses that was allocated before its capture): a CUDA graph does not own those, so the caller
    must hold them for as long as it replays the graph. Freed, their memory is reused by later allocations and the graph
    computes with whatever lands there (seen here: deterministic garbage actions that depended on the previous call;
    an earlier copy kept them alive only through a debugging hook that held the closures)."""
    import pi05_infer as P
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # the float32 in-projection in IEEE float32
    W, B, L = inf.weights, inf.buffers, inf.encoder_seq_len
    xt = torch.zeros(CHUNK, 32, dtype=torch.float32, device="cuda")
    w_in = ckpt["decoder_action_in_proj_w"].to("cuda", torch.float32)
    b_in = ckpt["decoder_action_in_proj_b"].to("cuda", torch.float32)
    w_out = ckpt["decoder_action_out_proj_w"].to("cuda")
    b_out = ckpt["decoder_action_out_proj_b"].to("cuda")
    tmp = torch.empty(CHUNK, 1024, dtype=torch.float32, device="cuda")
    v = torch.empty(CHUNK, 32, dtype=torch.bfloat16, device="cuda")
    dt = torch.tensor(-1.0 / 10, dtype=torch.bfloat16, device="cuda")

    def in_proj(x, weight, bias, out):  # realtime-vla passes its bf16 state buffer; the float32 state is used instead
        torch.addmm(b_in, xt, w_in, out=tmp)
        out.copy_(tmp)

    def out_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style, weight, bias, out, res):
        P.adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style)
        torch.addmm(b_out, x_normed, w_out, out=v)
        v.mul_(dt)
        xt.add_(v)

    def model():
        if getattr(inf, "optimized", False):  # as realtime-vla's record_run: the prefix on a side stream, the
            main = torch.cuda.current_stream()  # decoder's first step waits per layer on the prefix's events
            inf.encoder_stream.wait_stream(main)
            with torch.cuda.stream(inf.encoder_stream):
                P.vision_encoder(W, B, VIEWS)
                P.transformer_encoder(W, B, L)
            P.transformer_decoder(W, B, L, 10)
            main.wait_stream(inf.encoder_stream)
        else:
            P.vision_encoder(W, B, VIEWS)
            P.transformer_encoder(W, B, L)
            P.transformer_decoder(W, B, L, 10)

    saved = P.matmul_k_32_1024_bias, P.adarms_matmul_k_1024_32_bias_res
    P.matmul_k_32_1024_bias, P.adarms_matmul_k_1024_32_bias_res = in_proj, out_proj
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            model()
        torch.cuda.current_stream().wait_stream(s)
    finally:
        P.matmul_k_32_1024_bias, P.adarms_matmul_k_1024_32_bias_res = saved
    return xt, g, (xt, w_in, b_in, w_out, b_out, tmp, v, dt)


class Rtvla:
    """infer(obs, noise) -> openpi's reply dict; `output_transform` is the stock policy's (unnormalize, first 7 dims).
    `ckpt`: the converter's dict (`convert` or `load`); it is not changed."""

    def __init__(self, ckpt: dict, output_transform, slots: int = PROMPT_SLOTS):
        import numpy as np
        import torch

        sys.path.insert(0, SRC)
        sys.path.insert(0, str(OPT))
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi_client.image_tools import resize_with_pad
        from pi05_infer import Pi05Inference

        self.np, self.torch, self._resize = np, torch, resize_with_pad
        ckpt = dict(ckpt)
        self._emb = ckpt.pop("embedding_weight").to("cuda")
        ckpt["language_embeds"] = torch.zeros(slots, 2048, dtype=torch.bfloat16)
        t = time.time()
        self._inf = Pi05Inference(ckpt, num_views=VIEWS, chunk_size=CHUNK, discrete_state_input=False)
        torch.cuda.synchronize()
        self._state, self._graph, self._graph_tensors = f32_state_graph(self._inf, ckpt)  # held: see there
        torch.cuda.synchronize()
        self.capture_s = round(time.time() - t, 1)
        self.overlapped = bool(getattr(self._inf, "optimized", False))
        del ckpt
        self._slots, self._start = slots, VIEWS * 256
        self._scale = torch.tensor(math.sqrt(2048), dtype=torch.bfloat16, device="cuda")
        self._tok, self._prompts = PaligemmaTokenizer(200), {}
        self._pin = torch.empty((VIEWS, 224, 224, 3), dtype=torch.uint8).pin_memory()
        self._u8 = torch.empty((VIEWS, 224, 224, 3), dtype=torch.uint8, device="cuda")
        self._output_transform, self.stages = output_transform, {}

    def infer(self, obs: dict, noise=None) -> dict:
        np, torch = self.np, self.torch
        t0 = time.perf_counter()
        buf = self._inf.buffers
        with torch.no_grad():
            p = obs["prompt"]
            if p not in self._prompts:
                ids, mask = self._tok.tokenize(p)
                n = int(mask.sum())
                check_prompt(n, self._slots)
                if len(self._prompts) >= MAX_PROMPTS:
                    self._prompts.clear()
                self._prompts[p] = (self._emb[torch.as_tensor(ids[:n], device="cuda")] * self._scale, n)
            e, n = self._prompts[p]
            for v, k in enumerate(("observation/image", "observation/wrist_image")):
                im = np.asarray(obs[k])
                if im.shape != (224, 224, 3) or im.dtype != np.uint8:  # what LiberoInputs + ResizeImages do
                    im = self._resize(np.asarray(im * 255 if np.issubdtype(im.dtype, np.floating) else im, np.uint8), 224, 224)
                self._pin[v].numpy()[...] = im
            self._u8.copy_(self._pin, non_blocking=True)
            buf["observation_images_normalized"].copy_(self._u8.to(torch.float32) / 255.0 * 2.0 - 1.0)
            write_prompt(buf["encoder_x"], self._start, self._slots, e, n)
            buf["valid_encoder_len"].fill_(self._start + n)
            buf["decoder_rope_weights"].copy_(self._inf._rope_table[self._start + n:self._start + n + CHUNK])
            if noise is None:
                self._state.normal_()
            else:
                self._state.copy_(torch.from_numpy(np.asarray(noise, np.float32)).to("cuda"))
            t1 = time.perf_counter()
            self._graph.replay()
            out = self._state.float().cpu().numpy()
        t2 = time.perf_counter()
        res = self._output_transform({"state": np.asarray(obs["observation/state"], np.float32), "actions": out})
        if not np.isfinite(res["actions"]).all():  # never send non-finite actions to a robot
            raise RuntimeError("the fast path produced non-finite actions")
        res["policy_timing"] = {"infer_ms": (t2 - t1) * 1e3}
        self.stages = {"pre": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3, "post": (time.perf_counter() - t2) * 1e3}
        return res


class FastPolicy:
    """The openpi Policy interface over Rtvla. `_rng` as openpi's Policy: None (the default) draws the initial noise on
    the GPU; once set (serve.py's rs_seed hook sets it), each call splits it and draws the same noise openpi would."""

    def __init__(self, rt: Rtvla, action_horizon: int, action_dim: int, metadata: dict | None = None):
        import jax

        self._rt, self._rng, self.metadata = rt, None, dict(metadata or {})
        self._noise = jax.jit(lambda key: jax.random.normal(key, (1, action_horizon, action_dim))[0])
        self._split = jax.jit(jax.random.split)

    def infer(self, obs: dict) -> dict:
        noise = None
        if self._rng is not None:
            import numpy as np

            self._rng, sample = self._split(self._rng)
            noise = np.asarray(self._noise(sample))
        return self._rt.infer(obs, noise)


def log(phase: str, **kw) -> None:
    print(json.dumps({"rs": phase, "s_since_boot": round(time.time() - float(os.environ.get("RS_BOOT_T") or 0), 3),
                      **kw}), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["warm"]:  # boot.sh, first start: see warm_up
    os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ.get("RS_FAST_DIR", "/rs_local/fast"), "triton_cache")
    os.environ["JAX_PLATFORMS"] = "cpu"  # openpi's tokenizer module imports JAX; the GPU is for the kernels
    t0 = time.time()
    log("fast_warm_start")
    warm_up()
    log("fast_warm", s=round(time.time() - t0, 1))
elif __name__ == "__main__" and sys.argv[1:2] == ["save"]:  # serve.py, after ready on a first start: see save
    save(sys.argv[2], sys.argv[3])
