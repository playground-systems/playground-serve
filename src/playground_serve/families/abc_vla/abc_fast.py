"""Fast path of ABC-VLA: the stock model's own modules, weights, dtypes and shapes, run as one CUDA graph per prompt.

VLAInferencePolicy.infer (abc_minimal/policy.py, COMMIT) spends most of a single-observation call outside the GPU's
arithmetic: it tokenizes the prompt with sentencepiece on every call, builds 256 x 256 attention masks, embeds the
same text again, and launches about 6,000 kernels one by one from Python. FastEngine.infer computes the same function
with the same kernels:

- per prompt, once (official functions, same values): the token ids (tokenize_raw_input of `_raw_input`), the text
  embeddings (`_embed_text`), the attention masks of the ids padded to `fixed_seq_len` (`_attention_masks`), the
  rotary tables (`_freqs`) and the pool's padding mask;
- per call, outside the graph: the state normalization (numpy, as stock) and each camera's resize and pad
  (resize_pad_normalize_batch, as stock), copied into fixed input buffers with the noise and the RTC prefix;
- in one CUDA graph, captured from the official modules: `_normalize_images`, `_inject_state`, `_encode_images`
  (SigLIP, the soft-embedding norm and projection, masked_scatter), the sequence padded to 256 as `prepare` pads it,
  the Gemma decoder up to the feature layer, the fp32 attention pool and the fp32 diffusion head's 10 Euler steps
  (`DiffusionActionHead.sample`).

The kernels and shapes are the stock path's, so the actions are bit-identical to stock (measured on an RTX 4090: 0.0
difference on 4 real frames with zero noise, a random noise and an RTC prefix; NOTES.md). Measured and left out:
cutting the sequence at the prompt's last token (the padding gets zero attention) saved 0.7 ms of 34 and moved
actions by 0.002 to 0.003 mean; torch.compile under the graph saved 8 ms more but moved actions by 0.002 to 0.005
mean and took 126 s per cold start.

Two expressions cannot be captured as written, and run the same arithmetic from a constant made before the capture:
`_embed_text` and SigLIP's `gelu_tanh` each build a scalar tensor on the GPU with torch.tensor(...) in every call (a
host-to-device copy). The text embedding is computed once per prompt instead; gelu_tanh is replaced, on the module
instances and only while the graph is captured, by the same formula with that scalar made once (same dtype, same order
of operations). The stock path's modules are otherwise untouched.

Batched observations (state of shape (B, 14)) go to the stock policy unchanged.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import time
import warnings

import numpy as np
import torch

from abc_minimal.gemma.preprocessor import tokenize_raw_input
from abc_minimal.preprocess import normalize, resize_pad_normalize_batch, unnormalize
from deploy.policy.base import MAX_ABS_ACTION

MAX_GRAPHS = 16  # (prompt, prefix length) pairs kept; the least recently used one is dropped beyond that


def _gelu_tanh(x, c):
    """SigLIP's SiglipMLP.gelu_tanh with its constant torch.sqrt(torch.tensor(2.0 / torch.pi)) made once."""
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))


class FastEngine:
    """VLAInferencePolicy.infer for one observation."""

    def __init__(self, engine, max_graphs: int = MAX_GRAPHS, compile_body: bool = False):
        self.e, self.model, self.vla = engine, engine.model, engine.model.vla
        # compile_body: inductor compiles the graph body (fused kernels; not bit-identical, NOTES.md)
        self.body = self._body
        if compile_body:
            import torch._inductor.config as inductor_config

            inductor_config.emulate_precision_casts = True  # round as eager does between ops
            self.body = torch.compile(self._body, dynamic=False)
        self.device = engine.device
        if self.device.type != "cuda":
            raise RuntimeError("the fast path needs a CUDA device")
        self.dtype = next(self.vla.parameters()).dtype
        m = engine.model_config
        self.cams, self.size = tuple(engine.camera_keys), m.backbone.image_size
        self.chunk, self.adim, self.steps = engine.chunk_length, engine.action_dim, engine.diffusion_steps
        self.max_graphs = max_graphs
        dev = self.device
        self.images = torch.zeros(1, len(self.cams), 3, self.size, self.size, device=dev)
        self.state = torch.zeros(1, m.dit.state_dim, device=dev)
        self.noise = torch.zeros(1, self.chunk, self.adim, device=dev)
        self.prefix = torch.zeros(1, self.chunk, self.adim, device=dev)
        self.gelu_c = torch.sqrt(torch.tensor(2.0 / torch.pi, device=dev))
        self.prompts: dict[str, dict] = {}
        self.graphs: collections.OrderedDict = collections.OrderedDict()
        self.captures = 0

    # ----- per prompt
    def _prompt(self, prompt: str) -> dict:
        p = self.prompts.get(prompt)
        if p is not None:
            return p
        vla = self.vla
        placeholders = [[torch.empty(0)] * len(self.cams)]  # tokenize_raw_input only checks isinstance(Tensor)
        processed = tokenize_raw_input(vla.tokenizer, vla._raw_input(placeholders, [prompt]), vla.model_config,
                                       self.device)
        ids, length = processed["user_input_token_ids"], processed["prompt_lengths"][0]
        if length > vla.fixed_seq_len:  # the stock path's own check (GemmaVLABackbone.prepare)
            raise ValueError(f"multimodal sequence needs {length} tokens but fixed_seq_len={vla.fixed_seq_len}")
        n = vla.fixed_seq_len
        with torch.no_grad():  # prepare(): ids padded to fixed_seq_len, positions 0..n-1, pool mask = ~valid
            full = torch.full((1, n), vla.tokenizer.pad_id, dtype=ids.dtype, device=self.device)
            full[:, :length] = ids
            mask, local = vla._attention_masks(full, self.dtype)
            positions = torch.arange(n, device=self.device)
            p = {"ids": ids, "length": length, "n": n, "text": vla._embed_text(ids), "mask": mask, "local": local,
                 "freqs": vla._freqs(positions), "pad": ~(positions[None, :] < length)}
        if len(self.prompts) >= 4 * self.max_graphs:
            self.prompts.clear()
        self.prompts[prompt] = p
        return p

    # ----- the graph body: the stock computation from the normalized inputs on
    @torch.no_grad()
    def _body(self, p: dict, prefix_length: int | None) -> torch.Tensor:
        vla, model = self.vla, self.model
        normalized = vla._normalize_images(self.images)
        ctx = vla._inject_state(p["text"], p["ids"], self.state)
        ctx = vla._encode_images(ctx, normalized.to(self.dtype), p["ids"])
        embeds = torch.zeros(1, p["n"], ctx.shape[-1], dtype=ctx.dtype, device=ctx.device)
        embeds[:, : p["length"]] = ctx
        hidden = vla.gemma_model.model(hidden_states=embeds, freqs_cis=p["freqs"], mask=p["mask"],
                                       local_mask=p["local"], stop_layer_index=vla.feature_layer,
                                       apply_final_norm=False)
        cond = model.obs_pool(hidden.float(), key_padding_mask=p["pad"])
        kw = {} if prefix_length is None else {"action_prefix": self.prefix, "prefix_length": prefix_length}
        return model.diffusion_head.sample(cond, state=self.state.float(), num_steps=self.steps, noise=self.noise,
                                           **kw)

    @contextlib.contextmanager
    def _capturable(self):
        mlps = [block.mlp for block in self.vla.gemma_model.siglip_vision_model.encoder_blocks]
        for mlp in mlps:
            mlp.gelu_tanh = functools.partial(_gelu_tanh, c=self.gelu_c)
        try:
            yield
        finally:
            for mlp in mlps:
                del mlp.gelu_tanh

    def _graph(self, prompt: str, prefix_length: int | None):
        key = (prompt, prefix_length)
        if key in self.graphs:
            self.graphs.move_to_end(key)
            return self.graphs[key]
        t = time.time()
        p = self._prompt(prompt)
        with self._capturable():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.body(p, prefix_length)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.body(p, prefix_length)
        self.graphs[key] = (graph, out)
        self.captures += 1
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        self.last_capture_s = time.time() - t
        return self.graphs[key]

    # ----- one call
    def infer(self, obs: dict, noise=None, action_prefix=None, prefix_length: int | None = 0) -> np.ndarray:
        """VLAInferencePolicy.infer's contract for one observation."""
        e = self.e
        if action_prefix is not None and prefix_length is None:
            prefix_length = np.shape(action_prefix)[-2]
        prefix_length = int(prefix_length or 0)
        state = normalize(np.asarray(obs["state"], dtype=np.float32), e.norm_stats["state"])
        if state.ndim != 1:
            return e.infer(obs, noise=noise, action_prefix=action_prefix, prefix_length=prefix_length)
        prompt = str(obs.get("prompt", e.config.prompt))
        with torch.no_grad():
            for i, cam in enumerate(self.cams):
                image = torch.as_tensor(obs["images"][cam]).to(self.device)
                self.images[:, i].copy_(resize_pad_normalize_batch(image[None], self.size, self.size, preset=None))
            self.state.copy_(torch.from_numpy(state.reshape(1, -1)))
            if noise is None:
                self.noise.normal_()  # stock: sample() draws torch.randn when no noise is given
            else:
                self.noise.copy_(torch.from_numpy(np.asarray(noise, np.float32).reshape(1, self.chunk, self.adim)))
            if action_prefix is not None:
                prefix = e.normalized_action_prefix(action_prefix, prefix_length).reshape(1, self.chunk, self.adim)
                self.prefix.copy_(torch.from_numpy(prefix))
            graph, out = self._graph(prompt, prefix_length if action_prefix is not None else None)
            graph.replay()
            actions = out.float().cpu().numpy()
        return unnormalize(actions, e.norm_stats["actions"]).astype(np.float32)[0]


class FastDeployPolicy:
    """deploy.policy.base.DeployPolicy.infer's contract (zero noise by default, the bound warning) on FastEngine."""

    def __init__(self, deploy_policy, **kw):  # kw: FastEngine's (compile_body)
        self.stock = deploy_policy
        self.chunk_len, self.action_dim = deploy_policy.chunk_len, deploy_policy.action_dim
        self.engine = FastEngine(deploy_policy._policy, **kw)

    def infer(self, obs: dict, *, noise=None, action_prefix=None, prefix_length=None) -> dict:
        if noise is None:
            noise = np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        actions = self.engine.infer(obs, noise=noise, action_prefix=action_prefix, prefix_length=prefix_length)
        max_action = float(np.abs(actions).max())
        if max_action > MAX_ABS_ACTION:
            warnings.warn(f"Action bounds exceeded: max |action|={max_action:.3f}", RuntimeWarning, stacklevel=2)
        return {"actions": actions}
