"""Fast path of Xiaomi-Robotics-1: the whole forward as one CUDA graph per prompt, and a cached request builder.

Stock (serve.Policy with the official calls) on an RTX 4090, per call: 16.6 ms building the request on the CPU (8 ms of
it the evaluator's centre crop of 12 frames with PIL, 4 ms the video processor, the rest the chat template and the
tokenizer), 67.6 ms for the Qwen3-VL prefill (vision tower 17 ms; 199 `.item()` calls and 285 stream synchronizations
per call, plus the logits of all 476 tokens over the 151,936-token vocabulary, which the action head never reads), and
about 154 ms for the 5 flow steps of the 36-layer DiT, which launch about 9,000 small kernels from Python.

FastRequest builds the same request in about 1.2 ms (RTX 4090):
- the token ids, attention mask and video grid once per instruction, from one full processor call (they depend only
  on the instruction and the frame size, which the request check fixes at 256 x 256);
- the 12 frames to the GPU in one copy, entry.py's centre crop there with Pillow's own fixed-point bilinear arithmetic
  (GpuCenterCrop), then the video processor's rescale_and_normalize and patch layout: pixel_values_videos equal to the
  CPU processor's, bit for bit, which the first call of each instruction checks (else PIL crops in threads and the
  video processor on the GPU, also bit for bit);
- state and action_mask as the processor makes them (bf16).

FastForward computes MiBoTForActionGeneration.forward (modeling_mibot.py at the pinned revision) from those inputs:
- per prompt, once, eagerly, with the model's own functions: the text embeddings of the token ids, the multimodal
  rope position ids (`get_rope_index`), the text rotary tables, the causal mask (`create_causal_mask`), the flash
  attention lengths the stock path derives from the position ids (`_prepare_from_posids`), the placeholder indices of
  the video tokens, the vision tower's position embeddings, rotary tables and sequence lengths for the fixed video
  grid, and the DiT's rotary tables and attention mask;
- per call, in one captured CUDA graph: the vision tower (blocks and deepstack mergers), the video embeddings written
  at their placeholder positions, the 36 text decoder layers with the deepstack features added after layers 0 to 2,
  filling the key/value cache, the state projection and the 5 Euler steps of `dit_forward`;
- per call, eagerly, at the same point of torch's stream as stock: the noise draw `torch.randn_like(action_mask)`.

What changes against stock is only what stock computes and never uses (the final norm and the vocabulary logits) and
where values that are fixed per prompt come from (computed once instead of every call). The host synchronizations
are gone because the values they read are fixed: the vision attention gets its maximum sequence length as a Python
int (stock reads it from the GPU in every block), the text attention gets the lengths stock derives from the position
ids. The kernels on the GPU are the stock path's, so the actions are stock's (measured: NOTES.md).
"""

from __future__ import annotations

import collections
import concurrent.futures
import sys
import time

import numpy as np
import torch

MAX_GRAPHS = 8


PRECISION_BITS = 32 - 8 - 2  # Pillow's Resample.c for 8-bit images


def pil_bilinear_taps(in_size: int, out_size: int, in0: float = 0.0, in1: float | None = None):
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bilinear filter (libImaging/Resample.c): per output
    pixel the input index and the fixed-point weight of each tap, as (index, weight), both (out_size, ksize)."""
    import math

    in1 = float(in_size) if in1 is None else in1
    scale = (in1 - in0) / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    index = np.zeros((out_size, ksize), np.int64)
    weight = np.zeros((out_size, ksize), np.int64)
    for xx in range(out_size):
        center = in0 + (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        k = [max(0.0, 1.0 - abs((x + xmin - center + 0.5) * ss)) for x in range(xmax)]
        ww = sum(k)
        k = [w / ww if ww != 0.0 else w for w in k]
        for x in range(ksize):
            index[xx, x] = min(xmin + x, in_size - 1)
            if x < xmax:
                w = k[x] * (1 << PRECISION_BITS)
                weight[xx, x] = int(-0.5 + w) if k[x] < 0 else int(0.5 + w)
    return index, weight


def _clip8(acc: torch.Tensor) -> torch.Tensor:
    """Resample.c's clip8: the rounded sum shifted down to 8 bits, clamped to 0..255."""
    return torch.where(acc <= 0, torch.zeros_like(acc), torch.clamp(acc >> PRECISION_BITS, max=255))


class GpuCenterCrop:
    """entry.py's center_crop (PIL crop of the centre, then a bilinear resize back to the original size) for a batch of
    frames on the GPU, with Pillow's fixed-point arithmetic: a horizontal pass to 8 bits, then a vertical pass, each
    output pixel the rounded sum of its taps' integer weights (bit-identical to PIL: checked at every build)."""

    def __init__(self, size: int, crop_ratio: float, device):
        self.size = size
        crop = max(1, int(size * crop_ratio))
        self.left = (size - crop) // 2
        self.crop = crop
        idx, w = pil_bilinear_taps(crop, size)
        self.idx = torch.as_tensor(idx, device=device)
        self.w = torch.as_tensor(w, device=device)

    def _pass(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        acc = None
        for j in range(self.idx.shape[1]):
            taps = torch.index_select(x, dim, self.idx[:, j]).to(torch.int64)
            shape = [1] * x.dim()
            shape[dim] = -1
            term = taps * self.w[:, j].view(shape)
            acc = term if acc is None else acc + term
        return _clip8(acc + (1 << (PRECISION_BITS - 1))).to(torch.uint8)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (N, H, W, 3) uint8 on the GPU -> (N, H, W, 3) uint8."""
        x = frames[:, self.left:self.left + self.crop, self.left:self.left + self.crop, :]
        x = self._pass(x, 2)  # horizontal: along W
        return self._pass(x, 1)  # vertical: along H


class FastRequest:
    """serve.Policy.request's result, with the per-instruction parts cached and the pixels made on the GPU.

    Pixels: the 12 frames go to the GPU in one copy; GpuCenterCrop does entry.py's centre crop with Pillow's integer
    arithmetic, and the video processor's own rescale_and_normalize and its patch layout (Qwen3VLVideoProcessor
    ._preprocess) make pixel_values_videos. The first call of each instruction also runs the full CPU processor and
    compares: on any difference the GPU pixels are turned off for good and the PIL crops (in threads) with the video
    processor on the GPU are used instead (`self.gpu_pixels`, logged)."""

    def __init__(self, policy, serve, threads: int = 12):
        """`policy`: serve.Policy (stock); `serve`: the serve module (center_crop, frame_stack, STATE_DIM)."""
        self.policy, self.serve = policy, serve
        self.processor = policy.processor
        self.pool = concurrent.futures.ThreadPoolExecutor(threads, thread_name_prefix="xr1-crop")
        self.text: dict = {}
        self.device = policy.model.device
        v = policy.variant
        self.crop = GpuCenterCrop(v["image_size"][0], v["crop_ratio"], self.device)
        self.gpu_pixels = v["image_size"][0] == v["image_size"][1]
        self.gpu_pixels_reason = ""

    def _pixels_gpu(self, frames: np.ndarray) -> torch.Tensor:
        """frames (cameras, T, H, W, 3) uint8 -> pixel_values_videos, as the video processor makes them."""
        vp = self.processor.video_processor
        n_cam, t_len = frames.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device, non_blocking=True)
        x = self.crop(x.reshape(n_cam * t_len, *frames.shape[2:]))
        x = x.reshape(n_cam, t_len, *x.shape[1:]).permute(0, 1, 4, 2, 3)  # B T C H W, as _prepare_input_videos
        x = vp.rescale_and_normalize(x, vp.do_rescale, vp.rescale_factor, vp.do_normalize, tuple(vp.image_mean),
                                     tuple(vp.image_std))
        tp, ps, ms = vp.temporal_patch_size, vp.patch_size, vp.merge_size
        if x.shape[1] % tp:
            x = torch.cat([x, x[:, -1:].repeat(1, tp - x.shape[1] % tp, 1, 1, 1)], dim=1)
        b, t, c, h, w = x.shape
        gt, gh, gw = t // tp, h // ps, w // ps
        x = x.view(b, gt, tp, c, gh // ms, ms, ps, gw // ms, ms, ps).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return x.reshape(b * gt * gh * gw, c * tp * ps * ps)

    def _pixels_cpu(self, frames: np.ndarray):
        v = self.policy.variant
        jobs = [self.pool.submit(self.serve.center_crop, frame, v["crop_ratio"]) for cam in frames for frame in cam]
        crops = [j.result() for j in jobs]
        videos = [crops[i * frames.shape[1]:(i + 1) * frames.shape[1]] for i in range(frames.shape[0])]
        out = self.processor.video_processor(videos=videos, do_resize=False, return_tensors="pt", device="cuda")
        return out["pixel_values_videos"], out["video_grid_thw"]

    def __call__(self, obs: dict) -> dict:
        p, v = self.policy, self.policy.variant
        shape = (v["history"], *v["image_size"], 3)
        frames = np.stack([self.serve.frame_stack(obs[key], shape) for key in v["cameras"]])
        states = np.asarray(obs["state"], dtype=np.float32)
        if states.shape != (v["history"], v["state_dim"]):
            raise ValueError(f"state must be shape ({v['history']}, {v['state_dim']}), got {states.shape}")
        state = np.zeros((1, states.shape[0], self.serve.STATE_DIM), dtype=np.float32)
        state[0, :, : states.shape[-1]] = states
        instruction = str(obs["instruction"])
        cached = self.text.get(instruction)
        if cached is None:  # one full processor call per instruction: its text parts are reused
            request = p.request(obs)
            cached = {k: request[k] for k in ("input_ids", "attention_mask", "video_grid_thw")}
            cached["action_mask"] = request["action_mask"]
            if self.gpu_pixels:  # the GPU pixels must equal the processor's, bit for bit, or they are not used
                try:
                    same = torch.equal(self._pixels_gpu(frames).cpu(), request["pixel_values_videos"])
                    self.gpu_pixels_reason = "" if same else "GPU pixels differ from the processor's"
                except Exception as e:  # noqa: BLE001
                    same, self.gpu_pixels_reason = False, f"GPU pixels failed: {e!r}"[:200]
                self.gpu_pixels = same
            if len(self.text) > 64:
                self.text.clear()
            self.text[instruction] = cached
        if self.gpu_pixels:
            pixels = self._pixels_gpu(frames)
        else:
            pixels, grid = self._pixels_cpu(frames)
            if not torch.equal(grid.cpu(), cached["video_grid_thw"]):
                raise ValueError("video grid differs from the cached one")
        return {"input_ids": cached["input_ids"], "attention_mask": cached["attention_mask"],
                "pixel_values_videos": pixels, "video_grid_thw": cached["video_grid_thw"],
                "state": torch.as_tensor(state).to(torch.bfloat16), "action_mask": cached["action_mask"],
                "task_id": v["robot_type"]}


class FastForward:
    """forward(data) -> raw actions (1, 16, 60), as `model(**data).actions` (deploy/server.py's call)."""

    def __init__(self, model, max_graphs: int = MAX_GRAPHS, num_steps: int = 5, compile_dit: bool = False):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_flash_attention_utils import _is_packed_sequence, _prepare_from_posids
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.model, self.max_graphs, self.num_steps = model, max_graphs, num_steps
        self.mm = model.vlm.model  # Qwen3VLModel
        self.visual, self.lm = self.mm.visual, self.mm.language_model
        self.modeling = sys.modules[type(model).__module__]  # the checkpoint's modeling_mibot
        self.DynamicCache, self.create_causal_mask = DynamicCache, create_causal_mask
        self.is_packed, self.from_posids = _is_packed_sequence, _prepare_from_posids
        self.attn_functions = ALL_ATTENTION_FUNCTIONS
        self.graphs: collections.OrderedDict = collections.OrderedDict()
        self.captures, self.last_capture_s = 0, 0.0
        self.dit_step = self._dit_step
        if compile_dit:  # inductor fuses the DiT step's small kernels; not bit-identical (NOTES.md)
            import torch._inductor.config as inductor_config

            inductor_config.emulate_precision_casts = True  # round as eager does between ops: half the difference
            self.dit_step = torch.compile(self._dit_step, dynamic=False)

    def _dit_step(self, x, t, action_mask, state_embed, cos, sin, kv, attn_mask):
        return self.model.dit_forward(noisy_action=x, t=t, action_mask=action_mask, state_embed=state_embed,
                                      position_embeds=(cos, sin), past_key_values=kv, attn_mask=attn_mask)

    # ----- once per prompt (eager, the stock functions)
    def _static(self, data: dict) -> dict:
        mm, visual, lm, model = self.mm, self.visual, self.lm, self.model
        dev = model.device
        input_ids = data["input_ids"].to(dev)
        attention_mask = data["attention_mask"].to(dev)
        grid = data["video_grid_thw"].to(dev)
        s: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        # vision tower: Qwen3VLVisionModel.forward's grid-dependent values
        s["pos_embeds"] = visual.fast_pos_embed_interpolate(grid)
        seq_len = int(grid.prod(-1).sum())
        rotary = visual.rot_pos_emb(grid).reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        s["vis_pe"] = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
        s["vis_cu"] = torch.nn.functional.pad(cu, (1, 0), value=0)
        s["vis_max"] = int((s["vis_cu"][1:] - s["vis_cu"][:-1]).max())
        # text: embeddings, placeholders, rope positions, rotary tables, causal mask, flash lengths
        with torch.no_grad():
            s["text_embeds"] = mm.get_input_embeddings()(input_ids)
            video_mask = input_ids == mm.config.video_token_id
            s["vid_idx"] = video_mask[0].nonzero().squeeze(1)
            position_ids, _ = mm.get_rope_index(input_ids, None, grid, attention_mask=attention_mask)
            s["position_ids"] = position_ids
            text_position_ids = position_ids[0]
            cache_position = torch.arange(0, input_ids.shape[1], device=dev)
            s["cache_position"] = cache_position
            s["text_pe"] = lm.rotary_emb(s["text_embeds"], position_ids)
            s["causal_mask"] = self.create_causal_mask(
                config=lm.config, input_embeds=s["text_embeds"], attention_mask=attention_mask,
                cache_position=cache_position, past_key_values=self.DynamicCache(config=lm.config),
                position_ids=text_position_ids)
            s["fa"] = {}
            if s["causal_mask"] is None and bool(self.is_packed(text_position_ids, batch_size=1)):
                # what _flash_attention_forward derives from the position ids in every layer of the stock path
                n_heads, head_dim = lm.config.num_attention_heads, lm.config.head_dim
                q = torch.zeros(1, input_ids.shape[1], n_heads, head_dim, device=dev, dtype=s["text_embeds"].dtype)
                _, _, _, (cu_q, cu_k), (max_q, max_k) = self.from_posids(q, q, q, text_position_ids)
                s["fa"] = {"cu_seq_lens_q": cu_q, "cu_seq_lens_k": cu_k, "max_length_q": int(max_q),
                           "max_length_k": int(max_k)}
            # MiBoT forward: the DiT's rotary tables and attention mask
            action_mask = data["action_mask"].to(dev, model.dtype)
            state = data["state"]
            bs, action_length, _ = action_mask.shape
            q_len = action_length + state.shape[1] + 1
            dit_pos = (torch.arange(0, q_len, device=dev).view(1, 1, -1).repeat(3, bs, 1)
                       + position_ids.max(dim=-1)[0][..., None] + 1)
            s["dit_pe"] = model.rotary_emb(action_mask, dit_pos)
            dit_mask = torch.tril(torch.ones((bs, q_len, q_len), device=dev), diagonal=0)
            cache_mask = attention_mask[:, None, :].expand(-1, q_len, -1)
            s["dit_mask"] = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        return s

    # ----- the captured body
    def _vision_attention(self, attn, cu: torch.Tensor, max_seqlen: int):
        """Qwen3VLVisionAttention.forward's flash-attention branch with its maximum length given as an int."""
        apply_rope = self.modeling.apply_rotary_pos_emb_vision
        interface = self.attn_functions[attn.config._attn_implementation]

        def forward(hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
            cos, sin = position_embeddings
            query_states, key_states = apply_rope(query_states, key_states, cos, sin)
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            attn_output, _ = interface(attn, query_states, key_states, value_states, attention_mask=None,
                                       scaling=attn.scaling, dropout=0.0, cu_seq_lens_q=cu_seqlens,
                                       cu_seq_lens_k=cu_seqlens, max_length_q=max_seqlen, max_length_k=max_seqlen,
                                       is_causal=False, **kwargs)
            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            return attn.proj(attn_output)

        return forward

    def _body(self, s: dict, g: dict):
        visual, lm, model = self.visual, self.lm, self.model
        # Qwen3VLVisionModel.forward
        h = visual.patch_embed(g["pixels"].type(visual.dtype))
        h = h + s["pos_embeds"]
        h = h.reshape(h.shape[0], -1)
        deepstack = []
        for layer_num, blk in enumerate(visual.blocks):
            h = blk(h, cu_seqlens=s["vis_cu"], position_embeddings=s["vis_pe"])
            if layer_num in visual.deepstack_visual_indexes:
                deepstack.append(visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)](h))
        video_embeds = visual.merger(h)
        # Qwen3VLModel.forward: video embeddings at their placeholders (masked_scatter), then the text model's layers
        embeds = s["text_embeds"].clone()
        embeds[0, s["vid_idx"]] = video_embeds.to(embeds.dtype)
        cache = self.DynamicCache(config=lm.config)
        h = embeds
        for layer_idx, layer in enumerate(lm.layers):
            h = layer(h, attention_mask=s["causal_mask"], position_ids=None, past_key_values=cache,
                      cache_position=s["cache_position"], position_embeddings=s["text_pe"], **s["fa"])
            if layer_idx < len(deepstack):  # _deepstack_process with the placeholder indices
                local = h[0, s["vid_idx"], :].clone() + deepstack[layer_idx].to(h.device, h.dtype)
                h[0, s["vid_idx"], :] = local
        # MiBoTForActionGeneration.forward after the prefill
        state_embed = model.state_projector(g["state"])
        kv = [(cache[i][0], cache[i][1]) for i in range(len(cache))]  # what DiT.forward reads, layer by layer
        x = g["x"]
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / self.num_steps
            v = self.dit_step(x, t, g["action_mask"], state_embed, *s["dit_pe"], kv, s["dit_mask"])
            x = x + v * dt
        return x

    def _graph(self, key, data: dict, inputs: dict) -> dict:
        g = self.graphs.get(key)
        if g is not None:
            self.graphs.move_to_end(key)
            return g
        t0 = time.time()
        s = self._static(data)
        g = {k: v.clone() for k, v in inputs.items()}
        attns = [blk.attn for blk in self.visual.blocks]
        for attn in attns:
            attn.forward = self._vision_attention(attn, s["vis_cu"], s["vis_max"])
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._body(s, g)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._body(s, g)
        finally:
            for attn in attns:
                del attn.forward
        g.update(_graph=graph, _out=out, _static=s)
        self.graphs[key] = g
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        self.captures += 1
        self.last_capture_s = time.time() - t0
        return g

    @torch.no_grad()
    def __call__(self, data: dict) -> torch.Tensor:
        model = self.model
        dev, dtype = model.device, model.dtype
        pixels = data["pixel_values_videos"].to(dev, dtype)  # deploy/server.py: floating tensors to bf16
        state = data["state"].to(dev, dtype)
        action_mask = data["action_mask"].to(dev, dtype)
        key = (tuple(data["input_ids"].shape), data["input_ids"].cpu().numpy().tobytes(), tuple(pixels.shape))
        inputs = {"pixels": pixels, "state": state, "action_mask": action_mask, "x": torch.empty_like(action_mask)}
        g = self._graph(key, data, inputs)
        # the noise: stock draws it after the prefill; nothing between uses torch's stream, so drawing it here gives
        # the same numbers
        x = torch.randn_like(action_mask)
        for k, v in (("pixels", pixels), ("state", state), ("action_mask", action_mask), ("x", x)):
            g[k].copy_(v)
        g["_graph"].replay()
        return g["_out"].clone()
