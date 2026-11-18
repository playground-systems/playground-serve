"""Fast path of GR00T N1.7: the action head's 4 flow steps run as one CUDA graph per backbone sequence length.

Gr00tPolicy's call (gr00t/policy/gr00t_policy.py at the pinned commit) runs the Cosmos-Reason2 backbone once and then
`Gr00tN1d7ActionHead.get_action_with_features` (gr00t/model/gr00t_n1d7/gr00t_n1d7.py): 4 Euler steps of an alternating
cross/self-attention DiT over the state and action tokens. Measured on an RTX 4090: 39 ms for the backbone and 75 ms
for the action head, most of it Python launching small kernels one by one. FastActionHead is that method, line for
line, with its loop captured once per input shape:

- unchanged and eager: everything before the loop, including the noise draw `torch.randn(...)` from torch's stream at
  the same point (a seed gives the same noise as stock) and the masks of strict_action_padding_mask;
- copied into fixed buffers: the backbone features, the state features, the embodiment id, the image and backbone
  attention masks, the noise, the velocity strengths and the action masks;
- replayed: the loop (`action_encoder`, position embedding, `self.model`, `action_decoder`, the Euler update), the
  same kernels on the same shapes, so the actions are the stock path's bit for bit when the kernels pick the same
  algorithms (measured: NOTES.md).

Calls with an RTC prefix ("action" in the action input) or a batch above 1 run the stock method. The override is an
instance attribute of the action head and runs only inside `active()`; outside it the stock method runs, so the same
loaded model serves both paths (the boot self-test compares them in one process).
"""

from __future__ import annotations

import collections
import contextlib
import time

import torch

MAX_GRAPHS = 8


class FastActionHead:
    def __init__(self, head, max_graphs: int = MAX_GRAPHS):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import _expand_action_mask  # noqa: F401  (import check)
        from transformers.feature_extraction_utils import BatchFeature

        self.head, self.max_graphs, self.BatchFeature = head, max_graphs, BatchFeature
        self.stock = type(head).get_action_with_features.__get__(head)
        self.graphs: collections.OrderedDict = collections.OrderedDict()
        self.enabled, self.captures, self.last_capture_s = False, 0, 0.0
        head.get_action_with_features = self  # instance attribute: stock unless self.enabled

    @contextlib.contextmanager
    def active(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False

    def _loop(self, g: dict) -> torch.Tensor:
        """get_action_with_features' denoising loop, unchanged, on g's buffers."""
        head = self.head
        actions, action_mask, vel_strength = g["actions"], g["action_mask"], g["vel_strength"]
        action_token_mask, hidden_attention_mask = g["action_token_mask"], g["hidden_attention_mask"]
        state_features, vl_embeds, embodiment_id = g["state_features"], g["vl_embeds"], g["embodiment_id"]
        batch_size, device = vl_embeds.shape[0], vl_embeds.device
        dt = 1.0 / head.num_inference_timesteps
        for t in range(head.num_inference_timesteps):
            t_cont = t / float(head.num_inference_timesteps)
            t_discretized = int(t_cont * head.num_timestep_buckets)
            if action_mask is not None:
                actions = actions * action_mask
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            if action_token_mask is not None:
                action_features = action_features * action_token_mask.unsqueeze(-1).to(dtype=action_features.dtype)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if head.config.use_alternate_vl_dit:
                model_output = head.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                                          timestep=timesteps_tensor, image_mask=g["image_mask"],
                                          backbone_attention_mask=g["backbone_attention_mask"],
                                          hidden_attention_mask=hidden_attention_mask)
            else:
                model_output = head.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                                          timestep=timesteps_tensor,
                                          encoder_attention_mask=(g["backbone_attention_mask"]
                                                                  if head.config.strict_vl_padding_mask else None),
                                          hidden_attention_mask=hidden_attention_mask)
            pred = head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -head.action_horizon:]
            if action_mask is not None:
                pred_velocity = pred_velocity * action_mask
            actions = actions + dt * pred_velocity * vel_strength
            if action_mask is not None:
                actions = actions * action_mask
        return actions

    def _graph(self, inputs: dict) -> dict:
        key = tuple((k, None if v is None else (tuple(v.shape), str(v.dtype))) for k, v in sorted(inputs.items()))
        g = self.graphs.get(key)
        if g is not None:
            self.graphs.move_to_end(key)
            return g
        t0 = time.time()
        g = {k: None if v is None else v.clone() for k, v in inputs.items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._loop(g)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._loop(g)
        g["_graph"], g["_out"] = graph, out
        self.graphs[key] = g
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        self.captures += 1
        self.last_capture_s = time.time() - t0
        return g

    def __call__(self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None):
        if not self.enabled or "action" in action_input or backbone_features.shape[0] != 1:
            return self.stock(backbone_features, state_features, embodiment_id, backbone_output, action_input,
                              options)
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import _expand_action_mask

        head = self.head
        # get_action_with_features up to the loop, unchanged
        vl_embeds = backbone_features
        batch_size, device = vl_embeds.shape[0], vl_embeds.device
        actions = torch.randn(size=(batch_size, head.config.action_horizon, head.action_dim), dtype=vl_embeds.dtype,
                              device=device)
        vel_strength = torch.ones_like(actions)
        action_mask = action_token_mask = hidden_attention_mask = None
        if head.config.strict_action_padding_mask:
            action_mask = _expand_action_mask(action_input.get("action_mask"), actions)
            if action_mask is None:
                raise ValueError("action_mask is required when strict_action_padding_mask is enabled")
        if action_mask is not None:
            actions = actions * action_mask
            vel_strength = vel_strength * action_mask
            action_token_mask = action_mask.any(dim=-1)
            hidden_attention_mask = torch.cat(
                (torch.ones(state_features.shape[:2], device=device, dtype=torch.bool), action_token_mask), dim=1)
        inputs = {"actions": actions, "action_mask": action_mask, "vel_strength": vel_strength,
                  "action_token_mask": action_token_mask, "hidden_attention_mask": hidden_attention_mask,
                  "state_features": state_features, "vl_embeds": vl_embeds, "embodiment_id": embodiment_id,
                  "image_mask": getattr(backbone_output, "image_mask", None),
                  "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None)}
        g = self._graph(inputs)
        for k, v in inputs.items():
            if v is not None:
                g[k].copy_(v)
        g["_graph"].replay()
        return self.BatchFeature(data={"action_pred": g["_out"].clone(), "backbone_features": vl_embeds,
                                       "state_features": state_features})
