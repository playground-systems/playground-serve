"""Registry entries of the llm_agent family: the nine robot-use language models (llm_models.MODELS), each a model of
its own that `pgs up <name>` serves on a CPU worker, reached through OpenRouter with the OPENROUTER_API_KEY of the
machine that runs `pgs up` (a secret of the entry). No checkpoint, volume or GPU: the model runs at its provider, and
the worker holds the agent loop (serve.py). action_shape is (0, 0): the action layout comes with each episode's
embodiment card. gpu_validated holds the date the model answered a real tool call with images through OpenRouter."""

from __future__ import annotations

from playground_serve.families.common.proposed_spec import ProposedSpec
from playground_serve.families.llm_agent.llm_models import KEY_ENV, MODELS

FAMILY = "llm_agent"
IMAGE = "python:3.12-slim"
CPU = "cpu3c-2"  # Runpod's compute-optimized CPU worker, 2 vCPU
LICENSE = "the model provider's terms, through OpenRouter's"

SPECS = {name: ProposedSpec(name=name, family=FAMILY, checkpoint="", volume_gb=0, min_container_ram_gb=4, gpu_pools=(),
                            action_shape=(0, 0), robot="any (embodiment card)", image=IMAGE, cpu=CPU,
                            env={"PGS_AGENT_MODEL": name}, secrets=(KEY_ENV,), gpu_validated=m.verified,
                            license=LICENSE, hosted_id=m.slug,
                            agent={"provider": "openrouter", "slug": m.slug, "title": m.title, "effort": m.effort,
                                   "max_tokens": m.max_tokens, "prices": list(m.prices), "note": m.note})
         for name, m in MODELS.items()}
