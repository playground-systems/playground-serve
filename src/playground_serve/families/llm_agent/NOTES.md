# LLM agent family

A language model as a robot policy, behind the same websocket wire as every other Playground Serve model. It needs no
GPU. `serve.py` is the server, `llm_session.py` one episode, `llm_card.py` the embodiment cards, `llm_pricing.py` the
prices and caps, `llm_fake.py` the recorded-response provider for tests and dry runs, and `cards.json` the built-in cards.

## The harness: Inspect Robots' agent policy

The agent loop is `inspect-robots-agent` 0.30.0 (MIT, Copyright (c) 2026 robocurve,
https://github.com/robocurve/inspect-robots, `plugins/inspect-robots-agent`; sdist sha256
`6631bd808bf5898393f56f545675a5dcd1a54f7878fed5dce1703be339145fea`) on `inspect-robots` 1.0.0 (MIT, the same
repository). It is a dependency, pinned in `requirements.txt`, not a copy: nothing of it is changed. What it supplies:

- Tools built from the embodiment's action space: `move_to` (absolute hand pose), `move_joints` (absolute joints) or
  `move_by` (displacements), `sleep`, `take_pic` (images on demand), `done` and `give_up` (each with a `hindsight`
  answer). A move is a straight line interpolated at a safe speed from the declared per-step limits, and every move
  carries a `note` for the person watching.
- Each observation as text: the labelled state, the setpoint the commands hold, the readback's offset from it, how much
  of the last motion played; plus the camera images (PNG), all of them or on request.
- A model layer with one interface (chat-format messages with images in, text and tool calls out, usage reported) and
  one client per wire: Anthropic Messages (with thinking-block replay and prompt caching), OpenAI Chat Completions,
  OpenAI Responses, Google's Interactions API and Google's Live API. Providers by `provider/model` prefix: anthropic,
  openai, google, x-ai, groq, mistralai, deepseek, thinkingmachines; any OpenAI-compatible server through a base URL
  (vLLM, Ollama, Together, Fireworks, OpenRouter); OpenRouter as the fallback for any model id.
- A call budget per trial, retries with backoff, refusal and truncation handling.

What this family adds: the openpi wire, sessions per episode, the embodiment card and the conversion of the agent's
targets into the environment's native actions (`Driver`), the USD caps, and usage and transcript in each reply.

### Why this harness, and the others compared

The decision weighed four ways for a model to act:

| Way | Latency against control | Fit here |
|---|---|---|
| Low-level action chunks from the model | one model call per few control steps (LIBERO-Spatial: 44 calls of 5 steps per episode) | Rejected: the most calls, and models are poor at numeric deltas (EmbodiedBench's low-level manipulation track) |
| Waypoints or hand targets, a controller underneath | one call per motion; the controller plays and settles it | The engine's `adapters/llm_server.py` and the `fable_demos` experiment do this; kept as the core |
| Tool loop (move, grasp, look, wait) whose calls become action chunks | as the row above | The chosen default: the waypoint harness with typed tools and a provider layer |
| Code as policy (the model writes a control program) | one call per program | Not built: it needs a sandbox for model-written code on a shared service, and perception servers (CaP-X needs SAM 3 and grasp models on a GPU) |

Open-source harnesses read (2026-11-16):

| Project | License | Actions | Providers | Taken |
|---|---|---|---|---|
| inspect-robots-agent | MIT | tools over the embodiment's spaces | its own clients for five wires, prefixes, base URL, OpenRouter | the whole loop, as a dependency |
| inspect-robots-capx (CaP-X, capgym/cap-x) | MIT (CaP-X); the plugin's README states none (the repo is MIT) | code over perception and control primitives | OpenAI-compatible proxy | nothing yet: a later mode once perception servers exist |
| Code as Policies (google-research) | Apache-2.0 (monorepo) | hierarchical generated code | OpenAI completions | the idea only; frozen since 2024 |
| VoxPoser | MIT | code that builds value maps, then waypoints and a controller | OpenAI | the waypoint-plus-controller split |
| ROSA (nasa-jpl) | Apache-2.0 | LangChain tools over ROS | LangChain models | nothing: operations, not manipulation |
| RAI (RobotecAI) | Apache-2.0 | ROS 2 tools with perception | several | nothing yet |
| ros-mcp-server, robot_MCP | Apache-2.0 | MCP tools | any MCP client | a possible later front end; image results are their weak point |
| Gemini robotics-pointing-sample | Apache-2.0 | 2D points mapped to the table by a homography | Gemini | nothing yet |
| EmbodiedBench | none declared | low-level actions | OpenAI, Gemini, Anthropic and others | its result as a reason against low-level actions; no license, so no code |
| UK AISI's inspect_ai | MIT | (static evals) | many providers, images and tools | not used: it has no closed-loop robot control, and inspect-robots-agent already covers the providers |

The user's earlier experiments (`fable_demos` on LIBERO, a bimanual YAM Jenga scene) used the same pattern: absolute
hand targets in one tool call, a proportional controller underneath, the full history with images of the last two
observations, no caching. Their lessons are in the LIBERO card's docs: the image directions, the reach limit, rim
grasps of bowls, and that a move's report must say where the hand stopped.

## Wire

A server whose features say `"agent": true` keeps state per episode. Add to each request:

- `episode`: a string that is new for every episode. The server keys the episode by it, so a reconnect continues it.
  Without it, an episode is the socket and its prompt.
- `embodiment`: the embodiment card (a map, its JSON text, or a built-in card's name such as `libero_franka`); with every
  request, or at least the first of each episode. The card names the request's camera, state and prompt keys and the
  action layout the reply comes in. Without it the server looks for a built-in card whose keys the request has.
- `max_steps` (optional): the episode's step limit, which the model is told.

The reply's `actions` has the card's `chunk` rows. Its `agent` map holds the episode's calls, tokens, USD, model seconds,
`finished` once the agent stopped, and `turns`, the transcript's new messages without images. A request can take as
long as a model call (seconds to minutes): wait for the reply as for any other; the server answers pings meanwhile. A
request whose images are all black is a warm-up and starts no episode. A provider failure after the retries (a rejected
key, an outage) is the server's error frame.

## Memory, images and caching

Each episode is one conversation, append-only: `image_horizon` is off by default, so no earlier message is edited.
That keeps every thinking block valid on the models that check the history (Claude Opus 5.5, Sonnet 5.5, Haiku 5.5
and Fable 5.2 reject replayed thinking after an edit on accounts made after 2026-08-31) and keeps the prompt cache
warm (the Messages client marks the system prompt and the newest image for caching). At LIBERO's 224 x 224 images the
whole episode stays far below any context limit: the real episode below used 37K prompt tokens over six calls.
`PGS_AGENT_IMAGE_HORIZON=2` stubs older frames for a long episode on a wire that has no such check.

## Embodiment cards and the native conversion

A card (`llm_card.py`, built-in `cards.json`) gives the agent's interface as Inspect Robots spaces and the
environment's native wire. `libero_franka` maps LIBERO's delta actions to an absolute pose interface (x, y, z, rot6d,
gripper 0 to 1, the interface of the engine's own Inspect Robots embodiment): the server servoes each interpolated
target with a proportional controller (gain 1 / 0.05 m, 1 / 0.5 rad), then holds the last target until the hand is
within 4 mm and 2 degrees or stalls for 15 steps (at most 60 steps), and gives a gripper change 15 steps, before it asks
the model again. A `same` card passes absolute joint or pose targets through unchanged.

## Keys and logs

The settings come from the endpoint's environment only; a request cannot change the model, the key or the base URL.
Each episode's policy gets a mapping with the provider keys, so a key goes only in that provider's request header (or
to the endpoint's own base URL, which must be https). The log has counts: no prompt, image, note or key; httpx's
request lines are off. Each reply's `agent.turns` gives the client the episode's new messages without images.

## Measured (this Mac, 2026-11-16)

- Tests: `tests/families/test_llm_agent.py`, recorded tool calls on the Messages, Chat Completions and Responses wires
  and an OpenAI-compatible base URL, the driver against a stand-in LIBERO, caps, a rejected key, the server over the
  websocket.
- Dry run with recorded tool calls: one LIBERO-Spatial episode through the engine's native runtime (vla-eval, the
  engine's openpi proxy, this server): 44 requests, 8 model calls, each target reached within 3 mm.
- One real episode, `anthropic/claude-haiku-5-5`, effort low, LIBERO-Spatial task 1: 6 model calls, 2.8 s of model
  time per call, 24 uncached input, 28,745 cache-read, 8,682 cache-write and 2,888 output tokens, 0.0028 USD. The
  episode failed: the step limit (220) ran out while the model was still locating the bowl from the wrist camera.

## Provider status

| Provider (wire) | Status |
|---|---|
| Anthropic, Messages API | one real episode (Claude Haiku 5.5); recorded responses in tests |
| OpenAI, Chat Completions | recorded responses in tests only; never met the real API here |
| OpenAI, Responses API | recorded responses in tests only; never met the real API here |
| OpenAI-compatible base URL (vLLM, Ollama, Together, Fireworks, OpenRouter) | recorded responses in tests only |
| Google: OpenAI-compatible endpoint, Interactions, Live | not tested here (upstream tests only) |
| x-ai, groq, mistralai, deepseek | not tested here (upstream tests only) |

## Not built

- `pgs up` does not create this family's endpoint: it is a CPU endpoint without a volume, which the launcher does not
  make yet. Playground Bench's compute plane creates it; `start.sh` is its boot.
- Code-as-policy mode, and a card for any robot other than LIBERO's Panda.
- The step limit is passed only when the client sends `max_steps`; the engine's proxy does not yet.
