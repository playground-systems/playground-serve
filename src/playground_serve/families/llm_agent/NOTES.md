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

Playground Bench's custom environments use their own shape (its docs/cloud/CUSTOM.md, section 10): the observation
nested as `images` and `state` maps, and with an episode's first request `embodiment_card` (their card as JSON) and
`reset: true`. The server takes it too (`llm_card.from_env_card`: an absolute pose or joint interface, kind `same`) and
says `"embodiment_card": true` in its metadata. Each `reset` starts a new episode on the connection.

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
target (at most 1.2 cm per step) with a proportional controller (gain 2.5 / 0.05 m and 2.5 / 0.5 rad), then holds the
last target until the hand is within 5 mm and 2 degrees or stalls for 15 steps (at most 40 steps), and gives a gripper
change 12 steps, before it asks the model again. Measured with recorded moves in LIBERO: seven moves of a pick and place
took 155 control steps (180 with gain 1, 4 mm and 60 steps), each ending within 5 mm. A `same` card passes absolute
joint or pose targets through unchanged.

The card's docs are about the robot, the workspace, the cameras and time; they hold no hint about any task or object.
Each observation states the step budget (the client's `max_steps`, or a custom environment's `limits.maxSteps`): the
step, the steps left, and the docs give the rate (20 steps a second on LIBERO). Every episode also gets the default
notes (`llm_session.DEFAULT_NOTES`): the scene is simulated, plan against the step budget, look at every camera first,
read the gripper's state after closing. The default effort is `medium`.

Custom environments' cards (Playground Bench's `cloud/envs/card.ts`; copies of its Panda and two-arm YAM cards in
`tests/families/data/`) arrive as `embodiment_card` and become `same` cards (`from_env_card`). A plain MuJoCo scene's own
action layout (`custom`) is driven only when its card says `action.kind: "absolute"`; forces, velocities and rates are
refused with the reason. Checked against that repository's own episode runner (`env_run.run` with its stand-in session)
over the websocket: one episode per reset, absolute rows played as sent, the step budget and the success checks in the
prompt.

## Keys and logs

The settings come from the endpoint's environment only; a request cannot change the model, the key or the base URL.
Each episode's policy gets a mapping with the provider keys, so a key goes only in that provider's request header (or
to the endpoint's own base URL, which must be https). The log has counts: no prompt, image, note or key; httpx's
request lines are off. Each reply's `agent.turns` gives the client the episode's new messages without images.

Tokens are counted by `Meter`, an httpx transport around the agent's: it reads the usage object of each answered
request on every HTTP wire (inspect-robots-agent 0.30.0 keeps no usage on the Responses wire, so its own totals would
miss OpenAI's tokens), and the USD caps use those counts.

## Measured (this Mac, 2026-11-16)

LIBERO-Spatial task 1 ("pick up the black bowl between the plate and the ramekin and place it on the plate"), 220 steps,
three episodes per model (vla-eval's first three initial states), effort `medium`, the generic card and notes above,
through the engine's native runtime and its openpi proxy. Success is LIBERO's own.

| Model | Success | Steps used | Calls | Seconds per call | Tokens per episode (uncached in / cache read / cache write / out) | USD per episode |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 0 of 3 | 220, 220, 220 | 9, 8, 10 | 3.2, 4.4, 3.0 | about 40 / 65K / 11K / 5.5K | 0.0047, 0.0051, 0.0045 |
| claude-sonnet-5-5 | 0 of 3 | 220, 220, 220 | 10, 9, 9 | 5.1, 4.3, 4.9 | about 40 / 65K / 10K / 4.6K | 0.089, 0.066, 0.076 |
| claude-opus-5-5 | 2 of 3 | 145, 220, 150 | 6, 8, 7 | 6.3, 5.4, 5.2 | about 30 / 40K / 7.5K / 3.2K | 0.113, 0.118, 0.098 |

Three episodes per model is a small sample: these counts say what the harness can do, not how the models rank. What the
transcripts show: Haiku and Sonnet found the bowl and descended to it, then closed the jaws two to four times where the
fingers met on nothing (the grasp point over the bowl's inside or beside it), and gave up or ran out of steps. Opus
placed the grasp point on the bowl's rim and lifted it at the first close in every episode; its failure was a bowl
that slipped out while carried. Depth and reach were not the problem (each target was reached within 5 mm). The change
most likely to help the others: a sharper view at the moment of grasping (the wrist image is 224 pixels; a crop tool or
a higher-resolution wrist frame before a close), then the step budget, which a failed grasp uses up.

Earlier (before the step budget and the faster settle): one Haiku episode at effort low, 6 calls, 0.0028 USD, failed.

Tests: `tests/families/test_llm_agent.py`, recorded tool calls on the Messages, Chat Completions and Responses wires and
an OpenAI-compatible base URL, the driver against a stand-in LIBERO, the custom environments' cards, caps, a rejected
key, the server over the websocket.

## OpenAI models (read 2026-11-16)

From developers.openai.com/api/docs/pricing and the model pages: the API's Codex model is `gpt-5.3-codex` (deprecated
there; Responses API only; text and image input; function calling; reasoning effort low to xhigh; 400K context, 128K
output; 1.75 / 0.175 cached / 14 USD per million tokens). Codex itself now recommends the general models `gpt-6-astra`
(10 / 1.00 / 50), `gpt-6.1-sol` (2 / 0.10 / 10), `gpt-6.1-terra` (1 / 0.05 / 5) and `gpt-6-luna` (0.10 / 0.01 / 0.50,
cache writes 0.125): text and image input, function calling on the Responses API (on Chat Completions only with
reasoning effort `none`), effort none or low to max, 1.05M context. So OpenAI models run on the Responses wire by
default here. A real run was not possible: the service's OpenAI key answered 429 `credit_balance_exhausted` on the
first call of a `gpt-6-luna` episode (the server failed the request after the agent's retries, as designed).

## Provider status

| Provider (wire) | Status |
|---|---|
| Anthropic, Messages API | 10 real LIBERO episodes (Claude Haiku, Sonnet and Opus 5.5); recorded responses in tests |
| OpenAI, Responses API (default for openai/*) | recorded responses in tests; the real API refused the key (no credit) |
| OpenAI, Chat Completions | recorded responses in tests only; never met the real API here |
| OpenAI-compatible base URL (vLLM, Ollama, Together, Fireworks, OpenRouter) | recorded responses in tests only |
| Google: OpenAI-compatible endpoint, Interactions, Live | not tested here (upstream tests only) |
| x-ai, groq, mistralai, deepseek | not tested here (upstream tests only) |

## Not built

- `pgs up` does not create this family's endpoint: it is a CPU endpoint without a volume, which the launcher does not
  make yet. Playground Bench's compute plane creates it; `start.sh` is its boot.
- Code-as-policy mode; a crop or zoom tool for the grasp; a card for LIBERO's floor and living-room scenes.
