# LLM agent family

A language model as a robot policy, behind the same websocket wire as every other Playground Serve model. It needs no
GPU. The agent models are exactly nine, each a registry entry (`spec.py`, from `llm_models.py`), all reached through
OpenRouter with one key, `OPENROUTER_API_KEY`. `serve.py` is the server, `llm_session.py` one episode, `llm_models.py`
the nine and the OpenRouter request settings, `llm_card.py` the embodiment cards, `llm_pricing.py` the prices and caps,
`llm_fake.py` the recorded-response provider for tests and dry runs, and `cards.json` the built-in cards.

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
  one client per wire. This family uses one of them: Chat Completions at OpenRouter's base URL. The package's other
  clients (Anthropic Messages, OpenAI Responses, Google's wires) and its provider prefixes stay inside the pinned
  dependency, unused; the settings cannot reach them (the model must be one of the nine, the base URL is fixed).
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
| inspect-robots-agent | MIT | tools over the embodiment's spaces | its own clients for five wires; OpenRouter by base URL | the whole loop, as a dependency |
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
That keeps every replayed reasoning block valid and the prompt cache warm (for Fable 5.2 and Opus 5.5 the request
carries `cache_control`, `llm_models.py`). At
LIBERO's 256-pixel images a whole episode stays far below any context limit (the measured episodes below used 40K to
420K prompt tokens, most of them cache reads).
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

## OpenRouter (read 2026-11-17)

From openrouter.ai/docs (API reference, reasoning tokens, prompt caching, provider routing, errors, usage accounting):

- One endpoint, `https://openrouter.ai/api/v1/chat/completions`, the Chat Completions shape, a bearer key. The model is
  the slug (`anthropic/claude-opus-5.5`). Images are `image_url` parts with data URLs; tools are Chat Completions
  functions.
- Reasoning: one `reasoning: {effort}` object, mapped to each provider's own setting (Anthropic's thinking budget,
  OpenAI's effort, Gemini's thinking level). `OpenRouterTransport` moves the agent's `reasoning_effort` there and sets
  each model's own effort (`high` for Gemini 3.8 Flash, `medium` for the rest).
- Caching: the other providers cache a repeated prompt on their own where they cache at all; Anthropic models need
  `cache_control`, sent at the top level for Fable 5.2 and Opus 5.5.
- Cost: each response's `usage.cost` is the USD charged for that call (credits), reasoning and cache included. The
  server's caps and each reply's `usd` use that number (`usd_source: "reported"`); the list prices in `llm_models.py`
  are used only when a response lacks it.
- Routing: `provider: {require_parameters: true, data_collection: "deny", allow_fallbacks: true}`, so a provider that
  ignores tools or images, or keeps prompts, is never chosen. None of the nine lists `parallel_tool_calls`, so it is
  not sent; Gemini models take `max_tokens`, so that is the field set.
- Errors: 401 the key, 402 no credit, 403 moderation, 408 timeout, 429 rate limit, 502 the provider failed, 503 no
  provider meets the request; an error can also arrive inside a 200 body. The agent retries with backoff; what is left
  fails the request, named by its class (`credit: ...`, `key: ...`), as the server's error frame.

The Kimi entry: the list names "Kimi 3.1"; OpenRouter lists one Kimi 3.1 model, Kimi K3.1 Pro
(`moonshotai/kimi-k3.1-pro`), and that is what `kimi_3_1` serves. Its title stays "Kimi 3.1" and `pgs info kimi_3_1`
shows the slug.

## Keys and logs

The settings come from the endpoint's environment only (`PGS_AGENT_MODEL`, one of the nine, and the caps); a request
cannot change the model, the key or the base URL. The episode's policy gets a mapping with `OPENROUTER_API_KEY` only,
so the key goes only in the request header to OpenRouter. The log has counts: no prompt, image, note or key; httpx's
request lines are off. Each reply's `agent.turns` gives the client the episode's new messages without images.

Tokens and cost are counted by `Meter`, an httpx transport around the agent's: it reads the usage object (and its
`cost`) of each answered request, and the USD caps use those numbers.

## Measured (this Mac, 2026-11-17)

One tool-calling turn per model first: a recorded LIBERO frame pair (two cameras), the tools, the model's reasoning
setting. All nine answered with a tool call and reported their cost.

| Model | Tool called | Seconds | USD |
|---|---|---|---|
| Fable 5.2 | move_to | 30.6 | 0.130 |
| Opus 5.5 | move_to | 45.1 | 0.100 |
| Gemini 4 Carbon | move_to | 221 | 0.156 |
| Gemini 4 Argon | move_to | 7.9 | 0.011 |
| Gemini 3.8 Flash (high) | zoom | 124.6 | 0.061 |
| GPT-6 Astra | move_to | 13.8 | 0.047 |
| GPT 6.1 Sol | move_to | 12.8 | 0.0096 |
| Muse Wonder 1.0 | move_to | 11.4 | 0.037 |
| Kimi 3.1 | move_to | 10.2 | 0.014 |

Then LIBERO-Spatial task 1 ("pick up the black bowl between the plate and the ramekin and place it on the plate"), 220
steps, three episodes per model (vla-eval's first three initial states), each model's own reasoning setting, the
generic card and notes above, through the engine's native runtime and its openpi proxy. Success is LIBERO's own. The
cost is OpenRouter's reported cost.

| Model | Success | Calls | Seconds per call | USD per episode | What it struggles with |
|---|---|---|---|---|---|
| Fable 5.2 | 2 of 3 | 10, 5, 7 | 8.0 to 8.5 | 0.18 to 0.30 | Took the bowl by its rim at the first close every time; its failure released the bowl beside the plate's centre |
| Opus 5.5 | 1 of 3 | 10, 7, 9 | 7.0 to 7.6 | 0.11 to 0.14 | Rim grasp at the first close every time; two releases LIBERO did not count on the plate (it overshot, then said done) |
| Gemini 4 Carbon | 2 of 3 | 11, 7, 7 | 66 to 104 | 0.32 to 0.64 | Clean rim grasps; very slow (23K to 32K output tokens an episode); one release off the plate |
| Gemini 4 Argon | 1 of 3 | 5, 7, 8 | 6.2 to 7.0 | 0.04 to 0.06 | Fast rim grasps; two releases off the plate that it reported as done |
| Gemini 3.8 Flash (high) | 2 of 3 | 9, 8, 8 | 27 to 36 | 0.18 to 0.21 | Long reasoning (29K to 39K output tokens); knocked the bowl aside while repositioning once, then gave up |
| GPT-6 Astra | 0 of 3 | 9, 11, 10 | 6.9 to 7.6 | 0.19 to 0.22 | First close at the bowl's centre (wider than the jaws); the rim retry held, then the steps ran out at the plate |
| GPT 6.1 Sol | 2 of 3 | 11, 9, 8 | 7.3 to 8.3 | 0.035 to 0.046 | First close often inside the bowl, then a rim pinch; once pushed the bowl away and gave up |
| Muse Wonder 1.0 | 0 of 3 | 13, 9, 13 | 6.6 to 9.7 | 0.22 to 0.37 | Aimed at the bowl's centre and closed on nothing, repeatedly; used zoom most (4 calls) but ran out of steps |
| Kimi 3.1 | 0 of 3 | 7, 11, 10 | 4.9 to 6.8 | 0.03 to 0.07 | Grasps held, but the bowl slipped out in one fast carry and two releases missed the plate though it said done |

What the transcripts show: the bowl is wider than the open jaws, so a grasp holds only on its rim. Fable 5.2, Opus 5.5
and the two Gemini 4 models found that at the first close every time; GPT-6 Astra, GPT 6.1 Sol and Muse Wonder 1.0
first closed at the bowl's centre and met nothing. Among the models that held the bowl, most failures were the release:
held by its rim, the bowl's centre sits about 5 cm from the hand, and the bowl came down on the plate's edge or beside
it, after which the model called `done` without looking again. The change most likely to help next: a generic note to
look at the object after a release (and to offset a rim-held object's centre), then a larger step budget for the slow
reasoners. `zoom` was called 7 times in 27 episodes (Muse Wonder 4, Kimi, Gemini 3.8 Flash and Gemini 4 Carbon once each).
Prompt tokens per episode: 40K to 420K, most of them cache reads. Spend: 4.84 USD for the 27 episodes,
plus 0.57 USD for the one-turn checks.

Three episodes per model is a small sample: these counts say what the harness can do with each model, not how the
models rank.

Tests: `tests/families/test_llm_agent.py`: a recorded episode through OpenRouter (the request body: slug, reasoning,
caching, provider routing), one recorded tool call for each of the nine, OpenRouter's failure classes, the settings
refusing any model outside the nine, the driver against a stand-in LIBERO, the custom environments' cards, caps, a
rejected key, the server over the websocket.

## Not built

- `pgs up <name>` makes a CPU load-balancing endpoint (cpu3c, 2 vCPU, no volume) with `start.sh` as its boot; it has not
  run on Runpod yet.
- Code-as-policy mode; a card for LIBERO's floor and living-room scenes; a higher-resolution render on request in LIBERO
  (vla-eval renders it at 256 pixels, fixed), which the reply's `render` hint asks of clients that can.
