# Contributing

Issues and pull requests are welcome. For a bug, open an issue with the bug report form; for a model you would like
served, the model request form. Security problems go through [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and git. No GPU and no Runpod account: the tests use stand-ins for both.

```bash
git clone https://github.com/playground-systems/playground-serve
cd playground-serve
uv sync                 # Python 3.10+, with the dev group: pytest, the real openpi client, MuJoCo
uv run pgs --help
```

## Tests and the release check

```bash
uv run pytest -q          # the full suite, under two minutes on a laptop
scripts/check_release.sh  # no keys, tokens, home-directory paths, e-mail or IP addresses, private names
```

Both run on every pull request and every push to `main` ([.github/workflows/tests.yml](.github/workflows/tests.yml)):
the suite on Python 3.10 to 3.14, the client tests on Python 3.8, and the release check.

- The client (`from playground_serve import connect`) supports Python 3.8. The uv lock covers 3.10 and later, so the
  3.8 check has its own environment; [tests/client/test_import.py](tests/client/test_import.py) has the commands.
- Five tests skip in the development environment because it does not install what they need: openpi
  (`tests/test_models_openpi.py` and `tests/test_bundle_worker.py`, whose docstrings say what they need), torch (two
  Cosmos 3 checks) and LeRobot (`test_lerobot_units`; [docs/RUN.md](docs/RUN.md#development) has the command). None
  needs a GPU.
- `tests/run/test_sim.py` skips where MuJoCo cannot render. On Linux without a display, `MUJOCO_GL=egl` renders with
  Mesa's EGL (on Ubuntu: `libegl1`, `libegl-mesa0`, `libgl1-mesa-dri`), as the workflow does.
- `python3 docs/assets/make_assets.py` redraws the README's chart (from MODELS.md) and terminal picture;
  `uv run python docs/assets/make_social_preview.py` redraws the social preview.

## Adding a model family

A family is one folder, `src/playground_serve/families/<family>/`, and changes no shared file: `models.py` finds it by
itself. The existing families are the examples ([MolmoAct2](src/playground_serve/families/molmoact2/) is a complete
one), and [MODELS.md](MODELS.md#model-families) describes the volume layout and the checkpoint URIs.

- `spec.py` declares the entries (`SPECS` of `ProposedSpec` from `families/common/proposed_spec.py`): checkpoint at a
  pinned revision, GPU pools in order of preference, volume size, RAM, action shape, robot, license, secrets.
- `serve.py` serves one checkpoint behind the shared websocket server (`families/common/pgs_server.py`). Its stock path
  is the model's official inference code, called as the official server or evaluator calls it, and checked against it
  on the same inputs.
- `build_env.sh` builds the family's Python environment on the worker.
- `NOTES.md` holds the license of the code and the weights, what was measured, tried and left out, and the GPU spend.
- `tests/families/test_<family>.py` covers the spec, the bundle and whatever runs without a GPU.

A fast path ships only under these rules:

1. Same model: the same weights, precision, number of steps and inputs as the stock path. What may change is how the
   work reaches the GPU (CUDA graphs, compiled kernels, work done once per prompt, preprocessing on the GPU).
2. A boot self-test: `selftest_<entry>.json` holds a real frame and the actions the stock path returned for it. At
   every start the fast path must reproduce them within fixed bounds, or the worker serves stock and says why
   (`path_reason` in its metadata). A fast path serves only on the GPUs where it was measured.
3. Numbers with their conditions: every speed or difference in NOTES.md, MODELS.md or the README names the GPU, and the
   region and date where they matter, and gives the stock number next to the fast one.

## Commits and pull requests

- One change per pull request, with a short description of what changed and how you checked it.
- Commit messages: a first line under 80 characters, more lines below it when needed.
- Before you open it: `uv run pytest -q` and `scripts/check_release.sh` pass, and the docs say what is now true. No
  API keys, tokens, `.env` or state files in the tree.
- By contributing, you agree that your contribution is licensed under the Apache License 2.0, as section 5 of
  [LICENSE](LICENSE) says.
