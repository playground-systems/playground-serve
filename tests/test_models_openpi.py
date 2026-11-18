"""Each registered model against openpi at the pinned commit, on CPU and without weights.

Checks that the config name exists, that the worker's warm-up input (warmup.py) passes the config's input transforms
and comes out with the shapes and dtypes the model's compiled sampler takes, and the action shape the output
transforms return. Needs an environment with openpi installed from a checkout at models.OPENPI_COMMIT (skipped
otherwise) and network access to the public openpi-assets bucket (one norm_stats.json per model, a few kB) and to
big_vision (the 4 MB PaliGemma tokenizer). Set OPENPI_DATA_HOME to reuse a download cache.
"""

import ast
import os
import pathlib
import subprocess
import sys
import urllib.request

import pytest

from playground_serve import models

openpi = pytest.importorskip("openpi")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import tokenizer as _tokenizer  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402

sys.path.insert(0, str(pathlib.Path(models.__file__).parent / "worker"))
import warmup  # noqa: E402  (the worker's own module)

OPENPI_REPO = pathlib.Path(openpi.__file__).resolve().parents[2]
# (action horizon, robot action dims) returned to the client; also in MODELS.md.
ACTIONS = {
    "pi05_libero": (10, 7), "pi05_droid": (15, 8), "pi0_fast_droid": (10, 8), "pi0_droid": (10, 8),
    "pi05_aloha": (50, 14), "pi0_aloha_towel": (50, 14), "pi0_aloha_tupperware": (50, 14),
    "pi0_aloha_pen_uncap": (50, 14), "pi0_aloha_sim": (50, 14),
}


@pytest.fixture(scope="session", autouse=True)
def data_home(tmp_path_factory):
    home = os.environ.get("OPENPI_DATA_HOME") or str(tmp_path_factory.mktemp("openpi_data"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENPI_DATA_HOME", home)
        yield pathlib.Path(home)


class _NoFastProcessor:
    """Stands in for the FAST action tokenizer, which openpi loads from Hugging Face with trust_remote_code. On the
    input side FASTTokenizer.tokenize uses it only for action tokens, which inference does not have."""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None


def _norm_stats_dir(data_home: pathlib.Path, checkpoint: str, asset_id: str) -> pathlib.Path:
    """<checkpoint>/assets/<asset_id>/norm_stats.json from the public bucket, laid out as in the checkpoint."""
    assert checkpoint.startswith("gs://openpi-assets/")
    path = data_home / "rs_test_assets" / checkpoint[len("gs://"):] / "assets"
    target = path / asset_id / "norm_stats.json"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://storage.googleapis.com/{checkpoint[len('gs://'):]}/assets/{asset_id}/norm_stats.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    return path


def test_openpi_is_the_pinned_commit():
    head = subprocess.run(["git", "-C", str(OPENPI_REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode:
        pytest.skip("openpi is not installed from a git checkout")
    assert head.stdout.strip() == models.OPENPI_COMMIT


def test_registry_matches_serve_policy_defaults():
    """serve_policy.py's DEFAULT_CHECKPOINT, read with ast (importing the script needs lerobot)."""
    tree = ast.parse((OPENPI_REPO / "scripts" / "serve_policy.py").read_text())
    table = next(node.value for node in ast.walk(tree)
                 if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DEFAULT_CHECKPOINT")
    found = {key.attr: {kw.arg: kw.value.value for kw in value.keywords} for key, value in zip(table.keys, table.values)}
    defaults = {"pi05_libero": "LIBERO", "pi05_droid": "DROID", "pi05_aloha": "ALOHA", "pi0_aloha_sim": "ALOHA_SIM"}
    assert set(found) == set(defaults.values())
    for name, env in defaults.items():
        spec = models.get(name)
        assert found[env] == {"config": spec.openpi_config, "dir": spec.checkpoint}, name


@pytest.mark.parametrize("name", models.names())
def test_warmup_input_passes_transforms(name, data_home, monkeypatch):
    spec = models.get(name)
    train_config = _config.get_config(spec.openpi_config)  # raises if the name is not at this commit
    model_config = train_config.model
    monkeypatch.setattr(_tokenizer, "AutoProcessor", _NoFastProcessor)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    assets = _norm_stats_dir(data_home, spec.checkpoint, data_config.asset_id)
    norm_stats = _normalize.load(assets / data_config.asset_id)  # what checkpoints.load_norm_stats does

    # The transforms exactly as openpi's create_trained_policy builds them (policies/policy_config.py, no repack).
    inputs = [
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    robot, make_example = warmup.family(inputs)
    assert robot == spec.robot

    # As Policy.infer: transform, batch, convert to jax arrays, build the Observation the sampler takes.
    transformed = _transforms.compose(inputs)(make_example())
    observation = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed))
    expected, _ = model_config.inputs_spec(batch_size=1)
    assert jax.tree.structure(observation) == jax.tree.structure(expected)
    assert [(a.shape, a.dtype) for a in jax.tree.leaves(observation)] == [
        (s.shape, s.dtype) for s in jax.tree.leaves(expected)]

    if model_config.model_type == _model.ModelType.PI0_FAST:
        # ExtractFASTActions decodes with the FAST tokenizer (not loaded here): (action_horizon, action_dim) by config.
        assert (model_config.action_horizon, model_config.action_dim) == ACTIONS[name]
        return
    outputs = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    sampled = np.zeros((model_config.action_horizon, model_config.action_dim), np.float32)
    result = _transforms.compose(outputs)({"state": np.asarray(transformed["state"]), "actions": sampled})
    assert result["actions"].shape == ACTIONS[name]
