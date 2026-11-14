"""A new user's first day: no key yet (asked once on a terminal, checked, saved 0600; off a terminal, instructions and a
clear exit), the price and billing line before anything is created, the next step when ready, `pgs demo` against a
fake server, and plain advice for the common refusals (permission, credit, worker quota, stock)."""

from __future__ import annotations

import getpass
import io
import stat
import sys

import numpy as np
import pytest
from test_launcher import FakeApi, env  # noqa: F401  (env: the launcher fixture)

from playground_serve import cli, example_obs, state
from playground_serve.client.mock_server import Config, MockServer
from playground_serve.launcher import KEY_PAGE, LauncherError
from playground_serve.resolve import resolve_key
from playground_serve.runpod_api import ApiError

KEY = "test-key-0123456789"


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def no_key(tmp_path, monkeypatch):
    """No RUNPOD_API_KEY, an empty config dir, a state file in tmp_path; returns the list of keys the fake API got."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path / "proj"))
    keys = []

    class Api(FakeApi):
        def __init__(self, key):
            super().__init__()
            keys.append(key)

    monkeypatch.setattr(cli, "RunpodApi", Api)
    return keys


def test_off_a_terminal_no_key_explains_and_exits(no_key, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": pytest.fail("must not prompt off a terminal"))
    assert cli.main(["up", "pi05_libero"]) == 1
    err = capsys.readouterr().err
    assert "no Runpod API key" in err and KEY_PAGE in err and "RUNPOD_API_KEY" in err and "Read/Write" in err
    assert "Traceback" not in err and no_key == []


def test_on_a_terminal_the_key_is_asked_checked_saved_and_never_printed(no_key, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", Terminal())
    asked = []
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": asked.append(prompt) or f"  {KEY}\n")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert len(asked) == 1 and "hidden" in asked[0]
    assert KEY_PAGE in out and "saved to" in out and KEY not in out
    path = state.key_path()
    assert path.read_text().strip() == KEY and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert no_key[0] == KEY  # checked with one read before it was saved
    assert resolve_key() == KEY  # later commands and connect() find it
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": pytest.fail("asked again"))
    assert cli.main(["status"]) == 0


def test_a_refused_key_is_not_saved(no_key, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": KEY)

    class Refusing(FakeApi):
        def __init__(self, key):
            super().__init__()

        def list_endpoints(self):
            raise ApiError(401, "GET", "/serverless", "invalid api key")

    monkeypatch.setattr(cli, "RunpodApi", Refusing)
    assert cli.main(["up", "pi05_libero"]) == 1
    captured = capsys.readouterr()
    assert "nothing saved" in captured.err and "pasted the whole key" in captured.err
    assert KEY not in captured.out + captured.err and not state.key_path().exists()


@pytest.mark.parametrize("entry", ["  ", EOFError])
def test_an_empty_entry_saves_nothing(no_key, monkeypatch, capsys, entry):
    def ask(prompt=""):
        if entry is EOFError:
            raise EOFError  # Ctrl-D
        return entry

    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(getpass, "getpass", ask)
    assert cli.main(["up", "pi05_libero"]) == 1
    err = capsys.readouterr().err
    assert "no key entered" in err and "Traceback" not in err and not state.key_path().exists()


def test_save_key_tightens_an_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = state.key_path()
    path.parent.mkdir(parents=True)
    path.write_text("old\n")
    path.chmod(0o644)
    assert state.save_key(KEY) == path and path.read_text() == KEY + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ----- what `up` says before it creates anything and once it is ready


def test_price_and_billing_line_before_the_create_and_next_step_when_ready(env):  # noqa: F811
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    text = "\n".join(out)
    gpu = next(line for line in out if line.startswith("GPU:"))
    assert "1.34 USD/hr while a worker runs" in gpu and "stops by itself 5 min after the last request" in gpu
    assert text.index("GPU:") < text.index("Created volume")
    assert "connect(\"pi05_libero\")" in text and "When done: `pgs down`" in text


def test_ready_output_offers_the_demo_for_libero(tmp_path):
    from playground_serve.launcher import Launcher

    api, out = FakeApi(), []
    Launcher(api, tmp_path / "state.json", out=out.append, sleep=lambda s: None, measure=api.measure).up("pi05_libero")
    text = "\n".join(out)
    assert "`pgs demo pi05_libero`" in text
    assert ("from playground_serve import connect, example_obs\npolicy = connect(\"pi05_libero\")\n"
            "print(policy.act(example_obs()).shape)") in text


# ----- plain advice for the common refusals


def test_missing_permission_is_named(env):  # noqa: F811
    api, launcher, _, _ = env

    def refuse(body):
        raise ApiError(403, "POST", "/network-volumes", "this API key does not grant volumes:write")

    api.create_volume = refuse
    with pytest.raises(LauncherError, match="lacks the permission volumes:write") as e:
        launcher().up("pi05_libero")
    assert KEY_PAGE in str(e.value) and "Read/Write" in str(e.value)


def test_no_credit_stops_at_once_with_the_billing_page(env):  # noqa: F811
    api, launcher, _, _ = env

    def refuse(body):
        api.calls.append(("create_volume", body))
        raise ApiError(400, "POST", "/network-volumes", "insufficient balance to create this resource")

    api.create_volume = refuse
    with pytest.raises(LauncherError, match="needs credit") as e:
        launcher().up("pi05_libero")
    assert "console.runpod.io/user/billing" in str(e.value)
    assert len([c for c in api.calls if c[0] == "create_volume"]) == 1  # no other candidate tried
    assert api.endpoints == {}


def test_load_balancer_wording_is_not_taken_for_no_credit():
    from playground_serve.launcher import no_credit

    assert not no_credit(ApiError(400, "POST", "/serverless", "LOAD_BALANCER endpoints need a load balancer port"))
    assert no_credit(ApiError(402, "POST", "/serverless", "Your credit is too low"))


def test_full_worker_quota_says_what_to_do(env):  # noqa: F811
    api, launcher, out, _ = env
    quota = ApiError(400, "POST", "/serverless", "max workers across all endpoints must not exceed the account's "
                     "quota of 5")
    api.endpoint_errors.extend([quota] * 4)
    with pytest.raises(LauncherError, match="worker quota is full") as e:
        launcher().up("pi05_libero")
    assert "delete endpoints you do not use" in str(e.value)


def test_no_stock_says_to_try_again(env):  # noqa: F811
    api, launcher, _, _ = env
    for g in api.gpus:
        for d in g["dataCenters"]:
            d["availability"] = "NONE"
    with pytest.raises(LauncherError, match="try again in a few minutes"):
        launcher().up("pi05_libero")


def test_cli_adds_advice_to_a_refused_read(monkeypatch, tmp_path, capsys):
    class Refusing(FakeApi):
        def __init__(self, key):
            super().__init__()

        def list_endpoints(self):
            raise ApiError(403, "GET", "/serverless", "this API key does not grant serverless:read")

    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "RunpodApi", Refusing)
    assert cli.main(["down"]) == 1
    assert "lacks the permission serverless:read" in capsys.readouterr().err


# ----- pgs demo


def test_example_obs_is_libero_shaped():
    obs = example_obs()
    assert obs["observation/image"].shape == obs["observation/wrist_image"].shape == (224, 224, 3)
    assert obs["observation/image"].dtype == np.uint8 and obs["observation/state"].shape == (8,)
    assert isinstance(obs["prompt"], str)


def test_demo_against_a_fake_server(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    with MockServer(Config(infer_ms=2, api_key=KEY)) as mock:
        assert cli.main(["demo", mock.url]) == 0
        assert mock.cfg.requests == 6  # one to wake the worker, five timed
    out = capsys.readouterr().out
    assert "Actions: shape (10, 7)" in out and "Round trip over 5 more calls: median" in out
    assert "model time on the server" in out and KEY not in out


def test_demo_with_a_refused_key_is_one_sentence(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    with MockServer(Config(api_key="another-key")) as mock:
        assert cli.main(["demo", mock.url]) == 1
    err = capsys.readouterr().err
    assert err.startswith("pgs: demo failed:") and "Traceback" not in err and KEY not in err


def test_demo_refuses_models_without_an_example(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["demo", "pi05_droid"]) == 1
    assert "LIBERO models (pi05_libero) only" in capsys.readouterr().err
    assert cli.main(["demo", "no_such_model"]) == 1
    assert "unknown model" in capsys.readouterr().err
