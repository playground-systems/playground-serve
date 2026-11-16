"""`pgs run`: its arguments, the `pgs up` hint when no worker runs (it starts no GPU), the missing-LeRobot message, the
start prompt, and the whole command over the wire: the MolmoAct2 worker's own Policy and server with a stand-in for
predict_action, the client, the loop and a kinematic arm."""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest
from run_fakes import START, FakeRobot

import playground_serve.run as run_cmd
from playground_serve import cli, resolve
from playground_serve.run import sim, so101

PKG = pathlib.Path(cli.__file__).resolve().parent
FAMILY = PKG / "families" / "molmoact2"


@pytest.fixture
def key(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path))


def test_arguments():
    args = cli.parser().parse_args(["run", "so101", "--instruction", "put the cube in the bowl"])
    assert (args.robot, args.model, args.cameras, args.hz, args.max_seconds) == ("so101", None, "top=0,side=1", None,
                                                                                 60.0)
    assert not (args.dry_run or args.yes or args.sim) and args.port is None and args.id is None
    args = cli.parser().parse_args(["run", "so101", "--instruction", "x", "--model", "molmoact2_so101", "--port",
                                    "/dev/ttyACM0", "--cameras", "top=2,side=4", "--hz", "15", "--dry-run",
                                    "--max-seconds", "30", "--yes", "--sim", "--id", "blue"])
    assert (args.port, args.cameras, args.hz, args.max_seconds, args.id) == ("/dev/ttyACM0", "top=2,side=4", 15.0,
                                                                            30.0, "blue")
    assert args.dry_run and args.yes and args.sim
    for bad in (["run", "so101"], ["run", "ur5", "--instruction", "x"]):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(bad)


@pytest.mark.parametrize("state, says", [("missing", "no endpoint for molmoact2_so101"),
                                         ("cold", "no worker of molmoact2_so101 is running")])
def test_no_running_worker_prints_the_up_hint(key, monkeypatch, capsys, state, says):
    seen = []

    def fake_status(target, api_key=None, key_file=None):
        seen.append((target, api_key))
        return resolve.Status(state, None, {})

    monkeypatch.setattr(resolve, "status", fake_status)
    monkeypatch.setattr(resolve, "connect", lambda *a, **kw: pytest.fail("pgs run must not connect"))
    assert cli.main(["run", "so101", "--sim", "--instruction", "x"]) == 1
    err = capsys.readouterr().err
    assert says in err and "`pgs up molmoact2_so101`" in err and "starts no GPU" in err
    assert seen == [("molmoact2_so101", "test-key")]


def test_a_runpod_url_is_checked_too(key, monkeypatch, capsys):
    url = "wss://exampleendpoint.api.runpod.ai/ws"
    monkeypatch.setattr(resolve, "status", lambda target, api_key=None, key_file=None: resolve.Status("cold", None, {}))
    monkeypatch.setattr(resolve, "connect", lambda *a, **kw: pytest.fail("pgs run must not connect"))
    assert cli.main(["run", "so101", "--sim", "--instruction", "x", "--model", url]) == 1
    assert "`pgs up molmoact2_so101`" in capsys.readouterr().err


def test_bad_values(key, capsys):
    assert cli.main(["run", "so101", "--sim", "--instruction", "x", "--model", "pi05_libero"]) == 1
    assert "drives molmoact2_so101, not pi05_libero" in capsys.readouterr().err
    assert cli.main(["run", "so101", "--sim", "--instruction", "x", "--hz", "500"]) == 1
    assert "--hz" in capsys.readouterr().err
    assert cli.main(["run", "so101", "--instruction", "x", "--cameras", "top=0"]) == 1
    assert "--cameras must name exactly" in capsys.readouterr().err


def test_missing_lerobot(key, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    assert cli.main(["run", "so101", "--instruction", "x", "--port", "/dev/ttyACM0"]) == 1
    err = capsys.readouterr().err
    assert 'pip install "playground-serve[so101]"' in err and "Python 3.12" in err


def test_prompt(monkeypatch):
    from playground_serve.launcher import LauncherError

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(LauncherError, match="--yes"):
        run_cmd._confirm("will move", out=lambda *a: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    for answer, ok in (("y", True), ("yes", True), ("", False), ("n", False)):
        monkeypatch.setattr("builtins.input", lambda prompt, a=answer: a)
        if ok:
            run_cmd._confirm("will move", out=lambda *a: None)
        else:
            with pytest.raises(LauncherError, match="nothing moved"):
                run_cmd._confirm("will move", out=lambda *a: None)


# ----- over the wire
def load_serve():
    for d in (FAMILY, PKG / "families" / "common", PKG / "worker"):
        if str(d) not in sys.path:
            sys.path.append(str(d))
    spec = importlib.util.spec_from_file_location("molmoact2_serve_run_test", FAMILY / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Predict:
    """Stands in for MolmoAct2's predict_action: a chunk from the state it got, shoulder_pan +0.5 per action (model
    units), or `fixed` actions."""

    def __init__(self, fixed=None):
        self.calls, self.fixed = [], fixed

    def __call__(self, **kw):
        self.calls.append(kw)
        state = np.asarray(kw["state"], np.float64)
        actions = (self.fixed if self.fixed is not None else
                   state + np.arange(1, 31)[:, None] * np.array([0.5, 0, 0, 0, 0, 0]))

        class Out:
            pass

        Out.actions = np.asarray(actions, np.float32)[None]
        return Out()


@contextlib.contextmanager
def worker(predict):
    serve = load_serve()
    import pgs_server

    server = pgs_server.Server(serve.Policy("molmoact2_so101", predict), host="127.0.0.1", port=0).start()
    try:
        yield f"ws://127.0.0.1:{server.port}/ws"
    finally:
        server.stop()


@pytest.fixture
def arm(monkeypatch):
    robots = []

    def make(**kw):
        robots.append(FakeRobot())
        return robots[-1]

    monkeypatch.setattr(sim, "SimSO101", make)
    return robots


def test_over_the_wire(monkeypatch, arm, capsys, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no saved key: a ws:// URL needs none
    predict = Predict()
    with worker(predict) as url:
        code = cli.main(["run", "so101", "--sim", "--model", url, "--instruction", "Put the cube in the bowl.",
                         "--max-seconds", "1.2"])
    out = capsys.readouterr().out
    assert code == 0, out
    robot = arm[0]
    assert "Connected to molmoact2_so101" in out and "30 Hz" in out and "Stopped: time limit." in out
    assert "Policy:" in out and "Chunks:" in out
    first = predict.calls[0]
    # The worker got the state in the model's units, the instruction and both camera frames.
    assert np.allclose(first["state"], so101.MolmoAct2SO101(so101.OLD_GRIPPER_TICKS).to_model(START), atol=1e-4)
    assert first["task"] == "Put the cube in the bowl." and len(first["images"]) == 2
    assert first["norm_tag"] == "so100_so101_molmoact2"
    targets = robot.targets()
    assert targets[:, 0].max() > START[0] + 5 and np.allclose(targets[-1], START) and robot.torque_off is True


def test_wrong_units_are_refused_over_the_wire(monkeypatch, arm, capsys, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # All zeros in the model's units: shoulder_lift 0 is 90 degrees in LeRobot's, 150 from the arm.
    with worker(Predict(fixed=np.zeros((30, 6)))) as url:
        code = cli.main(["run", "so101", "--sim", "--model", url, "--instruction", "x"])
    err = capsys.readouterr().err
    assert code == 1 and "shoulder_lift by +150.0" in err and "nothing moved" in err
    assert arm[0].sent == []
