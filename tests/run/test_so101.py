"""The SO-101 side of `pgs run`: the unit mapping to MolmoAct2-SO100_101 (round trips, the documented conversions, the
training data's range), the calibration file and limits, the command-line values, the missing-LeRobot message, and
SO101's calls into LeRobot against a stand-in driver (a dry run writes nothing to the motors).

With LeRobot installed (Python 3.12: `uv run --python 3.12 --extra so101 --no-group dev --with pytest pytest
tests/run`), test_lerobot_units also checks the names and units against the real driver."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from playground_serve.run import so101
from playground_serve.run.so101 import MolmoAct2SO101, RobotError

# An SO-101 calibration file as LeRobot writes it (raw encoder ticks; wrist_roll spans the whole turn).
CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1600, "range_min": 780, "range_max": 3320},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -970, "range_min": 880, "range_max": 3230},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 850, "range_min": 1100, "range_max": 3310},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1820, "range_min": 670, "range_max": 2970},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 590, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1990, "range_min": 1970, "range_max": 3480},
}
# allenai/MolmoAct2-SO100_101 norm_stats.json (so100_so101_molmoact2, observation.state), q01 and q99.
STATE_Q01 = np.array([-41.9, 43.67, 38.39, 5.71, -63.45, 0.94])
STATE_Q99 = np.array([48.29, 185.26, 173.14, 91.78, 42.94, 44.14])


@pytest.fixture
def calibrations(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    folder = tmp_path / "robots" / "so_follower"
    folder.mkdir(parents=True)
    return folder


def write(folder, name="arm", cal=None):
    path = folder / f"{name}.json"
    path.write_text(json.dumps(cal or CALIBRATION))
    return path


def test_card_state_is_the_arm_folded_on_the_table():
    robot = MolmoAct2SO101(so101.OLD_GRIPPER_TICKS).to_robot(so101.CARD_STATE)
    # shoulder_lift and elbow_flex near their ends (folded), as in the card's sample image
    assert np.allclose(robot, [-0.527, -99.141, 91.406, 60.645, -3.604, 1.097], atol=1e-3)


def test_backward_compatibility_formulas():
    # LeRobot's docs/source/backwardcomp.mdx: new shoulder_lift = -(old - 90); new elbow_flex = old - 90.
    m = MolmoAct2SO101(so101.OLD_GRIPPER_TICKS)
    old = np.array([10.0, 120.0, 130.0, 40.0, -20.0, 30.0])
    new = m.to_robot(old)
    assert new[1] == -(old[1] - 90) and new[2] == old[2] - 90
    assert np.allclose(new[[0, 3, 4, 5]], old[[0, 3, 4, 5]])


def test_models_are_the_registrys_so101_entries():
    # Every registered SO-101 entry has a mapping here (a new one fails this until it gets one), and every mapping
    # names a registered SO-101 entry whose actions are the arm's six values.
    from playground_serve import models

    assert set(so101.MODELS) == {n for n in models.names() if models.get(n).robot == "so101"}
    for name in so101.MODELS:
        assert models.get(name).action_shape[-1] == len(so101.JOINTS)


def test_round_trips():
    rng = np.random.default_rng(0)
    m = MolmoAct2SO101(1510, "x")
    robot = rng.uniform(-100, 100, size=(30, 6))
    assert np.allclose(m.to_robot(m.to_model(robot)), robot)
    model = rng.uniform(-50, 200, size=(30, 6))
    assert np.allclose(m.to_model(m.to_robot(model)), model)
    assert m.to_robot(model).shape == (30, 6) and m.to_model(robot[0]).shape == (6,)


def test_gripper_scale_from_the_calibration():
    # Old calibration: 100 % = a quarter turn (1024 ticks) from closed. New: 100 % = the recorded range.
    m = MolmoAct2SO101(2048)  # a half turn recorded: 50 % new is a quarter turn, 100 % old
    assert m.to_model([0, 0, 0, 0, 0, 50])[5] == 100 and m.to_robot([0, 0, 0, 0, 0, 100])[5] == 50
    with pytest.raises(ValueError):
        MolmoAct2SO101(0)


def test_training_range_fits_a_calibrated_arm():
    # The middle 98 % of the training states, in LeRobot's units, lies inside a calibrated SO-101's range.
    lower, upper = so101.limits(CALIBRATION, margin=0)
    m = MolmoAct2SO101(CALIBRATION["gripper"]["range_max"] - CALIBRATION["gripper"]["range_min"])
    for q in (STATE_Q01, STATE_Q99):
        robot = m.to_robot(q)
        assert np.all(robot >= lower) and np.all(robot <= upper), robot


def test_limits_from_the_calibration():
    lower, upper = so101.limits(CALIBRATION)
    half_lift = (3230 - 880) / 2 * 360 / 4095
    assert np.isclose(upper[1], half_lift - 2) and np.isclose(lower[1], -half_lift + 2)
    assert np.isclose(upper[4], 180 - 2) and lower[5] == 0 and upper[5] == 100


def test_find_calibration(calibrations, monkeypatch):
    with pytest.raises(RobotError, match="lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/x"):
        so101.find_calibration(None, "/dev/x")
    path = write(calibrations, "blue")
    assert so101.find_calibration(None) == path
    write(calibrations, "red")
    with pytest.raises(RobotError, match="several .*blue, red.*--id"):
        so101.find_calibration(None)
    assert so101.find_calibration("red").name == "red.json"
    with pytest.raises(RobotError, match="no LeRobot calibration for --id green"):
        so101.find_calibration("green")


def test_calibration_dir_follows_lerobot(monkeypatch, tmp_path):
    for var in ("HF_LEROBOT_CALIBRATION", "HF_LEROBOT_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    assert str(so101.calibration_dir()).endswith(".cache/huggingface/lerobot/calibration/robots/so_follower")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert so101.calibration_dir() == tmp_path / "lerobot" / "calibration" / "robots" / "so_follower"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    assert so101.calibration_dir() == tmp_path / "lr" / "calibration" / "robots" / "so_follower"


def test_bad_calibration(calibrations):
    broken = {**CALIBRATION, "elbow_flex": {**CALIBRATION["elbow_flex"], "range_max": 1100}}
    with pytest.raises(RobotError, match="elbow_flex"):
        so101.load_calibration(write(calibrations, "a", broken))
    missing = {k: v for k, v in CALIBRATION.items() if k != "gripper"}
    with pytest.raises(RobotError, match="cannot read"):
        so101.load_calibration(write(calibrations, "b", missing))


def test_parse_cameras():
    assert so101.parse_cameras("top=0,side=1") == {"top": 0, "side": 1}
    assert so101.parse_cameras(" side=/dev/video4 , top=2") == {"side": "/dev/video4", "top": 2}
    for bad in ("top=0", "top=0,side=1,wrist=2", "top,side=1", "top=,side=1", ""):
        with pytest.raises(ValueError, match="--cameras"):
            so101.parse_cameras(bad)


def test_missing_lerobot_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    with pytest.raises(RobotError) as e:
        so101.import_lerobot()
    assert 'pip install "playground-serve[so101]"' in str(e.value) and "Python 3.12" in str(e.value)
    assert "--sim" in str(e.value)


# ----- SO101 against a stand-in for LeRobot's driver
class FakeBus:
    def __init__(self, calibrated=True):
        self.calibrated, self.connected, self.writes, self.torque_disabled_at_close = calibrated, False, [], None

    def connect(self):
        self.connected = True

    @property
    def is_connected(self):
        return self.connected

    @property
    def is_calibrated(self):
        return self.calibrated

    def sync_read(self, name, normalize=True):
        assert name == "Present_Position" and normalize is False
        return {j: 2000 + i for i, j in enumerate(so101.JOINTS)}

    def sync_write(self, name, values, normalize=True):
        self.writes.append((name, values, normalize))

    def disconnect(self, disable_torque=True):
        self.connected, self.torque_disabled_at_close = False, disable_torque


class FakeCamera:
    def __init__(self, cfg):
        self.cfg, self.is_connected = cfg, False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


def fake_lerobot(monkeypatch, calibrated=True):
    made = []

    class Config:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Follower:
        def __init__(self, config):
            self.config, self.bus = config, FakeBus(calibrated)
            self.cameras = {k: FakeCamera(v) for k, v in config.cameras.items()}
            self.configured, self.actions = False, []
            made.append(self)

        def configure(self):
            self.configured = True

        def get_observation(self):
            obs = {f"{j}.pos": float(i) for i, j in enumerate(so101.JOINTS)}
            return {**obs, **{k: np.zeros((480, 640, 3), np.uint8) for k in self.cameras}}

        def send_action(self, action):
            self.actions.append(action)
            return action

    root = types.ModuleType("lerobot")
    cams = types.ModuleType("lerobot.cameras.opencv")
    cams.OpenCVCameraConfig = lambda **kw: kw
    robots = types.ModuleType("lerobot.robots.so_follower")
    robots.SO101Follower, robots.SO101FollowerConfig = Follower, Config
    for name, mod in {"lerobot": root, "lerobot.cameras": types.ModuleType("lerobot.cameras"),
                      "lerobot.cameras.opencv": cams, "lerobot.robots": types.ModuleType("lerobot.robots"),
                      "lerobot.robots.so_follower": robots}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return made


def test_connect_latches_the_pose_then_configures(monkeypatch, calibrations):
    made = fake_lerobot(monkeypatch)
    path = write(calibrations, "blue")
    arm = so101.SO101("/dev/ttyACM0", {"top": 0, "side": "/dev/video2"}, None)
    arm.connect()
    follower = made[0]
    c = follower.config
    assert c.use_degrees is True and c.id == "blue" and c.calibration_dir == path.parent and c.port == "/dev/ttyACM0"
    assert c.cameras["top"] == {"index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    assert follower.bus.writes == [("Goal_Position", {j: 2000 + i for i, j in enumerate(so101.JOINTS)}, False)]
    assert follower.configured and all(cam.is_connected for cam in follower.cameras.values())
    state, images = arm.read()
    assert np.allclose(state, range(6)) and set(images) == {"top", "side"}
    arm.send(np.arange(6.0))
    arm.hold()
    assert follower.actions == [{f"{j}.pos": float(i) for i, j in enumerate(so101.JOINTS)}] * 2
    arm.disconnect()
    assert follower.bus.torque_disabled_at_close is True and not follower.cameras["top"].is_connected
    assert arm.gripper_span_ticks == 3480 - 1970 and np.isclose(arm.upper[0], (3320 - 780) / 2 * 360 / 4095 - 2)


def test_dry_run_writes_nothing_to_the_motors(monkeypatch, calibrations):
    made = fake_lerobot(monkeypatch)
    write(calibrations)
    arm = so101.SO101("/dev/ttyACM0", {"top": 0, "side": 1}, None, dry_run=True)
    arm.connect()
    follower = made[0]
    assert follower.bus.writes == [] and not follower.configured
    assert follower.config.disable_torque_on_disconnect is False
    arm.read()
    with pytest.raises(RuntimeError, match="dry run"):
        arm.send(np.zeros(6))
    arm.disconnect(torque_off=True)
    assert follower.bus.torque_disabled_at_close is False and follower.actions == []


def test_a_failed_setup_turns_the_torque_off(monkeypatch, calibrations):
    made = fake_lerobot(monkeypatch)
    write(calibrations)

    def configure_fails():
        raise ConnectionError("no answer from motor 4")

    arm = so101.SO101("/dev/ttyACM0", {"top": 0, "side": 1}, None)
    real_follower = sys.modules["lerobot.robots.so_follower"].SO101Follower

    def follower(config):
        robot = real_follower(config)
        robot.configure = configure_fails
        return robot

    monkeypatch.setattr(sys.modules["lerobot.robots.so_follower"], "SO101Follower", follower)
    with pytest.raises(ConnectionError):
        arm.connect()
    assert made[0].bus.torque_disabled_at_close is True and not made[0].cameras["top"].is_connected


def test_uncalibrated_motors_are_refused(monkeypatch, calibrations):
    made = fake_lerobot(monkeypatch, calibrated=False)
    write(calibrations, "blue")
    arm = so101.SO101("/dev/ttyACM0", {"top": 0, "side": 1}, None)
    with pytest.raises(RobotError, match="do not hold the calibration.*lerobot-calibrate.*--robot.id=blue"):
        arm.connect()
    bus = made[0].bus
    assert bus.writes == [] and bus.connected is False and bus.torque_disabled_at_close is False


def test_lerobot_units():
    """The real driver, without an arm: the class names, use_degrees, and the formulas so101.py relies on."""
    pytest.importorskip("lerobot.robots.so_follower")
    from lerobot.motors import MotorCalibration, MotorNormMode
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port="/dev/null", id="test", use_degrees=True, cameras={}))
    assert list(robot.bus.motors) == list(so101.JOINTS)
    modes = [m.norm_mode for m in robot.bus.motors.values()]
    assert modes == [MotorNormMode.DEGREES] * 5 + [MotorNormMode.RANGE_0_100]
    robot.bus.calibration = {k: MotorCalibration(**v) for k, v in CALIBRATION.items()}
    ids = {robot.bus.motors[j].id: j for j in so101.JOINTS}
    raw = {1: 780, 2: 2055, 3: 1100, 4: 2970, 5: 4095, 6: 1970 + 755}
    norm = robot.bus._normalize(raw)
    for i, value in norm.items():
        c = CALIBRATION[ids[i]]
        if ids[i] == "gripper":
            assert np.isclose(value, 50.0)
        else:
            assert np.isclose(value, (raw[i] - (c["range_min"] + c["range_max"]) / 2) * 360 / so101.TICKS)
