"""The SO-101 arm for `pgs run so101`: LeRobot's SO-101 follower driver with its calibration file, two OpenCV cameras,
and the unit mapping between LeRobot's values and MolmoAct2-SO100_101's. docs/RUN.md lists the sources.

LeRobot (pinned in the `so101` extra, LEROBOT) is imported only in `SO101.connect()`, so the mapping, the calibration
file and the limits work, and are tested, without it.

Robot units, what LeRobot's `SOFollower` reads and takes with `use_degrees=True` (lerobot/motors/motors_bus.py
`_normalize`): joints 1 to 5 in degrees, (raw - mid) * 360 / 4095 with mid the middle of the range recorded at
calibration; the gripper in percent of its recorded range, 0 closed and 100 open.

Model units, what allenai/MolmoAct2-SO100_101 was trained on (norm_stats.json, tag so100_so101_molmoact2: joint order
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper): LeRobot's calibration from before its
PR #777, in which the joints are degrees from the arm stretched out horizontally and the gripper is percent of a
quarter turn from closed. The 1220 community datasets the checkpoint was trained on were recorded that way (q01 to q99
of shoulder_lift: 44 to 185 degrees).

    model joint 1 to 5 = SIGN * robot + OFFSET  (shoulder_lift: 90 - robot, elbow_flex: robot + 90, others equal)
    model gripper      = robot gripper * (range_max - range_min) / OLD_GRIPPER_TICKS
"""

from __future__ import annotations

import glob
import json
import os
import pathlib

import numpy as np

from .loop import Robot

LEROBOT = "0.7.0"  # the version `pip install "playground-serve[so101]"` installs; read for the names and units here
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICKS = 4095  # STS3215: 4096 positions a turn; LeRobot's DEGREES mode scales by 360 / 4095
# LeRobot's backward-compatibility page (docs/source/backwardcomp.mdx, "Hardware API redesign") and the reference
# SO-101 deployment the MolmoAct2 README links (github.com/irenegracekp/molmoact2-so101 at 9819f6a, inference.py
# defaults --joint-signs 1,-1,1,1,1,1 --joint-offsets 0,90,90,0,0,0). Checked against TheRobotStudio's two MuJoCo files
# of the arm (so101_old_calib.xml and so101_new_calib.xml): their forward kinematics agree to 2 mm under this map.
SIGN = np.array([1, -1, 1, 1, 1, 1], dtype=np.float64)
OFFSET = np.array([0, 90, 90, 0, 0, 0], dtype=np.float64)
# Old calibration (lerobot/common/robot_devices/robots/feetech_calibration.py before PR #777): the gripper's 100 % is
# the "rotated position", a quarter turn (convert_degrees_to_steps(90) = 1024 steps) from the closed "zero position".
OLD_GRIPPER_TICKS = 1024.0
# The model card's sample state (Beegbrain/pick_lemon_and_drop_in_bowl, frame 0: the arm folded on the table), model
# units. Tests and the simulator use it.
CARD_STATE = np.array([-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971786975860596])
CAMERA = {"width": 640, "height": 480, "fps": 30}  # the size of the model card's sample frames
LIMIT_MARGIN_DEG = 2.0  # targets stay this far inside each joint's calibrated range (not the gripper's)
# Speed caps, robot units per second: degrees for the joints, percent for the gripper. The model card's sample chunk
# moves shoulder_lift by up to 2 degrees per 1/30 s step (60 degrees per second).
MAX_SPEED = np.array([120.0] * 5 + [150.0])
# The first target may lie at most this far from the arm's reading, else nothing moves (loop.Settings.max_start_jump):
# a bigger jump of a joint means wrong units or a wrong calibration. The gripper may open or close at once.
MAX_START_JUMP = (30.0,) * 5 + (100.0,)
CALIBRATE = ("lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id={id}")


class RobotError(RuntimeError):
    """The arm cannot be used as asked; the message says what to do."""


def missing_lerobot() -> str:
    return (f'pgs run so101 needs LeRobot {LEROBOT} (its SO-101 driver, which needs Python 3.12 or later): '
            f'pip install "playground-serve[so101]" in a Python 3.12 environment. `pgs run so101 --sim` runs without '
            'it (pip install "playground-serve[sim]").')


def import_lerobot():
    try:
        import lerobot  # noqa: F401
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        raise RobotError(missing_lerobot()) from None
    return SO101Follower, SO101FollowerConfig, OpenCVCameraConfig


# ----- calibration
def calibration_dir() -> pathlib.Path:
    """Where LeRobot keeps the SO-101 follower's calibration files (lerobot/utils/constants.py, robots/robot.py: the
    robot class's name is "so_follower")."""
    if os.environ.get("HF_LEROBOT_CALIBRATION"):
        root = pathlib.Path(os.environ["HF_LEROBOT_CALIBRATION"])
    elif os.environ.get("HF_LEROBOT_HOME"):
        root = pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / "calibration"
    else:
        hf = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache", "huggingface")
        root = pathlib.Path(hf) / "lerobot" / "calibration"
    return root.expanduser() / "robots" / "so_follower"


def find_calibration(robot_id: str | None, port: str = "PORT") -> pathlib.Path:
    """The calibration file of `robot_id`, or the only one there is."""
    folder = calibration_dir()
    how = CALIBRATE.format(port=port, id=robot_id or "my_so101")
    if robot_id:
        path = folder / f"{robot_id}.json"
        if not path.is_file():
            raise RobotError(f"no LeRobot calibration for --id {robot_id} ({path}): calibrate the arm with `{how}`")
        return path
    found = sorted(folder.glob("*.json"))
    if len(found) == 1:
        return found[0]
    if found:
        raise RobotError(f"several SO-101 calibrations in {folder} ({', '.join(p.stem for p in found)}): name the "
                         "arm's with --id")
    older = folder.parent / "so101_follower"
    extra = (f" (LeRobot before 0.4 kept them in {older}: calibrate again with LeRobot {LEROBOT}, or copy the file)"
             if older.is_dir() else "")
    raise RobotError(f"no LeRobot calibration of an SO-101 follower in {folder}{extra}: calibrate the arm once with "
                     f"`{how}`")


def load_calibration(path: pathlib.Path) -> dict:
    """{motor: {"id", "drive_mode", "homing_offset", "range_min", "range_max"}} as LeRobot writes it."""
    try:
        cal = json.loads(pathlib.Path(path).read_text())
        bad = [m for m in JOINTS if not (0 <= cal[m]["range_min"] < cal[m]["range_max"] <= TICKS)]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RobotError(f"cannot read the calibration {path} ({type(e).__name__}: {e}): calibrate again") from None
    if bad:
        raise RobotError(f"calibration {path} has an empty or impossible range for {', '.join(bad)}: calibrate again")
    return cal


def limits(cal: dict, margin: float = LIMIT_MARGIN_DEG) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) in robot units: each joint's recorded range in LeRobot degrees, `margin` inside it (symmetric
    around 0, as LeRobot puts 0 in the middle of the range), and 0 to 100 for the gripper."""
    half = np.array([(cal[m]["range_max"] - cal[m]["range_min"]) / 2 * 360 / TICKS for m in JOINTS[:5]])
    return np.append(-half + margin, 0.0), np.append(half - margin, 100.0)


# ----- units
class MolmoAct2SO101:
    """allenai/MolmoAct2-SO100_101's observation and actions for an SO-101 read through LeRobot (module docstring).
    `gripper_span_ticks` is the gripper's calibrated range_max - range_min."""

    cameras = ("top", "side")  # the request keys are "<role>_cam"; the card: camera order does not matter

    def __init__(self, gripper_span_ticks: float, instruction: str = "") -> None:
        if gripper_span_ticks <= 0:
            raise ValueError("gripper_span_ticks must be positive")
        self.instruction = instruction
        self.scale = np.ones(len(JOINTS))
        self.scale[-1] = gripper_span_ticks / OLD_GRIPPER_TICKS

    def to_model(self, robot) -> np.ndarray:
        return SIGN * np.asarray(robot, dtype=np.float64) * self.scale + OFFSET

    def to_robot(self, model) -> np.ndarray:
        """Joint targets in robot units, for one action (6,) or a chunk (T, 6)."""
        return (np.asarray(model, dtype=np.float64) - OFFSET) * SIGN / self.scale

    def observation(self, state, images: dict) -> dict:
        obs = {f"{role}_cam": images[role] for role in self.cameras}
        return {**obs, "state": self.to_model(state).astype(np.float32), "instruction": self.instruction}


MODELS = {"molmoact2_so101": MolmoAct2SO101}  # the models `pgs run so101` can drive


# ----- command-line values
def parse_cameras(spec: str, roles=MolmoAct2SO101.cameras) -> dict:
    """"top=0,side=1" -> {"top": 0, "side": 1}: camera role to OpenCV device index or path."""
    cams = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        role, sep, dev = part.partition("=")
        if not sep or not role.strip() or not dev.strip():
            raise ValueError(f"--cameras takes ROLE=DEVICE pairs such as top=0,side=1, not {part!r}")
        dev = dev.strip()
        cams[role.strip()] = int(dev) if dev.isdigit() else dev
    if sorted(cams) != sorted(roles):
        raise ValueError(f"--cameras must name exactly the model's cameras {', '.join(roles)} (got "
                         f"{', '.join(cams) or 'none'}), for example top=0,side=1")
    return cams


def find_port() -> str:
    """The serial port of the arm's bus when exactly one USB serial device is attached."""
    ports = sorted(set(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")))
    if len(ports) == 1:
        return ports[0]
    seen = f"found {', '.join(ports)}" if ports else "found none"
    raise RobotError(f"pass --port with the arm's serial port ({seen}; `lerobot-find-port` names it)")


# ----- the arm
class SO101(Robot):
    """LeRobot's SO-101 follower (`SOFollower`, use_degrees=True) with two OpenCV cameras, in robot units.

    connect(): opens the bus, checks that the motors hold the calibration file's values (nothing is written if not),
    opens the cameras and, unless dry_run, applies LeRobot's motor settings (position mode, PID, the gripper's torque
    and current limits), which leaves the torque on, after setting each goal to the present position so that turning
    the torque on moves nothing. With dry_run nothing is written to the motors: their torque stays as it was.
    """

    joints = JOINTS
    max_speed = MAX_SPEED

    def __init__(self, port: str, cameras: dict, robot_id: str | None = None, dry_run: bool = False) -> None:
        self.port, self.cameras, self.dry_run = port, dict(cameras), dry_run
        self.calibration = find_calibration(robot_id, port)
        cal = load_calibration(self.calibration)
        self.lower, self.upper = limits(cal)
        self.gripper_span_ticks = cal["gripper"]["range_max"] - cal["gripper"]["range_min"]
        self.robot = None
        self.last: np.ndarray | None = None

    def connect(self) -> None:
        SO101Follower, SO101FollowerConfig, OpenCVCameraConfig = import_lerobot()
        cams = {role: OpenCVCameraConfig(index_or_path=dev, **CAMERA) for role, dev in self.cameras.items()}
        config = SO101FollowerConfig(port=self.port, id=self.calibration.stem, calibration_dir=self.calibration.parent,
                                     use_degrees=True, cameras=cams, disable_torque_on_disconnect=not self.dry_run)
        robot = SO101Follower(config)
        robot.bus.connect()
        wrote = False  # once anything is written, a failure turns the torque off on the way out
        try:
            if not robot.bus.is_calibrated:
                how = CALIBRATE.format(port=self.port, id=self.calibration.stem)
                raise RobotError(f"the motors on {self.port} do not hold the calibration in {self.calibration}: run "
                                 f"`{how}` and press ENTER to write that file to them (or c to calibrate again)")
            for cam in robot.cameras.values():
                cam.connect()
            if not self.dry_run:
                present = robot.bus.sync_read("Present_Position", normalize=False)
                wrote = True
                robot.bus.sync_write("Goal_Position", present, normalize=False)
                robot.configure()
        except BaseException:
            self._close(robot, torque_off=wrote)
            raise
        self.robot = robot

    def read(self) -> tuple[np.ndarray, dict]:
        obs = self.robot.get_observation()
        return (np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float64),
                {role: np.asarray(obs[role]) for role in self.cameras})

    def send(self, target) -> None:
        if self.dry_run:
            raise RuntimeError("dry run: no motor commands")
        target = np.asarray(target, dtype=np.float64)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})
        self.last = target

    def hold(self) -> None:
        if self.last is not None:
            self.send(self.last)

    def disconnect(self, torque_off: bool = True) -> None:
        if self.robot is not None:
            robot, self.robot = self.robot, None
            self._close(robot, torque_off and not self.dry_run)

    @staticmethod
    def _close(robot, torque_off: bool) -> None:
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=torque_off)
        finally:
            for cam in robot.cameras.values():
                if cam.is_connected:
                    cam.disconnect()
