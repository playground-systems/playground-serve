"""A simulated SO-101 for `pgs run so101 --sim`: the whole loop (cameras, units, policy, chunk playback, limits)
without an arm, as a smoke test of the plumbing and the timing. It says nothing about task success.

The arm is TheRobotStudio's MuJoCo model of the SO-101 (github.com/TheRobotStudio/SO-ARM100, Apache License 2.0,
Simulation/SO101/so101_new_calib.xml and its meshes at COMMIT), downloaded once (about 16 MB) into the cache and checked
against FILES. It stands on a table with a cube and a bowl; two cameras, "top" (looking down, the arm at the image's top
edge) and "side" (at table height, the arm on the left), render 640 x 480 frames, the two kinds of view of the model
card's sample.

Units are LeRobot's, as on the real arm, so the same mapping to the model runs: the model's joint zero is the middle of
each joint's range, as in LeRobot's calibration, and LeRobot's own kinematics feeds its degrees to this model's URDF
twin unchanged (lerobot/model/kinematics.py), so joints 1 to 5 are degrees of the MuJoCo joints. The gripper's percent
spans the jaw's 0 to 100 degrees (closed to open). The physics advances in real time at each read(); the servos are
the file's position actuators.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import time
import urllib.request

import numpy as np

from .loop import Robot
from .so101 import CARD_STATE, JOINTS, LIMIT_MARGIN_DEG, MAX_SPEED, TICKS, MolmoAct2SO101

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"  # TheRobotStudio/SO-ARM100, 2026-10-06
URL = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{commit}/Simulation/SO101/{path}"
FILES = {  # path under Simulation/SO101 -> SHA-256
    "so101_new_calib.xml": "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580",
    "assets/waveshare_mounting_plate_so101_v2.stl": "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9",
    "assets/sts3215_03a_v1.stl": "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    "assets/motor_holder_so101_base_v1.stl": "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b",
    "assets/wrist_roll_follower_so101_v1.stl": "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4",
    "assets/moving_jaw_so101_v1.stl": "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d",
    "assets/base_motor_holder_so101_v1.stl": "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424",
    "assets/upper_arm_so101_v1.stl": "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f",
    "assets/wrist_roll_pitch_so101_v2.stl": "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae",
    "assets/under_arm_so101_v1.stl": "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71",
    "assets/rotation_pitch_so101_v1.stl": "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34",
    "assets/motor_holder_so101_wrist_v1.stl": "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011",
    "assets/sts3215_03a_no_horn_v1.stl": "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887",
    "assets/base_so101_v2.stl": "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
}
WIDTH, HEIGHT = 640, 480
GRIPPER_OPEN_DEG = 100.0  # the jaw's upper limit in the model; 0 degrees is closed
# Camera position and the point it looks at (meters; the arm's base at the origin, facing +x), and the world direction
# that is "up" in the image.
CAMERAS = {"top": ((0.44, 0.0, 0.40), (0.21, 0.0, 0.0), (-1.0, 0.0, 0.0)),
           "side": ((0.34, -0.46, 0.15), (0.17, 0.0, 0.05), (0.0, 0.0, 1.0))}
MISSING = ('pgs run so101 --sim needs MuJoCo: pip install "playground-serve[sim]" (or pip install mujoco)')


def cache_dir() -> pathlib.Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return pathlib.Path(root).expanduser() / "playground-serve" / "so101-mujoco" / COMMIT[:12]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(folder: pathlib.Path | None = None, out=print, opener=urllib.request.urlopen) -> pathlib.Path:
    """The SO-101 model's files in `folder` (default cache_dir()), downloading the missing ones; returns the folder."""
    folder = folder or cache_dir()
    missing = [p for p, digest in FILES.items() if not (folder / p).is_file() or _sha256(folder / p) != digest]
    if missing:
        out(f"Downloading TheRobotStudio's SO-101 MuJoCo model ({len(missing)} files, about 16 MB, once) to {folder}")
    for path in missing:
        with opener(URL.format(commit=COMMIT, path=path), timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != FILES[path]:
            raise OSError(f"{path} from TheRobotStudio/SO-ARM100@{COMMIT[:7]} does not match its checksum")
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{path}.part"
        tmp.write_bytes(data)
        tmp.replace(folder / path)
    return folder


def _xyaxes(pos, target, up) -> str:
    pos, target, up = (np.asarray(v, dtype=np.float64) for v in (pos, target, up))
    forward = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return " ".join(f"{v:.4f}" for v in (*right, *np.cross(right, forward)))


def scene_xml(arm_file: str = "so101_new_calib.xml") -> str:
    """The arm on a table, a cube, a bowl (a disk and a ring of 16 slabs) and the two cameras."""
    wall = []
    for k in range(16):
        a = 2 * np.pi * k / 16
        wall.append(f'<geom type="box" size="0.004 0.013 0.016" pos="{0.058 * np.cos(a):.4f} {0.058 * np.sin(a):.4f} '
                    f'0.018" euler="0 0.25 {a:.4f}" material="red"/>')
    cams = "\n    ".join(f'<camera name="{name}" pos="{" ".join(map(str, pos))}" xyaxes="{_xyaxes(pos, at, up)}" '
                         f'fovy="58"/>' for name, (pos, at, up) in CAMERAS.items())
    return f"""<mujoco model="pgs_so101_scene">
  <include file="{arm_file}"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="wall" type="skybox" builtin="flat" rgb1="0.82 0.85 0.86" rgb2="0.82 0.85 0.86" width="64"
      height="64"/>
    <material name="table" rgba="0.72 0.60 0.45 1"/>
    <material name="red" rgba="0.72 0.07 0.07 1"/>
    <material name="cube" rgba="0.95 0.78 0.10 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" directional="true" diffuse="0.45 0.45 0.45"/>
    <geom name="table" type="plane" size="1 1 0.05" material="table"/>
    <body name="cube" pos="0.23 -0.07 0.0155">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" material="cube" mass="0.02" friction="1.5 0.01 0.001"/>
    </body>
    <body name="bowl" pos="0.24 0.10 0">
      <geom type="cylinder" size="0.05 0.002" pos="0 0 0.002" material="red"/>
      {chr(10).join("      " + w for w in wall).strip()}
    </body>
    {cams}
  </worldbody>
</mujoco>
"""


class SimSO101(Robot):
    """The simulated arm in LeRobot's units (module docstring). `model_dir` holds the arm's files (default: fetch())."""

    joints = JOINTS
    max_speed = MAX_SPEED
    gripper_span_ticks = GRIPPER_OPEN_DEG * TICKS / 360

    def __init__(self, model_dir: pathlib.Path | str | None = None, arm_file: str = "so101_new_calib.xml",
                 clock=time.monotonic, out=print) -> None:
        self.model_dir, self.arm_file, self.clock, self.out = model_dir, arm_file, clock, out
        self.lower = self.upper = None
        self.model = self.data = self.renderer = None
        self.last: np.ndarray | None = None

    # robot units <-> MuJoCo joint positions (radians)
    def to_qpos(self, robot) -> np.ndarray:
        robot = np.asarray(robot, dtype=np.float64)
        return np.append(np.deg2rad(robot[:5]), np.deg2rad(robot[5] / 100 * GRIPPER_OPEN_DEG))

    def from_qpos(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.append(np.rad2deg(q[:5]), np.rad2deg(q[5]) / GRIPPER_OPEN_DEG * 100)

    def connect(self) -> None:
        try:
            import mujoco
        except ImportError:
            from .so101 import RobotError

            raise RobotError(MISSING) from None
        self._mj = mujoco
        folder = pathlib.Path(self.model_dir) if self.model_dir else fetch(out=self.out)
        scene = folder / "pgs_scene.xml"
        scene.write_text(scene_xml(self.arm_file))
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self._qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self._act = np.array([self.model.actuator(j).id for j in JOINTS])
        lo, hi = self.from_qpos(self.model.jnt_range[[self.model.joint(j).id for j in JOINTS], 0]), \
            self.from_qpos(self.model.jnt_range[[self.model.joint(j).id for j in JOINTS], 1])
        self.lower = np.append(lo[:5] + LIMIT_MARGIN_DEG, 0.0)
        self.upper = np.append(hi[:5] - LIMIT_MARGIN_DEG, 100.0)
        rest = MolmoAct2SO101(self.gripper_span_ticks).to_robot(CARD_STATE)  # folded on the table, as in the card
        self.data.qpos[self._qadr] = self.to_qpos(rest)
        self.data.ctrl[self._act] = self.to_qpos(rest)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # 2 to 3 times faster on a Mac
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self._t = self.clock()

    def _advance(self) -> None:
        dt = self.model.opt.timestep
        steps = min(int((self.clock() - self._t) / dt), int(0.1 / dt))  # real time, at most 0.1 s per call
        if steps > 0:
            self._mj.mj_step(self.model, self.data, nstep=steps)
            self._t += steps * dt
        if self.clock() - self._t > 0.1:  # the caller paused: do not replay the gap
            self._t = self.clock()

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return np.array(self.renderer.render(), dtype=np.uint8, copy=True)

    def read(self) -> tuple[np.ndarray, dict]:
        self._advance()
        state = self.from_qpos(self.data.qpos[self._qadr])
        return state, {name: self.render(name) for name in CAMERAS}

    def send(self, target) -> None:
        self.last = np.asarray(target, dtype=np.float64)
        self.data.ctrl[self._act] = self.to_qpos(self.last)

    def hold(self) -> None:
        pass  # the actuators keep their last target

    def disconnect(self, torque_off: bool = True) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
