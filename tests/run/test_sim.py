"""The simulated SO-101 without downloading the arm: a stand-in MuJoCo arm with the SO-101's joint names, ranges and
position servos (capsules instead of meshes), the scene, the units, real-time stepping, rendering; and the download's
checksum check."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from playground_serve.run import sim, so101

mujoco = pytest.importorskip("mujoco")

ARM = """<mujoco model="stand_in_arm">
  <compiler angle="radian" autolimits="true"/>
  <default><position kp="50" forcerange="-3 3"/><joint damping="0.5" armature="0.01"/></default>
  <worldbody>
    <body name="base">
      <geom type="cylinder" size="0.03 0.02" pos="0 0 0.02" rgba="1 0.8 0.1 1"/>
      <body name="shoulder" pos="0 0 0.05"><joint name="shoulder_pan" axis="0 0 1" range="-1.92 1.92"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.015"/>
        <body name="upper" pos="0 0 0.05"><joint name="shoulder_lift" axis="0 1 0" range="-1.745 1.745"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.11" size="0.015"/>
          <body name="lower" pos="0 0 0.11"><joint name="elbow_flex" axis="0 1 0" range="-1.69 1.69"/>
            <geom type="capsule" fromto="0 0 0 0.13 0 0" size="0.012"/>
            <body name="wrist" pos="0.13 0 0"><joint name="wrist_flex" axis="0 1 0" range="-1.658 1.658"/>
              <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.01"/>
              <body name="hand" pos="0.04 0 0"><joint name="wrist_roll" axis="1 0 0" range="-2.74 2.84"/>
                <geom type="box" size="0.02 0.01 0.01"/>
                <body name="jaw" pos="0.02 0 0"><joint name="gripper" axis="0 0 1" range="-0.1745 1.745"/>
                  <geom type="box" size="0.03 0.004 0.004" pos="0.03 0 0"/>
                </body></body></body></body></body></body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" ctrlrange="-1.92 1.92"/>
    <position name="shoulder_lift" joint="shoulder_lift" ctrlrange="-1.745 1.745"/>
    <position name="elbow_flex" joint="elbow_flex" ctrlrange="-1.69 1.69"/>
    <position name="wrist_flex" joint="wrist_flex" ctrlrange="-1.658 1.658"/>
    <position name="wrist_roll" joint="wrist_roll" ctrlrange="-2.74 2.84"/>
    <position name="gripper" joint="gripper" ctrlrange="-0.1745 1.745"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot(tmp_path):
    yield from make_robot(tmp_path)


def make_robot(tmp_path, **kwargs):
    (tmp_path / "arm.xml").write_text(ARM)
    r = sim.SimSO101(model_dir=tmp_path, arm_file="arm.xml", **kwargs)
    try:
        r.connect()
    except Exception as e:  # noqa: BLE001  (no OpenGL context on this machine)
        if r.model is not None and r.renderer is None:
            pytest.skip(f"MuJoCo cannot render here: {e}")
        raise
    yield r
    r.disconnect()


def test_units_and_limits(robot):
    assert np.allclose(robot.from_qpos(robot.to_qpos([10, -20, 30, -40, 50, 25])), [10, -20, 30, -40, 50, 25])
    assert np.isclose(robot.to_qpos([0, 0, 0, 0, 0, 100])[5], np.deg2rad(sim.GRIPPER_OPEN_DEG))
    assert np.allclose(robot.lower[:2], [np.rad2deg(-1.92) + 2, np.rad2deg(-1.745) + 2])
    assert robot.lower[5] == 0 and robot.upper[5] == 100
    # Starts folded, at the model card's sample state in LeRobot's units.
    state, images = robot.read()
    rest = so101.MolmoAct2SO101(robot.gripper_span_ticks).to_robot(so101.CARD_STATE)
    assert np.allclose(state, rest, atol=1.0)
    assert set(images) == {"top", "side"}
    for frame in images.values():
        assert frame.shape == (sim.HEIGHT, sim.WIDTH, 3) and frame.dtype == np.uint8 and frame.std() > 1


def test_servos_follow_targets_in_real_time(tmp_path):
    now = [100.0]  # the simulator's clock, moved by hand: what is checked does not depend on this machine's speed
    for robot in make_robot(tmp_path, clock=lambda: now[0]):
        dt = robot.model.opt.timestep
        state, _ = robot.read()
        target = state.copy()
        target[0] += 20  # shoulder_pan
        robot.send(target)
        sim0 = robot.data.time
        for _ in range(12):  # 0.6 s, a read every 0.05 s
            now[0] += 0.05
            state, _ = robot.read()
        assert abs(state[0] - target[0]) < 2
        # Simulated time follows the clock to within one physics step, never ahead of it ...
        assert 0.6 - dt - 1e-9 <= robot.data.time - sim0 <= 0.6 + 1e-9
        # ... and at most 0.1 s per read: after a pause the simulation falls behind instead of replaying the gap.
        before = robot.data.time
        now[0] += 5.0
        robot.read()
        assert 0.1 - dt - 1e-9 <= robot.data.time - before <= 0.1 + 1e-9
        now[0] += 0.05
        before = robot.data.time
        robot.read()
        assert 0.05 - dt - 1e-9 <= robot.data.time - before <= 0.05 + dt + 1e-9  # real time again from there


def test_scene_has_the_two_cameras():
    xml = sim.scene_xml("x.xml")
    assert '<include file="x.xml"/>' in xml and xml.count("<camera ") == 2
    assert 'name="top"' in xml and 'name="side"' in xml and 'name="cube"' in xml and 'name="bowl"' in xml


def test_fetch_checks_the_files(monkeypatch, tmp_path):
    good, bad = b"<mujoco/>", b"tampered"
    monkeypatch.setattr(sim, "FILES", {"so101_new_calib.xml": hashlib.sha256(good).hexdigest()})
    urls = []

    def opener(data):
        def open_url(url, timeout):
            urls.append(url)
            return io.BytesIO(data)
        return open_url

    with pytest.raises(OSError, match="checksum"):
        sim.fetch(tmp_path, out=lambda *a: None, opener=opener(bad))
    assert not (tmp_path / "so101_new_calib.xml").exists()
    assert sim.fetch(tmp_path, out=lambda *a: None, opener=opener(good)) == tmp_path
    assert (tmp_path / "so101_new_calib.xml").read_bytes() == good
    assert urls[-1] == sim.URL.format(commit=sim.COMMIT, path="so101_new_calib.xml")
    sim.fetch(tmp_path, out=lambda *a: None, opener=opener(bad))  # present and intact: not downloaded again
    assert len(urls) == 2
