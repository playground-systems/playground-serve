"""`pgs run`'s control loop against a kinematic robot and a stand-in policy: rate, chunk playback, holding on a late
reply, the limits, the refused start, Ctrl-C (back to the start pose, then torque off; twice: torque off at once), the
dry run and the stall."""

from __future__ import annotations

import os
import signal
import threading
import time

import numpy as np
import pytest
from run_fakes import START, FakePolicy, FakeRobot, mapping

from playground_serve.run import loop
from playground_serve.run.loop import Guard, Refused, Settings, Stop, ctrl_c, describe, run

QUIET = dict(status_s=0)


def test_guard_clamps_caps_and_rejects():
    g = Guard(lower=[-10, 0], upper=[10, 100], step=[2, 5])
    last = np.array([9.0, 50.0])
    assert np.allclose(g([50, 52], last), [10, 52])  # clamped to the range, inside the step
    assert g.clamped == 1 and g.capped == 0
    assert np.allclose(g([-10, 90], last), [7, 55])  # each joint moves at most its step
    assert g.capped == 1
    assert np.allclose(g([np.nan, 0], last), last) and np.allclose(g([1, 2, 3], last), last)
    assert g.rejected == 2


def test_plays_chunks_at_the_rate():
    robot, policy = FakeRobot(), FakePolicy(delays=(0.01,))
    report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=1.0, **QUIET), out=lambda *a: None)
    assert report["reason"] == "time limit" and robot.torque_off is True
    assert 40 <= report["ticks"] <= 51
    assert 18 <= np.median(report["periods"]) * 1e3 <= 25
    q = report["queue"]
    assert q["chunks"] >= 3 and q["errors"] == 0 and q["stale_chunks"] == 0
    # Asked again with 15 of 30 actions left, answered in about 10 ms: about 15 played per chunk.
    assert 12 <= np.median(report["played"]) <= 17
    # The plan moves shoulder_pan 0.5 degree per action, and each chunk starts from the arm's pose then: the targets
    # rise by 0.5 per tick with no jump at chunk boundaries (aligned merge), then go back to the start.
    targets = robot.targets()
    pan = targets[:, 0]
    steps = np.diff(pan[: report["sent"]])
    assert np.all(np.abs(steps) <= 0.5 + 1e-6)
    assert np.allclose(targets[-1], START) and report["holds"] <= 5
    assert "Stopped: time limit." in describe(report)


def test_late_reply_holds_the_arm():
    # Chunk of 30 at 50 Hz = 0.6 s; the third reply takes 0.9 s, so the queue runs dry and the arm holds.
    robot, policy = FakeRobot(), FakePolicy(delays=(0.01, 0.01, 0.9, 0.01))
    report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=2.0, **QUIET), out=lambda *a: None)
    assert report["holds"] >= 10 and robot.holds == report["holds"]
    assert report["longest_hold_s"] >= 0.2
    assert report["sent"] + report["holds"] == report["ticks"]


def test_targets_stay_inside_the_limits_and_the_speed_cap():
    # The plan rushes shoulder_pan at 10 degrees per action toward +300; the range ends at 20, the cap is 120 deg/s.
    upper = [20.0, 100, 100, 100, 100, 100]
    robot, policy = FakeRobot(upper=upper), FakePolicy(step=(10, 0, 0, 0, 0, 0))
    hz = 30
    report = run(robot, policy, mapping(), Settings(hz=hz, max_seconds=1.0, **QUIET), out=lambda *a: None)
    targets = robot.targets()
    assert targets[:, 0].max() <= 20 + 1e-9
    assert np.abs(np.diff(targets, axis=0)).max(axis=0)[0] <= 120 / hz + 1e-9
    assert report["capped"] > 0 and report["clamped"] > 0


def test_first_target_far_from_the_arm_is_refused():
    # elbow 40 degrees away from the arm at the first action: wrong units or calibration, nothing moves.
    robot, policy = FakeRobot(), FakePolicy(offset=(0, 0, 40, 0, 0, 0))
    with pytest.raises(Refused, match="elbow_flex by \\+40"):
        run(robot, policy, mapping(), Settings(hz=50, max_seconds=1.0, **QUIET), out=lambda *a: None)
    assert robot.sent == [] and robot.connected is False


def test_the_gripper_may_jump_at_the_start():
    # so101.MAX_START_JUMP leaves the gripper out: closing 60 % at once is no sign of wrong units.
    from playground_serve.run.so101 import MAX_START_JUMP

    policy = FakePolicy(step=(0, 0, 0, 0, 0, 0), offset=(0, 0, 0, 0, 0, 60))
    with pytest.raises(Refused, match="gripper"):
        run(FakeRobot(), policy, mapping(), Settings(hz=50, max_seconds=0.2, **QUIET), out=lambda *a: None)
    robot = FakeRobot()
    run(robot, policy, mapping(), Settings(hz=50, max_seconds=0.2, max_start_jump=MAX_START_JUMP, **QUIET),
        out=lambda *a: None)
    assert robot.targets()[:, 5].max() > START[5] + 10


def test_wrong_action_size_is_refused():
    class SevenDims(FakePolicy):
        def infer(self, obs):
            return {"actions": np.zeros((10, 7), np.float32)}

    robot = FakeRobot()
    with pytest.raises(Refused, match="shape"):
        run(robot, SevenDims(), mapping(), Settings(hz=50, max_seconds=1.0, **QUIET), out=lambda *a: None)
    assert robot.sent == []


def test_slow_move_to_the_first_target():
    # First action 20 degrees away on shoulder_pan: reached at a quarter of the cap (120 / 50 / 4 = 0.6 per tick).
    robot, policy = FakeRobot(), FakePolicy(step=(0, 0, 0, 0, 0, 0), offset=(20, 0, 0, 0, 0, 0))
    flags = Stop()
    threading.Timer(0.3, flags.stop.set).start()
    run(robot, policy, mapping(), Settings(hz=50, max_seconds=5.0, **QUIET), flags, out=lambda *a: None)
    first = robot.targets()[:5, 0]
    assert np.allclose(np.diff(np.append(START[0], first)), 0.6)


def test_stop_goes_back_to_the_start_then_torque_off():
    robot, policy = FakeRobot(), FakePolicy(step=(1, 0, 0, 0, 0, 0))
    flags = Stop()
    threading.Timer(0.4, flags.stop.set).start()
    lines = []
    report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=5.0, **QUIET), flags, out=lines.append)
    assert report["reason"] == "Ctrl-C" and report["seconds"] < 1.0
    targets = robot.targets()
    assert targets[:, 0].max() > START[0] + 5  # it moved
    assert np.allclose(targets[-1], START)
    assert np.abs(np.diff(targets[-10:], axis=0)).max() <= 120 / 50 / 4 + 1e-9  # back at a quarter of the cap
    assert robot.torque_off is True and "Going back to the start pose." in lines


def test_second_ctrl_c_turns_the_torque_off_at_once():
    robot, policy = FakeRobot(), FakePolicy(step=(1, 0, 0, 0, 0, 0))
    flags = Stop()

    def press_twice():
        time.sleep(0.4)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)

    lines = []
    before = signal.getsignal(signal.SIGINT)
    threading.Thread(target=press_twice, daemon=True).start()
    with ctrl_c(flags, out=lines.append, first="first", second="second"):
        report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=5.0, **QUIET), flags, out=lines.append)
    assert signal.getsignal(signal.SIGINT) is before
    assert flags.abort.is_set() and report["reason"] == "Ctrl-C" and lines.count("\nfirst") == 1
    assert robot.torque_off is True
    assert not np.allclose(robot.state, START)  # the way back was cut short


def test_dry_run_sends_no_motor_command():
    robot, policy = FakeRobot(), FakePolicy()
    lines = []
    report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=0.7, dry_run=True, **QUIET),
                 out=lines.append)
    assert robot.sent == [] and robot.holds == 0 and robot.torque_off is False
    assert report["sent"] > 0 and any(line.startswith("chunk ") for line in lines)
    assert "nothing sent" in describe(report)


def test_stall_ends_the_run():
    # The connection is lost after the first chunk of the loop: the queue runs dry, the arm holds, the run ends.
    robot, policy = FakeRobot(), FakePolicy(fail_after=2)
    report = run(robot, policy, mapping(), Settings(hz=50, max_seconds=5.0, stall_s=0.3, **QUIET),
                 out=lambda *a: None)
    assert report["reason"].startswith("no action for 0.3 s") and report["queue"]["errors"] >= 1
    assert report["holds"] > 0 and np.allclose(robot.state, START) and robot.torque_off is True


def test_an_error_still_brings_the_arm_back():
    class Breaks(FakeRobot):
        def read(self):
            if self.reads > 20:
                raise TimeoutError("camera frame too old")
            return super().read()

    robot = Breaks()
    with pytest.raises(TimeoutError):
        run(robot, FakePolicy(), mapping(), Settings(hz=50, max_seconds=5.0, **QUIET), out=lambda *a: None)
    assert np.allclose(robot.state, START) and robot.torque_off is True


def test_the_way_back_after_an_error_is_slow():
    # A camera fails in the middle of the run, far from the start: every step back is at the start speed, from where
    # the arm was sent last (not from an older target).
    class Breaks(FakeRobot):
        def read(self):
            if self.reads > 60:
                raise TimeoutError("camera frame too old")
            return super().read()

    robot = Breaks()
    with pytest.raises(TimeoutError):
        run(robot, FakePolicy(step=(1, 0, 0, 0, 0, 0)), mapping(), Settings(hz=50, max_seconds=5.0, **QUIET),
            out=lambda *a: None)
    targets = robot.targets()
    assert targets[:, 0].max() > START[0] + 20
    assert np.abs(np.diff(targets, axis=0)).max() <= 120 / 50 + 1e-9  # never more than the play cap per tick
    assert np.allclose(targets[-1], START) and robot.torque_off is True


def test_non_finite_first_target_is_refused():
    class NaNGripper(FakePolicy):
        def infer(self, obs):
            reply = super().infer(obs)
            reply["actions"][:, 5] = np.nan
            return reply

    robot = FakeRobot()
    with pytest.raises(Refused, match="not finite"):
        run(robot, NaNGripper(offset=(80, 0, 0, 0, 0, 0)), mapping(), Settings(hz=50, max_seconds=1.0, **QUIET),
            out=lambda *a: None)
    assert robot.sent == []


def test_a_chunk_of_another_size_is_held_not_played():
    class Widens(FakePolicy):
        def infer(self, obs):
            reply = super().infer(obs)
            if len(self.observations) > 2:
                reply["actions"] = np.zeros((30, 7), np.float32)
            return reply

    robot = FakeRobot()
    report = run(robot, Widens(), mapping(), Settings(hz=50, max_seconds=1.5, **QUIET), out=lambda *a: None)
    assert report["rejected"] > 0 and report["reason"] == "time limit" and robot.torque_off is True
    assert np.abs(np.diff(robot.targets(), axis=0)).max() <= 120 / 50 + 1e-9


def test_observation_is_in_model_units():
    robot, policy = FakeRobot(), FakePolicy()
    run(robot, policy, mapping("pick it up"), Settings(hz=50, max_seconds=0.2, dry_run=True, **QUIET),
        out=lambda *a: None)
    obs = policy.observations[0]
    assert set(obs) == {"top_cam", "side_cam", "state", "instruction"} and obs["instruction"] == "pick it up"
    assert obs["state"].dtype == np.float32
    assert np.allclose(obs["state"], [0, 150, 150, 50, 0, 10])  # 90 - (-60), -(60) ... see so101.SIGN, OFFSET


def test_ctrl_c_outside_the_main_thread_is_a_no_op():
    flags, out = Stop(), []

    def body():
        with ctrl_c(flags):
            out.append(signal.getsignal(signal.SIGINT))

    before = signal.getsignal(signal.SIGINT)
    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert out == [before]


def test_ramp_stops_when_asked():
    robot, until = FakeRobot(), threading.Event()
    end = loop.ramp(robot, START.copy(), START + 10, np.full(6, 1.0), 0.0, until, sleep=lambda s: None)
    assert np.allclose(end, START + 10) and len(robot.sent) == 10
    until.set()
    assert np.allclose(loop.ramp(robot, end, START, np.full(6, 1.0), 0.0, until), end)
