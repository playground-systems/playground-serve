"""The control loop of `pgs run`: observe, ask the policy without holding up the motion, play its action chunks at the
model's action rate through ActionQueue, and keep every target inside the robot's limits.

    robot.connect()
    with ctrl_c(flags := Stop()):
        report = run(robot, policy, mapping, Settings(hz=30, max_seconds=60), flags)
    print(describe(report))

Units: the loop works in the robot's own units (for the SO-101 LeRobot's degrees and gripper percent); `mapping` turns
a reading into the model's observation (`observation(state, images)`) and the model's actions into robot targets
(`to_robot(actions)`), so101.MolmoAct2SO101.

Safety, on the robot side and always on:
- Every target is clamped to the robot's calibrated range (robot.lower, robot.upper) and moves each joint at most
  robot.max_speed / hz from the previous target (as if hz were at least 10, so a low rate does not mean big steps).
- Before any motion, the policy's first target must lie within Settings.max_start_jump of the arm's pose (per joint;
  the SO-101 leaves its gripper out); a larger jump means wrong units or a wrong calibration, and nothing moves
  (Refused). The arm then goes to that target at
  Settings.start_speed times the speed cap, and the chunks play from there.
- When no action is queued (a late reply, a lost connection) the robot holds its last target; after Settings.stall_s
  seconds without one the run ends.
- At the end (time limit, Ctrl-C, stall, an error) the arm goes back to its start pose at the start speed and the
  torque goes off. A second Ctrl-C skips the way back: torque off at once (the arm goes limp and may sag).
- Dry run: nothing is sent to the motors and their torque is left as it was; each chunk's targets are printed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import signal
import threading
import time

import numpy as np

from ..client.action_queue import ActionQueue


class Refused(RuntimeError):
    """The run ended before any motion; the message says why."""


class Robot:
    """What `pgs run` drives (so101.SO101, sim.SimSO101). Positions and targets are float arrays in the robot's own
    units, in `joints` order; every target the loop sends lies in [lower, upper] and moves at most max_speed (units per
    second) from the previous one."""

    joints: tuple = ()
    lower: np.ndarray
    upper: np.ndarray
    max_speed: np.ndarray

    def connect(self) -> None:
        raise NotImplementedError

    def read(self) -> tuple:
        """(joint positions, {camera role: H x W x 3 uint8 RGB frame})."""
        raise NotImplementedError

    def send(self, target) -> None:
        """Joint position targets."""
        raise NotImplementedError

    def hold(self) -> None:
        """Keep the last target sent."""
        raise NotImplementedError

    def disconnect(self, torque_off: bool = True) -> None:
        raise NotImplementedError


@dataclasses.dataclass
class Settings:
    hz: float = 30.0  # control rate: one target per tick (the model's action rate plays the chunk at its own speed)
    max_seconds: float = 60.0
    dry_run: bool = False
    max_start_jump: float | tuple = 30.0  # robot units (degrees; gripper percent), one value or one per joint
    start_speed: float = 0.25  # fraction of the speed cap for the moves to the first target and back to the start
    stall_s: float = 10.0  # end the run after this long without an action
    threshold: float = 0.5  # ActionQueue: ask for the next chunk when less than this fraction of a chunk is left
    status_s: float = 5.0  # a progress line every this many seconds (0: none)


class Stop:
    """`stop`: end the run and bring the arm back (first Ctrl-C). `abort`: torque off at once (second Ctrl-C)."""

    def __init__(self) -> None:
        self.stop, self.abort = threading.Event(), threading.Event()


@contextlib.contextmanager
def ctrl_c(flags: Stop, out=print, first: str = "Stopping.", second: str = "Stopping at once."):
    """While inside: Ctrl-C sets flags.stop, a second one flags.abort, a third raises KeyboardInterrupt."""
    if threading.current_thread() is not threading.main_thread():
        yield flags
        return
    previous = signal.getsignal(signal.SIGINT)

    def say(text: str) -> None:
        try:
            out(f"\n{text}")
        except RuntimeError:  # the signal came in the middle of a print ("reentrant call"): the flag is what counts
            pass

    def handler(signum, frame):
        if not flags.stop.is_set():
            flags.stop.set()
            say(first)
        else:
            flags.abort.set()
            signal.signal(signal.SIGINT, signal.default_int_handler)
            say(second)

    signal.signal(signal.SIGINT, handler)
    try:
        yield flags
    finally:
        signal.signal(signal.SIGINT, previous)


class Guard:
    """Each target the loop sends: inside [lower, upper], and at most `step` per joint from the previous target. A
    target of the wrong shape or with a non-finite value is replaced by the previous target (hold)."""

    def __init__(self, lower, upper, step) -> None:
        self.lower, self.upper, self.step = (np.asarray(x, dtype=np.float64) for x in (lower, upper, step))
        self.clamped = self.capped = self.rejected = 0  # targets changed by each rule

    def clip(self, target) -> np.ndarray:
        return np.clip(np.asarray(target, dtype=np.float64), self.lower, self.upper)

    def __call__(self, target, last) -> np.ndarray:
        target = None if target is None else np.asarray(target, dtype=np.float64)
        if target is None or target.shape != last.shape or not np.isfinite(target).all():
            self.rejected += 1
            return last.copy()
        inside = self.clip(target)
        self.clamped += bool((inside != target).any())
        move = np.clip(inside - last, -self.step, self.step)
        self.capped += bool((move != inside - last).any())
        return last + move


class Timed:
    """The policy, keeping each reply's timings and its actions (ActionQueue calls it from its own thread)."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls: list = []  # (round trip ms, server infer_ms, model_ms) per reply
        self.last = None

    @property
    def timing(self) -> dict:
        return self.policy.timing

    def infer(self, obs: dict) -> dict:
        reply = self.policy.infer(obs)
        timing, stages = self.policy.timing, reply.get("policy_timing") or {}
        self.calls.append((timing.get("rtt_ms"), timing.get("infer_ms"), stages.get("model_ms")))
        self.last = np.asarray(reply["actions"])
        return reply


class Sender:
    """robot.send, keeping the last target sent: the arm's goal, where the way back starts whatever went wrong."""

    def __init__(self, robot, last) -> None:
        self.robot, self.last = robot, np.asarray(last, dtype=np.float64).copy()

    def send(self, target) -> None:
        self.robot.send(target)
        self.last = np.asarray(target, dtype=np.float64).copy()


def ramp(robot, last, goal, step, tick: float, until: threading.Event, sleep=time.sleep) -> np.ndarray:
    """Targets from `last` to `goal`, at most `step` apart, one per tick, until there or until `until` is set.
    `robot` is anything with send()."""
    while not until.is_set():
        move = np.clip(goal - last, -step, step)
        if not move.any():
            break
        last = last + move
        robot.send(last)
        sleep(tick)
    return last


def _fmt(values) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in values) + "]"


def run(robot, policy, mapping, settings: Settings, flags: Stop | None = None, out=print, clock=time.monotonic,
        sleep=time.sleep) -> dict:
    """Drive a connected robot until the time limit, flags.stop or a stall, and always end with robot.disconnect()
    (torque off except in a dry run). Raises Refused when the first target is too far. Returns the report
    (`describe` prints it)."""
    s, flags = settings, flags or Stop()
    tick = 1.0 / s.hz
    max_speed = np.asarray(robot.max_speed, dtype=np.float64)
    guard = Guard(robot.lower, robot.upper, max_speed / max(s.hz, 10.0))
    slow = guard.step * s.start_speed
    timed = Timed(policy)
    report: dict = {"reason": "", "dry_run": s.dry_run, "hz": s.hz}
    queue, start, sender = None, None, None
    try:
        start, images = robot.read()
        sender = Sender(robot, start)
        last = start.copy()
        out(f"Joints: {', '.join(robot.joints)}. Arm at {_fmt(start)}.")
        raw = np.asarray(timed.infer(mapping.observation(start, images))["actions"])
        if raw.ndim != 2 or raw.shape[1] != len(robot.joints):
            raise Refused(f"the policy answered actions of shape {raw.shape}, not (horizon, {len(robot.joints)}): "
                          "is this model made for this robot?")
        first = mapping.to_robot(raw)
        if not np.isfinite(first).all():
            raise Refused("the policy's first chunk holds values that are not finite numbers: nothing moved")
        gap = first[0] - start
        limit = np.broadcast_to(np.asarray(s.max_start_jump, dtype=np.float64), gap.shape)
        j = int(np.argmax(np.abs(gap) - limit))
        out(f"First chunk: {len(first)} actions in {timed.calls[0][0]:.0f} ms; first target {_fmt(first[0])}.")
        if not abs(gap[j]) <= limit[j]:
            raise Refused(f"the policy's first target moves {robot.joints[j]} by {gap[j]:+.1f} from where the arm is "
                          f"(at most {limit[j]:g} allowed): the units or the calibration are likely wrong, so "
                          "nothing moved. Check the readings with --dry-run (docs/RUN.md, Units).")
        if not s.dry_run and not flags.stop.is_set():
            last = ramp(sender, last, guard.clip(first[0]), slow, tick, flags.stop, sleep)
        if not flags.stop.is_set():
            queue = ActionQueue(timed, threshold=s.threshold, clock=clock)
            _play(robot, sender, queue, timed, mapping, guard, last, s, flags, report, out, clock, sleep)
        else:
            report["reason"] = "Ctrl-C"
    finally:
        try:
            if queue is not None:
                queue.close(timeout=1.0)
                report["queue"] = queue.stats()
            report.update(clamped=guard.clamped, capped=guard.capped, rejected=guard.rejected,
                          calls=list(timed.calls))
            if sender is not None and not s.dry_run and not flags.abort.is_set() and \
                    not np.array_equal(sender.last, start):
                out("Going back to the start pose.")
                ramp(sender, sender.last, guard.clip(start), slow, tick, flags.abort, sleep)
                if not flags.abort.is_set():
                    sleep(0.5)  # let the arm settle before the torque goes off
        except Exception as e:  # noqa: BLE001  (the torque must still go off)
            out(f"Could not bring the arm back ({type(e).__name__}: {e}).")
        finally:
            robot.disconnect(torque_off=not s.dry_run)
    return report


def _play(robot, sender, queue, timed, mapping, guard, last, s, flags, report, out, clock, sleep) -> None:
    tick = 1.0 / s.hz
    t_start = base = clock()
    prev, i, periods, late = None, 0, [], 0
    sent = holds = 0
    hold_since, longest_hold = None, 0.0
    seen, at_chunk, played = queue.chunks, queue.executed, []
    next_status = s.status_s
    report["reason"] = "time limit"
    while True:
        now = clock()
        if prev is not None:
            periods.append(now - prev)
        prev = now
        if flags.stop.is_set():
            report["reason"] = "Ctrl-C"
            break
        if now - t_start >= s.max_seconds:
            break
        state, images = robot.read()
        queue.put_observation(mapping.observation(state, images))
        action = queue.get_action()
        if action is None:
            hold_since = now if hold_since is None else hold_since
            longest_hold = max(longest_hold, now - hold_since)
            holds += 1
            if not s.dry_run:
                robot.hold()
            if now - hold_since >= s.stall_s:
                report["reason"] = f"no action for {s.stall_s:g} s (policy not answering)"
                break
        else:
            hold_since = None
            try:
                wanted = mapping.to_robot(action)
            except ValueError:  # an action of another size: the guard rejects it and the arm holds
                wanted = None
            last = guard(wanted, last)
            if not s.dry_run:
                sender.send(last)
            sent += 1
        if queue.chunks != seen:
            if seen > 0:
                played.append(queue.executed - at_chunk)
            seen, at_chunk = queue.chunks, queue.executed
            if s.dry_run and timed.last is not None:
                chunk = mapping.to_robot(timed.last)
                gap = chunk[0] - state
                j = int(np.argmax(np.abs(gap)))
                rtt, _, model_ms = timed.calls[-1]
                out(f"chunk {seen}: reply {rtt:.0f} ms" + (f" (model {model_ms:.0f} ms)" if model_ms else "")
                    + f", arm {_fmt(state)}, targets {_fmt(chunk[0])} to {_fmt(chunk[-1])}, largest move from the "
                    f"arm {robot.joints[j]} {gap[j]:+.1f}")
        if s.status_s and now - t_start >= next_status and not s.dry_run:
            next_status += s.status_s
            out(f"{now - t_start:5.1f} s: {sent} targets sent, {holds} ticks held, {queue.chunks} chunks")
        i += 1
        wait = base + i * tick - clock()
        if wait > 0:
            sleep(wait)
        else:
            late += 1
            if wait < -tick:  # more than a tick behind: keep the rate from here instead of catching up
                base -= wait
    report.update(seconds=clock() - t_start, ticks=i, periods=periods, late=late, sent=sent, holds=holds,
                  longest_hold_s=longest_hold, played=played)


def _pct(values, *qs):
    values = [v for v in values if v is not None]
    return tuple(float(np.percentile(values, q)) for q in qs) if values else None


def describe(report: dict) -> str:
    """The report of `run` as a few lines."""
    lines = [f"Stopped: {report['reason']}." if report.get("reason") else "Stopped."]
    if "ticks" in report:
        p = _pct([x * 1e3 for x in report["periods"]], 50, 99) or (0.0, 0.0)
        verb = "planned (dry run, nothing sent)" if report["dry_run"] else "sent"
        lines.append(f"Ran {report['seconds']:.1f} s: {report['ticks']} ticks at {report['hz']:g} Hz (period p50 "
                     f"{p[0]:.1f} ms, p99 {p[1]:.1f} ms, {report['late']} late); {report['sent']} targets {verb}, "
                     f"{report['holds']} ticks held (longest hold {report['longest_hold_s']:.2f} s).")
    calls = report.get("calls") or []
    if calls:
        later = calls[1:] or calls
        rtt, infer, model = (_pct([c[k] for c in later], 50, 95) for k in range(3))
        text = f"Policy: {len(calls)} replies; round trip p50 {rtt[0]:.0f} ms, p95 {rtt[1]:.0f} ms"
        text += f"; server time p50 {infer[0]:.0f} ms" if infer else ""
        text += f", model time p50 {model[0]:.0f} ms" if model else ""
        lines.append(text + f" (first reply {calls[0][0]:.0f} ms, not counted).")
    q = report.get("queue") or {}
    if q.get("chunks"):
        played = report.get("played") or []
        pl = (f"actions played per chunk before the next one arrived: median {np.median(played):.0f} (min "
              f"{min(played)}, max {max(played)}); " if played else "")
        lines.append(f"Chunks: {q['chunks']}; {pl}{q['dropped']} actions skipped as already past, "
                     f"{q['stale_chunks']} stale chunks, {q['errors']} errors"
                     + (f" (last: {q['last_error']})" if q.get("last_error") else "") + ".")
    lines.append(f"Limits: {report.get('clamped', 0)} targets clamped to the joint range, {report.get('capped', 0)} "
                 f"slowed by the speed cap, {report.get('rejected', 0)} rejected (not finite).")
    return "\n".join(lines)
