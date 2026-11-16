"""`pgs run ROBOT`: drive a robot with a model already served by `pgs up`, on the robot side, with its safety limits.

    pgs up molmoact2_so101
    pgs run so101 --model molmoact2_so101 --instruction "put the cube in the bowl"

It reads the cameras and joints, sends them in the model's keys and units, and plays the returned action chunks on the
motors at the model's action rate (loop.py: limits, hold, stop). It starts no GPU: without a running worker it prints
the `pgs up` command and stops. Robots: so101 (so101.py, LeRobot's driver; sim.py with --sim). docs/RUN.md is the
guide. Only the parser is imported by `pgs`; the rest when the command runs.
"""

from __future__ import annotations

import argparse
import re
import sys

ROBOTS = {"so101": "molmoact2_so101"}  # robot -> default model
RUNPOD_URL = re.compile(r"wss?://[A-Za-z0-9]+\.api\.runpod\.ai(?:[:/]|$)")


def add_parser(sub, common: argparse.ArgumentParser) -> None:
    s = sub.add_parser("run", parents=[common], help="drive a robot (so101) with a running model; starts no GPU",
                       description="Drive a robot with a model that `pgs up` serves. The robot moves: read "
                       "docs/RUN.md first. Ctrl-C stops it and brings the arm back to its start pose; a second "
                       "Ctrl-C turns the torque off at once.")
    s.add_argument("robot", choices=sorted(ROBOTS), help="the robot: so101 (SO-101 arm through LeRobot)")
    s.add_argument("--model", help="model name or wss:// URL (default for so101: molmoact2_so101)")
    s.add_argument("--instruction", required=True, help='the task in words, e.g. "put the cube in the bowl"')
    s.add_argument("--port", help="the arm's serial port (default: the only USB serial device; `lerobot-find-port` "
                   "names it)")
    s.add_argument("--cameras", default="top=0,side=1", metavar="ROLE=DEV,...",
                   help="camera role to OpenCV device index or path (default top=0,side=1)")
    s.add_argument("--hz", type=float, help="control rate in Hz (default: the model's action rate, 30 for "
                   "molmoact2_so101)")
    s.add_argument("--dry-run", action="store_true", help="read the arm and cameras and print the planned targets; "
                   "send no motor command")
    s.add_argument("--max-seconds", type=float, default=60.0, help="stop after this long (default 60)")
    s.add_argument("--yes", action="store_true", help="start without the prompt")
    s.add_argument("--sim", action="store_true", help="a simulated SO-101 (MuJoCo) instead of the arm")
    s.add_argument("--id", help="LeRobot id of the arm's calibration file (default: the only one)")


def _endpoint_up(model: str, key: str, name: str | None = None) -> None:
    """Raise LauncherError with the `pgs up` command unless a worker of the model's endpoint is running. `model` is a
    model name or the endpoint's wss:// URL; `name` the model's name for the hint."""
    from ..launcher import LauncherError
    from ..resolve import EndpointNotFound, status

    up = f"Start one first: `pgs up {name or model}` (pgs run starts no GPU)."
    try:
        found = status(model, api_key=key)
    except (EndpointNotFound, PermissionError, ConnectionError, ValueError, OSError) as e:
        raise LauncherError(f"cannot check the endpoint of {model}: {e}. {up}") from None
    if found.state == "missing":
        raise LauncherError(f"no endpoint for {model} on this Runpod account. {up}")
    if found.state != "warm":
        raise LauncherError(f"no worker of {model} is running. {up} Then run this within its idle time (5 minutes "
                            "unless `pgs up --idle` set another).")


def _confirm(text: str, out) -> None:
    from ..launcher import LauncherError

    out(text)
    if not sys.stdin.isatty():
        raise LauncherError("the robot will move and this is not a terminal: pass --yes to start without the prompt")
    try:
        answer = input("Start? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        raise LauncherError("not started; nothing moved")


def main(args, key: str, out=print) -> int:
    """`pgs run`. Errors that a user can act on are LauncherError (one sentence, exit code 1)."""
    from ..launcher import LauncherError
    from ..resolve import WS_URL, connect
    from . import loop, so101

    model = args.model or ROBOTS[args.robot]
    url = bool(WS_URL.match(model))
    if not url and model not in so101.MODELS:
        raise LauncherError(f"pgs run {args.robot} drives {', '.join(so101.MODELS)}, not {model}")
    if args.hz is not None and not 1 <= args.hz <= 100:
        raise LauncherError("--hz must be between 1 and 100")
    if args.max_seconds <= 0:
        raise LauncherError("--max-seconds must be positive")
    try:
        if args.sim:
            from .sim import SimSO101

            robot, what = SimSO101(out=out), "the simulated SO-101 (MuJoCo)"
        else:
            cameras = so101.parse_cameras(args.cameras)
            so101.import_lerobot()
            port = args.port or so101.find_port()
            robot = so101.SO101(port, cameras, args.id, args.dry_run)
            what = f"the SO-101 on {port} (calibration {robot.calibration})"
    except (so101.RobotError, ValueError) as e:
        raise LauncherError(str(e)) from None
    if not url or RUNPOD_URL.match(model):  # another ws:// server (yours, a test) is used as it is
        _endpoint_up(model, key, None if not url else ROBOTS[args.robot])
    if not (args.sim or args.dry_run or args.yes):
        _confirm(f"{what[0].upper()}{what[1:]} will MOVE, driven by {model} ({args.instruction!r}), for up to "
                 f"{args.max_seconds:g} s. Limits: the calibrated joint ranges, at most "
                 f"{so101.MAX_SPEED[0]:g} degrees per second per joint, a slow move to the first target; it holds "
                 "still when no action arrives. Keep people and things out of its reach. Ctrl-C stops it and brings "
                 "it back to its start pose; a second Ctrl-C turns the torque off at once (the arm goes limp).", out)
    out(f"Connecting to {model}...")
    with connect(model, api_key=key or None, wait=30.0, recv_timeout=10.0) as policy:
        name = policy.metadata.get("model") or model
        if name not in so101.MODELS:
            raise LauncherError(f"the endpoint serves {name}; pgs run {args.robot} drives {', '.join(so101.MODELS)}")
        mapping = so101.MODELS[name](robot.gripper_span_ticks, args.instruction)
        hz = args.hz or float(policy.metadata.get("action_hz") or 30.0)
        if not 1 <= hz <= 100:
            raise LauncherError(f"the endpoint names an action rate of {hz:g} Hz; pass --hz (1 to 100)")
        mode = "dry run: nothing is sent to the motors" if args.dry_run else "Ctrl-C stops"
        out(f"Connected to {name} ({policy.metadata.get('playground_serve', {}).get('path', 'stock')} path); "
            f"{what}; {hz:g} Hz for up to {args.max_seconds:g} s ({mode}).")
        flags = loop.Stop()
        first = ("Stopping (Ctrl-C again: stop at once)." if args.dry_run or args.sim else
                 "Stopping: the arm goes back to its start pose (Ctrl-C again: torque off at once).")
        settings = loop.Settings(hz=hz, max_seconds=args.max_seconds, dry_run=args.dry_run,
                                 max_start_jump=so101.MAX_START_JUMP)
        # From here Ctrl-C only sets flags, so that it cannot leave the arm connected: run() always disconnects it.
        with loop.ctrl_c(flags, out, first=first, second="Stopping at once; torque off."):
            try:
                robot.connect()
            except so101.RobotError as e:
                raise LauncherError(str(e)) from None
            except Exception as e:  # noqa: BLE001  (serial port, cameras, driver: one sentence)
                raise LauncherError(f"could not connect {what}: {type(e).__name__}: {e}") from None
            try:
                report = loop.run(robot, policy, mapping, settings, flags, out=out)
            except loop.Refused as e:
                raise LauncherError(str(e)) from None
    out(loop.describe(report))
    return 0
