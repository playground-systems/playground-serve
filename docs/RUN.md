# `pgs run`: drive a robot arm

`pgs run` connects a robot to a model that `pgs up` serves and makes it move. It reads the cameras and joints, sends
them in the keys and units the model was trained on, and plays the returned action chunks on the motors at the model's
action rate, inside limits that the robot side enforces. The first robot is the SO-101 arm, through LeRobot's driver,
with Ai2's MolmoAct2 for the SO-100/101 (`molmoact2_so101`).

```bash
pgs up molmoact2_so101
pgs run so101 --model molmoact2_so101 --instruction "put the cube in the bowl"
```

**Status.** Ran end to end in simulation (`--sim`) with the real model on an RTX 4090 and in tests with stand-ins for
the arm and the model. It has not run on a real SO-101. The unit mapping below comes from the model's files, LeRobot's
source and a reference deployment, not from a measurement on an arm. The first run on a real arm needs a person next to
it, with the dry run first (Setup, step 6).

`pgs run` starts no GPU. If no worker of the model runs (named, or as a `wss://...api.runpod.ai` URL), it prints the
`pgs up` command and stops; start it, then run `pgs run` within the worker's idle time (5 minutes unless `pgs up
--idle` set another). A connected `pgs run` keeps the worker running; GPU billing stops 5 minutes after it ends.

## Install

LeRobot 0.5 and later need Python 3.12, so `pgs run so101` does too:

```bash
pip install "playground-serve[so101] @ git+https://github.com/playground-systems/playground-serve"   # Python 3.12
```

The `so101` extra installs LeRobot 0.7.0 with its Feetech motor support (the version `pgs run` was written against;
it also installs PyTorch, which LeRobot imports). Without LeRobot, `pgs run so101` says how to install it. The
simulator needs only MuJoCo: `pip install "playground-serve[sim] @ git+..."` on Python 3.10 or later.

## Setup of a real SO-101

1. Build the follower arm and set its motor ids as LeRobot's SO-101 guide says
   (https://huggingface.co/docs/lerobot/so101, `lerobot-setup-motors --robot.type=so101_follower --robot.port=PORT`).
2. Find the arm's serial port with `lerobot-find-port` (macOS: `/dev/tty.usbmodem...`, Linux: `/dev/ttyACM0`).
   `pgs run` takes it with `--port`, or finds it when it is the only USB serial device.
3. Calibrate the arm once with LeRobot 0.7.0:

   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=PORT --robot.id=my_so101
   ```

   Put every joint in the middle of its range, press ENTER, then move each joint through its whole range, to the
   mechanical stops, and press ENTER. The units below depend on that sweep (Units, assumption 1). LeRobot saves the file
   to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_so101.json` (`HF_LEROBOT_CALIBRATION` moves
   it). `pgs run` uses the only file there, or the one `--id my_so101` names, and refuses to start if the motors hold
   other values (run `lerobot-calibrate` again and press ENTER to write the file to them).
4. Place two fixed cameras, 640 x 480 at 30 frames per second, as the model card's sample: `top` above the table
   looking down at the workspace, the arm at the image's top edge; `side` at table height looking across the
   workspace, the arm at one side. The model card says the order of the views does not matter. Do not use a wrist
   camera: the views it was trained on are third-person views, and the reference deployment found its wrist camera
   out of distribution. `lerobot-find-cameras opencv` lists the device indexes; pass them as `--cameras top=0,side=1`.
5. Start the model: `pgs up molmoact2_so101` (about 2 minutes the first time; MODELS.md).
6. Dry run, with the arm at rest:

   ```bash
   pgs run so101 --instruction "put the cube in the bowl" --dry-run
   ```

   It reads the joints and cameras, sends no command to the motors (it does not even turn their torque on or off), and
   prints, for each chunk, the arm's reading and the chunk's first and last targets. Check:
   - At rest (folded), shoulder_lift reads about -95 to -100 and elbow_flex about +90 to +97 (LeRobot's degrees),
     and the gripper about 0 when closed. The model card's sample state, at rest, reads (-0.5, -99.1, 91.4, 60.6,
     -3.6, about 1) in these units.
   - The printed targets stay within a few degrees of the arm ("largest move from the arm"). A target tens of
     degrees away means the units or the calibration are wrong; `pgs run` refuses to move when the first one is
     more than 30 degrees away.
   - With the torque off (it is after calibration), move the arm by hand: the `arm [...]` readings follow, and the
     targets follow the arm.
7. First motion, with a hand on the power switch and nothing fragile in reach:

   ```bash
   pgs run so101 --instruction "put the cube in the bowl" --max-seconds 20
   ```

   It states that the arm will move and waits for `y` (`--yes` skips this). Ctrl-C stops the policy and brings the arm
   back to where it started; a second Ctrl-C turns the torque off at once.

## Units

The model, allenai/MolmoAct2-SO100_101, takes the state and returns absolute joint targets in the units of the data
it was trained on: 1220 community SO-100/101 LeRobot datasets, recorded with LeRobot's calibration from before its
PR #777 (2025). LeRobot 0.7.0, which `pgs run` drives the arm with, uses the newer calibration. `pgs run` converts
between them (`run/so101.py`, `MolmoAct2SO101`):

| Joint, in the model's order | LeRobot 0.7.0, `use_degrees=True` | MolmoAct2-SO100_101 | Model from LeRobot |
|---|---|---|---|
| shoulder_pan | degrees, 0 in the middle of the calibrated range | degrees, 0 with the arm stretched out horizontally | same |
| shoulder_lift | degrees, as above | degrees, as above, opposite direction | 90 - x |
| elbow_flex | degrees, as above | degrees, as above | x + 90 |
| wrist_flex | degrees, as above | degrees, as above | same |
| wrist_roll | degrees, as above | degrees, as above | same |
| gripper | percent of the calibrated range, 0 closed, 100 open | percent of a quarter turn from closed | x * (range_max - range_min) / 1024 |

LeRobot's degrees are (raw - mid) x 360 / 4095 with mid the middle of the recorded range (raw: the STS3215's encoder,
4096 positions a turn). Actions go back the inverse way. The camera frames go as `top_cam` and `side_cam` at their own
size (JPEG, quality 90); the model resizes each to 378 x 378. The task goes as `instruction`.

Sources:

- Model: `norm_stats.json` of allenai/MolmoAct2-SO100_101 at revision 152569fe (joint names and order, absolute
  joint pose, 30 actions per chunk; the middle 98 % of shoulder_lift is 44 to 185 degrees and of elbow_flex 38 to 173,
  which only the old calibration gives) and its model card (raw robot state in, actions in robot scale, camera order
  free, two 640 x 480 RealSense samples). The training data list: allenai/MolmoAct2-SO100_101-Dataset.
- Old calibration: LeRobot at commit 42bf1e8b, `lerobot/common/robot_devices/robots/feetech_calibration.py`,
  `run_arm_manual_calibration`: the zero is "a straight horizontal position with gripper upwards and closed", the
  rotated position a quarter turn from it, the gripper linear from 0 % (closed) to 100 % (rotated).
- New calibration: LeRobot 0.7.0, `motors/motors_bus.py` (`_normalize`, `_unnormalize`) and
  `robots/so_follower/` (`use_degrees` is True by default; the gripper is always 0 to 100). The SO-ARM100 repository's
  `Simulation/SO101/README.md`: in LeRobot the gripper's 0 is fully closed and 100 fully open.
- Conversion: LeRobot's backward-compatibility page (`docs/source/backwardcomp.mdx`, "Hardware API redesign":
  shoulder_lift = -(old - 90), elbow_flex = old - 90), and the open SO-101 deployment the MolmoAct2 README links
  (github.com/irenegracekp/molmoact2-so101 at 9819f6a: signs 1, -1, 1, 1, 1, 1 and offsets 0, 90, 90, 0, 0, 0 on
  LeRobot 0.5.1 with `use_degrees=True`, whose normalization code is the same as 0.7.0's).
- Cross-check: TheRobotStudio's MuJoCo files of the arm in both calibrations (SO-ARM100 at a758567,
  `so101_old_calib.xml` and `so101_new_calib.xml`) give the same gripper position, within 2 mm over 20 random poses,
  only under old = new + (0, -90, +90, 0, 0) in their joint directions. LeRobot feeds its degrees unchanged to the
  new-calibration model (`lerobot/model/kinematics.py`); with the training data's positive shoulder_lift, that makes the
  old LeRobot shoulder_lift the opposite of the file's, which is the formula above. The model card's sample state,
  converted, puts the simulated arm in the folded pose of the card's photo.

Confirmed from those sources: the joint order, degrees, the shoulder_lift reversal and the two 90-degree offsets
(three independent sources agree), the gripper's direction. Assumed, to be checked on a real arm:

1. Each joint's zero. LeRobot puts 0 in the middle of the range you swept at calibration. The 90-degree offsets hold
   for a sweep from stop to stop; a sweep that stops d degrees short at one end shifts that joint's reading by d / 2.
2. The gripper's scale. `pgs run` maps the old quarter turn to LeRobot's percent through the calibrated range. The
   reference deployment passes LeRobot's percent unchanged instead; with a gripper range of 133 degrees, `pgs run`
   opens the gripper to about two thirds of what that deployment commands for the same model output.
3. The arm. Most of the training data comes from SO-100 arms, and MolmoAct2's README says its out-of-the-box
   performance was verified on the SO-100 only.
4. The cameras. The training datasets have many camera setups; the placement above follows the model card's sample.

## Safety

On the robot side, always on (`run/loop.py`):

- Every target stays inside the joint's calibrated range, 2 degrees in from each end (the gripper: 0 to 100 %).
- Each target moves each joint at most 120 degrees per second (the gripper 150 % per second) from the previous one:
  4 degrees per tick at 30 Hz. At a rate below 10 Hz the step stays at the 10 Hz size. The model card's sample chunk
  moves shoulder_lift by up to 2 degrees a step.
- Before the first motion, the model's first target must be within 30 degrees of every joint's reading (the gripper
  may open or close at once); else nothing moves and the message names the joint (wrong units or calibration). The
  arm then goes to that target at a quarter of the speed cap (30 degrees per second), and the chunks play from there.
- When no action is queued, because a reply is late or the connection dropped, the arm holds its last target. After 10
  seconds without an action the run ends.
- The run ends after `--max-seconds` (default 60), on Ctrl-C, after 10 seconds without actions, or on an error (a
  camera or bus failure). Then the arm goes back to its start pose at 30 degrees per second and the torque goes off. A
  second Ctrl-C turns the torque off at once: the arm goes limp and sags onto the table. A third ends the program.
- Before any motion, a prompt says that the arm will move, for how long and how to stop it (`--yes` skips it; without a
  terminal `pgs run` refuses to start without `--yes`).
- On connect, each motor's goal is set to its present position before LeRobot's settings turn the torque on, so the arm
  does not jump to an old goal. LeRobot's settings: position mode, P 16, I 0, D 32, the gripper's torque at 50 %. If
  that setup fails part way, the torque goes off.
- A chunk that is not finite or not 6 wide is never played: the first one stops the run before any motion, a later one
  makes the arm hold.
- `--dry-run` writes nothing to the motors: no settings, no goals, no torque change.

Not covered: contact or collision detection, limits in Cartesian space (the arm can reach the table within its joint
ranges), current or temperature monitoring beyond the motors' own protections. Keep a hand on the power switch.

## Timing

The model returns 30 actions per call, 1 s at its 30 Hz. `pgs run` ticks at the model's rate (`--hz` changes it, and
then the chunk plays faster or slower than it was trained). Each tick it reads the joints and cameras, hands the
observation to `ActionQueue` and sends one target. The queue asks the model for the next chunk in the background when
fewer than 15 actions are left, sending the newest observation, and merges the reply aligned: the actions that played
while the request was in flight are skipped, the rest replaces what was left (README, `ActionQueue`).

Measured in `--sim` with the real model (2026-11-16, one RTX 4090 worker on the fast path in US-IL-1, the client on a
Mac on home Wi-Fi in the US east, two 640 x 480 frames per request as JPEG, one run of 180 s):

| What | Measured |
|---|---|
| Control rate | 5354 ticks at 30 Hz: tick period p50 33.3 ms, p99 38.3 ms; 67 ticks late (1.3 %), the longest 326 ms; 29.7 Hz on average |
| Round trip of a request | p50 216 ms, p95 315 ms, p99 447 ms (357 requests) |
| Time on the server | p50 112 ms (JPEG decode and model); the model's call alone p50 109 ms, p95 116 ms |
| Actions played per chunk before the next one arrived | median 15 (5th to 95th percentile 12 to 17) of 30; about 8 more per chunk were skipped as already past |
| Holding | 22 ticks in 180 s without an action, the longest 0.20 s (7 of them at the start, waiting for the first queued chunk) |
| Age of an action when played | 0.50 s on average from its observation, at most 1.5 s |
| Errors | none; no stale chunk |

In the simulation the arm unfolded, reached toward the cube and the bowl, then held a reaching pose with its wrist at
the end of its range (the limit clamped those targets). The rendered scene is not what the model was trained on, so
this says nothing about the task. A 20 s dry run against the same worker printed first targets within 7 degrees of
the arm's reading at rest.

## Simulation: `--sim`

`pgs run so101 --sim` replaces the arm with TheRobotStudio's MuJoCo model of the SO-101 (SO-ARM100 repository, Apache
License 2.0, `Simulation/SO101/so101_new_calib.xml` and its meshes at commit a758567; about 16 MB, downloaded once to
`~/.cache/playground-serve/so101-mujoco/` and checked against stored SHA-256 sums). The arm stands on a table with a
cube and a bowl, and two cameras render the `top` and `side` views at 640 x 480. It reads and takes LeRobot's units, so
the same conversion to the model runs, and the same loop and limits. It checks the plumbing, the units and the timing;
it does not show that a task succeeds (the rendered scene is not what the model was trained on).

## Options

| Option | Meaning |
|---|---|
| `ROBOT` | `so101` |
| `--model` | Model name or `wss://` URL (default `molmoact2_so101`) |
| `--instruction` | The task in words (required) |
| `--port` | The arm's serial port (default: the only USB serial device) |
| `--cameras` | Camera role to OpenCV device index or path (default `top=0,side=1`) |
| `--hz` | Control rate (default: the model's action rate, 30) |
| `--dry-run` | Read the arm and cameras, print the planned targets, send nothing to the motors |
| `--max-seconds` | Time limit (default 60) |
| `--yes` | Start without the prompt |
| `--sim` | The simulated SO-101 instead of the arm |
| `--id` | LeRobot id of the arm's calibration file, when there are several |

## Development

`uv run pytest -q tests/run` runs the loop against a kinematic arm and a stand-in policy (rate, chunk playback, hold
on a late reply, limits, the refused start, Ctrl-C, dry run, stall), the unit mapping, the command with the MolmoAct2
worker's own server, and the simulator with a stand-in arm. The development environment has no LeRobot (its numpy 2
conflicts with openpi-client's numpy below 2), so `tests/run/test_so101.py` drives SO101 through a stand-in for
LeRobot's driver; `test_lerobot_units` checks the real driver's names and formulas when LeRobot is installed. In an
environment of its own, so that the development one stays:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/pgs-lerobot uv run --python 3.12 --extra so101 --extra sim --no-group dev \
  --with pytest pytest -q tests/run
```
