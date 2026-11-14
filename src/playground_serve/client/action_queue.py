"""An action queue that keeps a robot's control loop fed from a remote policy without waiting on the network.

The pattern is LeRobot's remote inference (github.com/huggingface/lerobot at commit
fe1c54594f6ba275fccf210b9f7b11f0e2324b89: docs/source/remote_inference.mdx, `chunk_merge=aligned`, and
src/lerobot/policies/rtc/action_queue.py, `ActionQueue`), cut down to one thread and numpy:

- The control loop calls `put_observation(obs)` and `get_action()` once per tick. Neither waits on the network.
- A background thread calls `policy.infer` with the newest observation whenever fewer than `threshold` x (chunk length)
  actions are left. One request is in flight at a time, and each observation is sent at most once.
- Aligned merge: when a chunk arrives, its first k actions are dropped, k = the number of actions the control loop took
  from the queue since that observation was put, and the rest replaces whatever is left of the old chunk. As in
  LeRobot's `_check_and_resolve_delays`, k counts actions executed, not ticks elapsed: ticks on which the robot held
  (empty queue) did not move it along the plan, so they are not skipped. If k >= the chunk length the chunk is stale:
  it is counted and discarded, the old remainder stays, and the thread asks again with a newer observation.
- `get_action()` returns None when the queue is empty: hold the robot. Empty ticks and the longest run of them are
  counted.
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class ActionQueue:
    """with ActionQueue(policy, threshold=0.5) as queue:
        while running:                                 # at the control rate
            queue.put_observation(read_observation())
            action = queue.get_action()                # None: queue empty, hold
    """

    def __init__(self, policy=None, *, threshold: float = 0.5, clock=time.monotonic) -> None:
        self.policy, self.threshold, self._clock = policy, threshold, clock
        self._cond = threading.Condition()
        self._actions: collections.deque = collections.deque()  # (action, time of its observation)
        self._chunk_len = 0
        self._obs, self._obs_seq, self._obs_cursor, self._obs_time = None, 0, 0, 0.0
        self._closed = False
        self.executed = 0  # actions handed to the control loop
        self.ticks = self.empty_ticks = self.max_empty_run = self._empty_run = 0
        self.requests = self.chunks = self.stale_chunks = self.dropped = self.replaced = self.errors = 0
        self.last_error: str | None = None
        self.age_ms_sum = self.age_ms_max = 0.0
        self.rtt_ms: collections.deque = collections.deque(maxlen=100_000)
        self._thread = None
        if policy is not None:
            self._thread = threading.Thread(target=self._run, name="playground-serve-queue", daemon=True)
            self._thread.start()

    def __len__(self) -> int:
        with self._cond:
            return len(self._actions)

    def put_observation(self, obs: dict) -> None:
        """The newest observation; the inference thread sends it when the queue runs low. Do not modify it after."""
        with self._cond:
            self._obs, self._obs_cursor, self._obs_time = obs, self.executed, self._clock()
            self._obs_seq += 1
            self._cond.notify()

    def get_action(self) -> np.ndarray | None:
        """The next action, or None if the queue is empty (the robot should hold its last target)."""
        with self._cond:
            self.ticks += 1
            if not self._actions:
                self.empty_ticks += 1
                self._empty_run += 1
                self.max_empty_run = max(self.max_empty_run, self._empty_run)
                return None
            action, obs_time = self._actions.popleft()
            self.executed += 1
            self._empty_run = 0
            age = (self._clock() - obs_time) * 1e3
            self.age_ms_sum += age
            self.age_ms_max = max(self.age_ms_max, age)
            if self._low():
                self._cond.notify()
            return action

    def merge(self, chunk, cursor: int, obs_time: float) -> int:
        """Aligned merge of a chunk computed from the observation put when `executed` was `cursor`. Returns the number
        of its actions queued (0 if stale)."""
        chunk = np.asarray(chunk)
        with self._cond:
            self.chunks += 1
            self._chunk_len = len(chunk)
            k = self.executed - cursor
            if k >= len(chunk):
                self.stale_chunks += 1
                self.dropped += len(chunk)
                return 0
            self.dropped += k
            self.replaced += len(self._actions)
            self._actions = collections.deque((a, obs_time) for a in chunk[k:])
            return len(chunk) - k

    def _low(self) -> bool:
        return not self._actions or len(self._actions) < self.threshold * self._chunk_len

    def _run(self) -> None:
        sent = 0  # sequence number of the last observation sent
        while True:
            with self._cond:
                while not self._closed and not (self._obs_seq != sent and self._low()):
                    self._cond.wait()
                if self._closed:
                    return
                obs, cursor, obs_time, sent = self._obs, self._obs_cursor, self._obs_time, self._obs_seq
                self.requests += 1
            try:
                chunk = self.policy.infer(obs)["actions"]
            except Exception as e:  # the policy reconnects on its next call; keep the loop alive
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("inference failed: %s", self.last_error)
                continue
            self.rtt_ms.append(self.policy.timing["rtt_ms"])
            self.merge(chunk, cursor, obs_time)

    def stats(self) -> dict:
        with self._cond:
            return {"ticks": self.ticks, "empty_ticks": self.empty_ticks, "max_empty_run": self.max_empty_run,
                    "executed": self.executed, "requests": self.requests, "chunks": self.chunks,
                    "stale_chunks": self.stale_chunks, "dropped": self.dropped, "replaced": self.replaced,
                    "errors": self.errors, "last_error": self.last_error,
                    "mean_age_ms": self.age_ms_sum / self.executed if self.executed else None,
                    "max_age_ms": self.age_ms_max if self.executed else None}

    def close(self, timeout: float = 10.0) -> None:
        """Stop the inference thread (after its request in flight, if any). The policy stays open."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def __enter__(self) -> "ActionQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
