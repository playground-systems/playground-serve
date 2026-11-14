"""ActionQueue: aligned merge arithmetic, empty-queue counting, and a background thread that survives reconnects."""

from __future__ import annotations

import threading
import time

import numpy as np
from conftest import free_port

from playground_serve.client import ActionQueue, connect
from playground_serve.client.mock_server import Config, MockServer


def rows(start: int, n: int) -> np.ndarray:
    """A chunk whose action i is filled with start + i, so every action is recognisable."""
    return np.repeat(np.arange(start, start + n, dtype=np.float32)[:, None], 7, axis=1)


def test_aligned_merge():
    now = [10.0]
    q = ActionQueue(clock=lambda: now[0])
    assert q.merge(rows(100, 10), cursor=0, obs_time=10.0) == 10
    now[0] = 10.25
    assert [q.get_action()[0] for _ in range(3)] == [100, 101, 102]
    assert q.stats()["mean_age_ms"] == 250.0
    # Observation taken when 1 action had been executed; 2 more were executed while it was in flight: drop 2 and
    # replace the 7 actions left of the old chunk.
    assert q.merge(rows(200, 10), cursor=1, obs_time=10.2) == 8
    assert len(q) == 8 and q.dropped == 2 and q.replaced == 7
    assert q.get_action()[0] == 202
    # Arrived with nothing executed since its observation: nothing dropped.
    assert q.merge(rows(300, 10), cursor=q.executed, obs_time=10.25) == 10
    assert q.get_action()[0] == 300


def test_stale_chunk_keeps_old_remainder():
    q = ActionQueue()
    q.merge(rows(100, 20), cursor=0, obs_time=0.0)  # e.g. a policy with a 20-step horizon
    for _ in range(12):
        q.get_action()
    # 12 actions executed since the observation, the chunk holds 10: all stale, the old remainder stays.
    assert q.merge(rows(200, 10), cursor=0, obs_time=0.0) == 0
    assert q.stale_chunks == 1 and q.dropped == 10 and q.replaced == 0 and len(q) == 8
    assert q.get_action()[0] == 112
    # Exactly as many executed as the chunk holds is stale too.
    q.merge(rows(300, 10), cursor=q.executed - 10, obs_time=0.0)
    assert q.stale_chunks == 2 and q.get_action()[0] == 113


def test_empty_queue_is_reported():
    q = ActionQueue(threshold=0.5)
    assert [q.get_action() for _ in range(3)] == [None, None, None]
    assert q.empty_ticks == 3 and q.max_empty_run == 3 and q._low()
    q.merge(rows(0, 10), cursor=0, obs_time=0.0)
    assert not q._low()
    for _ in range(5):
        assert q.get_action() is not None
    assert len(q) == 5 and not q._low()  # 5 left: not below half a chunk
    q.get_action()
    assert q._low()  # 4 left
    for _ in range(4):
        q.get_action()
    assert q.get_action() is None and q.get_action() is None
    s = q.stats()
    assert s["ticks"] == 15 and s["empty_ticks"] == 5 and s["max_empty_run"] == 3 and s["executed"] == 10


def run_loop(q: ActionQueue, observations, seconds: float, hz: float = 50) -> tuple[list, float]:
    """A control loop at `hz`; returns the actions taken (None when empty) and the longest put+get call in ms."""
    actions, slowest, t0 = [], 0.0, time.monotonic()
    for i in range(int(seconds * hz)):
        time.sleep(max(0.0, t0 + i / hz - time.monotonic()))
        t = time.perf_counter()
        q.put_observation(observations[i % 4])
        actions.append(q.get_action())
        slowest = max(slowest, (time.perf_counter() - t) * 1e3)
    return actions, slowest


def test_queue_feeds_control_loop(observations):
    cfg = Config(infer_ms=20, net_p50_ms=10, net_p95_ms=20)
    with MockServer(cfg) as mock, connect(mock.url) as policy, ActionQueue(policy, threshold=0.5) as q:
        actions, slowest = run_loop(q, observations, 1.0)
        assert all(a.shape == (7,) for a in actions if a is not None)
        assert slowest < 20  # the loop never waited on the network
        assert q.chunks >= 5 and q.requests - q.chunks <= 1 and q.errors == 0
        assert q.empty_ticks <= 5  # the first chunk (one round trip) and little else at 50 Hz with ~40 ms replies


def test_queue_survives_dropped_sockets(observations):
    cfg = Config(infer_ms=5, drop_every=3)  # every third request: abort the TCP connection without replying
    with MockServer(cfg) as mock, connect(mock.url) as policy, ActionQueue(policy) as q:
        actions, _ = run_loop(q, observations, 1.5)
        assert policy.reconnects >= 3 and q.errors == 0
        assert sum(a is not None for a in actions[-25:]) >= 20  # still fed after the drops


def test_queue_survives_server_restart(observations):
    port = free_port()
    mock = MockServer(Config(infer_ms=5), port=port).__enter__()
    with connect(mock.url) as policy, ActionQueue(policy) as q:
        result: list = []
        loop = threading.Thread(target=lambda: result.append(run_loop(q, observations, 3.0)))
        loop.start()
        time.sleep(0.5)
        mock.__exit__(None, None, None)  # the server goes away while the queue thread runs
        chunks_before = q.chunks
        time.sleep(0.3)
        mock = MockServer(Config(infer_ms=5), port=port).__enter__()
        loop.join()
        actions = result[0][0]
        assert policy.reconnects >= 1 and q.chunks > chunks_before + 10
        assert sum(a is not None for a in actions[-50:]) >= 45  # fed again after the restart
    mock.__exit__(None, None, None)
