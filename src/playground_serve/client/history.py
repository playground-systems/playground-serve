"""Observation history for models that take the last few control ticks (Xiaomi-Robotics-1: 4 ticks, 2 apart).

    history = History.from_metadata(policy.metadata)   # or History(keys, length=4, interval=2)
    actions = policy.act(history(obs))                 # obs: this tick's frames and state, as for any model
    history.reset()                                    # at an episode's start

Each call adds this tick's value of every key in `keys` and replaces it in the returned observation with a stack of
`length` values, oldest first, `interval` ticks apart, ending with this tick's: (length, H, W, 3) for a camera frame,
(length, D) for the state. At an episode's start the oldest value stands in for ticks that do not exist yet, as the
models' own evaluation code does (Xiaomi-Robotics-1's sample_history). The client sends a frame stack as one JPEG per
frame where the server advertises "jpeg_stack". Imports on Python 3.8 like the client.
"""

from __future__ import annotations

import collections

import numpy as np


class History:
    def __init__(self, keys, length: int, interval: int = 1) -> None:
        if length < 1 or interval < 1:
            raise ValueError("length and interval must be at least 1")
        self.keys, self.length, self.interval = list(keys), int(length), int(interval)
        self._items = {k: collections.deque(maxlen=(self.length - 1) * self.interval + 1) for k in self.keys}

    @classmethod
    def from_metadata(cls, metadata: dict) -> "History":
        """From a server's metadata: "history" (length), "history_interval", "cameras" and the state key."""
        length = int(metadata.get("history") or 0)
        if length < 1:
            raise ValueError("this model's metadata names no history: it takes one tick per request")
        keys = list(metadata.get("cameras") or []) + [metadata.get("state_key", "state")]
        return cls(keys, length, int(metadata.get("history_interval") or 1))

    def reset(self) -> None:
        for items in self._items.values():
            items.clear()

    def __call__(self, obs: dict) -> dict:
        out = dict(obs)
        for key in self.keys:
            if key not in obs:
                raise KeyError(f"observation lacks {key!r}, which this history keeps")
            items = self._items[key]
            items.append(np.asarray(obs[key]))
            n = len(items)
            picks = [max(0, n - 1 - (self.length - 1 - i) * self.interval) for i in range(self.length)]
            out[key] = np.ascontiguousarray(np.stack([items[i] for i in picks], axis=0))
        return out
