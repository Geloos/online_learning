"""Adaptive per-kind cap on the length of DDD's after-drift mode.

Each drift kind ("fast" / "slow") has its own cap. It starts at cap_init.
After cap_min_history natural exits of that kind it becomes
clip(margin * p90(recent natural exit times), floor, ceil). A forced exit
multiplies it by growth (capped at ceil) and is NOT recorded as an exit
time — recording the cap-length episodes would feed the cap back into
itself.
"""

from __future__ import annotations

from collections import deque

import numpy as np

KINDS = ("fast", "slow")


class AdaptiveCap:
    def __init__(self, *, cap_init, cap_floor, cap_ceil, cap_margin,
                 cap_growth, cap_history, cap_min_history):
        self.floor = cap_floor
        self.ceil = cap_ceil
        self.margin = cap_margin
        self.growth = cap_growth
        self.min_history = cap_min_history
        self.cap = {k: float(cap_init) for k in KINDS}
        self.exit_times = {k: deque(maxlen=cap_history) for k in KINDS}
        self.n_natural = {k: 0 for k in KINDS}
        self.n_forced = {k: 0 for k in KINDS}

    @classmethod
    def from_cfg(cls, cfg: dict) -> "AdaptiveCap":
        keys = ("cap_init", "cap_floor", "cap_ceil", "cap_margin",
                "cap_growth", "cap_history", "cap_min_history")
        return cls(**{k: cfg[k] for k in keys})

    def get(self, kind: str) -> float:
        return self.cap[kind]

    def on_natural_exit(self, kind: str, steps: int) -> None:
        self.exit_times[kind].append(steps)
        self.n_natural[kind] += 1
        if len(self.exit_times[kind]) >= self.min_history:
            p90 = float(np.percentile(list(self.exit_times[kind]), 90))
            self.cap[kind] = float(np.clip(self.margin * p90,
                                           self.floor, self.ceil))

    def on_forced_exit(self, kind: str) -> None:
        self.n_forced[kind] += 1
        self.cap[kind] = min(self.ceil, self.cap[kind] * self.growth)
