"""Accuracy trackers used by the DDD state machine."""

from __future__ import annotations

import math
from collections import deque


class PrequentialTracker:
    """Accuracy since the last reset, plus its binomial standard deviation
    std = sqrt(acc * (1 - acc) / n). Minku & Yao do not define std; the
    binomial form is our choice (std = 0 when n = 0)."""

    __slots__ = ("n", "n_correct")

    def __init__(self):
        self.n = 0
        self.n_correct = 0

    def reset(self) -> None:
        self.n = 0
        self.n_correct = 0

    def update(self, correct: bool) -> None:
        self.n += 1
        self.n_correct += int(bool(correct))

    @property
    def acc(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if self.n == 0:
            return 0.0
        a = self.acc
        return math.sqrt(a * (1.0 - a) / self.n)

    def copy(self) -> "PrequentialTracker":
        other = PrequentialTracker()
        other.n = self.n
        other.n_correct = self.n_correct
        return other


class SlidingAcc:
    """Accuracy over the last ``window`` correctness values (None if empty)."""

    def __init__(self, window: int):
        self.values: deque = deque(maxlen=window)
        self._sum = 0

    def reset(self) -> None:
        self.values.clear()
        self._sum = 0

    def update(self, correct: bool) -> None:
        v = int(bool(correct))
        if len(self.values) == self.values.maxlen:
            self._sum -= self.values[0]
        self.values.append(v)
        self._sum += v

    @property
    def acc(self) -> float | None:
        return self._sum / len(self.values) if self.values else None
