"""Duck-typed fakes for JVM-free tests of the DDD state machine."""

import itertools

_ids = itertools.count()


class FakeInst:
    def __init__(self, y: int = 0):
        self.y_index = y


class FakeEnsemble:
    """Predicts ``pred`` (settable); records learn calls into an optional
    shared event log as ('predict'|'learn', ensemble id)."""

    def __init__(self, pred=0, log=None):
        self.id = next(_ids)
        self.pred = pred
        self.log = log
        self.learned: list[float] = []
        self.copied_from = None

    def predict(self, inst):
        if self.log is not None:
            self.log.append(("predict", self.id))
        return self.pred

    def learn(self, inst, lam):
        self.learned.append(lam)
        if self.log is not None:
            self.log.append(("learn", self.id))

    def frozen_copy(self):
        c = FakeEnsemble(self.pred, self.log)
        c.copied_from = self
        return c


class FakeFactory:
    def __init__(self, pred=0, log=None):
        self.pred = pred
        self.log = log
        self.made: list[FakeEnsemble] = []

    def __call__(self):
        e = FakeEnsemble(self.pred, self.log)
        self.made.append(e)
        return e


class StubSignal:
    """Fires on the scripted t values; counts resets."""

    def __init__(self, fire_at=()):
        self.fire_at = set(fire_at)
        self.seen: list[tuple[int, int]] = []
        self.resets = 0
        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []
        self.logs: dict = {}

    def update(self, t, err):
        self.seen.append((t, err))
        if t in self.fire_at:
            self.alarms.append(t)
            return True
        return False

    def reset(self):
        self.resets += 1
