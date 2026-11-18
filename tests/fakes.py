"""Duck-typed fakes for JVM-free unit tests of verifier and ensemble."""


class FakeInstance:
    def __init__(self, t: int, y: int = 0):
        self.t = t
        self.y_index = y


class RecordingModel:
    """Model that records what it saw; optionally always right/wrong."""

    def __init__(self, right: bool = True):
        self.right = right
        self.trained: list[int] = []
        self.predicted: list[int] = []

    def train(self, inst):
        self.trained.append(inst.t)

    def predict(self, inst):
        self.predicted.append(inst.t)
        return inst.y_index if self.right else inst.y_index + 1


class ModelFactory:
    """nb_factory handing out pre-built models in order (S first, then R,
    for each verification in turn)."""

    def __init__(self, models):
        self.models = list(models)
        self.handed_out: list[RecordingModel] = []

    def __call__(self):
        m = self.models.pop(0)
        self.handed_out.append(m)
        return m


def false_alarm_factory(n: int = 10) -> ModelFactory:
    """Every verification: S always right, R always wrong -> b=E, c=0."""
    return ModelFactory([RecordingModel(right=(i % 2 == 0))
                         for i in range(2 * n)])


def confirmed_factory(n: int = 10) -> ModelFactory:
    """Every verification: S always wrong, R always right -> b=0, c=E
    (CONFIRMED whenever E >= 5 at alpha=0.05)."""
    return ModelFactory([RecordingModel(right=(i % 2 == 1))
                         for i in range(2 * n)])


class StubDetector:
    """Fires exactly on its scripted add_element call indices."""

    def __init__(self, fire_at=()):
        self.fire_at = set(fire_at)
        self.n_calls = 0
        self._fired = False
        self.reset_count = 0

    def add_element(self, element):
        self._fired = self.n_calls in self.fire_at
        self.n_calls += 1

    def detected_change(self):
        return self._fired

    def reset(self):
        self.reset_count += 1
