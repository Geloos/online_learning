"""Duck-typed fakes for JVM-free unit tests of the detector ensembles."""


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
