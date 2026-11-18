"""Alarm verification with two Naive Bayes shadow models (PHASE 7).

At alarm time t_a a Verification is created:

* S ("stable") trains on the pre-alarm buffer slice
  [t_a - D - N + 1 .. t_a - D] and is then frozen.
* R ("recent") trains online on instances t_a+1 .. t_a+N.
* Test phase t_a+N+1 .. t_a+N+E: S and R both predict, NEITHER trains.
  b += (S right and R wrong); c += (S wrong and R right).
* At t = t_a+N+E the one-sided McNemar test decides the verdict.

The model objects only need ``train(instance)`` and ``predict(instance)``;
instances only need ``y_index`` — so tests can use plain-Python fakes and the
real runs use capymoa NaiveBayes / LabeledInstance.
"""

from __future__ import annotations

from scipy.stats import binomtest

CONFIRMED = "CONFIRMED"
FALSE_ALARM = "FALSE_ALARM"
INCONCLUSIVE = "INCONCLUSIVE"


def mcnemar_one_sided(b: int, c: int) -> float:
    """One-sided exact McNemar p-value: P(X >= c | X ~ Bin(b + c, 0.5))."""
    if b + c == 0:
        return 1.0
    return binomtest(c, b + c, 0.5, alternative="greater").pvalue


def verdict(b: int, c: int, alpha: float) -> str:
    p = mcnemar_one_sided(b, c)
    if p < alpha:
        return CONFIRMED
    if b >= c:
        return FALSE_ALARM
    return INCONCLUSIVE


class Verification:
    """One in-flight verification of a single ensemble (or shadow) alarm."""

    def __init__(
        self,
        t_alarm: int,
        trigger: str,
        cluster: str,
        n_train: int,
        gap: int,
        n_test: int,
        alpha: float,
        shadow: bool,
        nb_factory,
        buffer,
    ):
        """``buffer`` is an iterable of (t, instance) pairs that must cover at
        least [t_alarm - gap - n_train + 1 .. t_alarm - gap]."""
        assert t_alarm >= n_train + gap, (
            f"alarm at t={t_alarm} before the buffer can hold N+D="
            f"{n_train + gap} instances (MIN_TRAINING_SIZE should prevent this)"
        )
        self.t_alarm = t_alarm
        self.trigger = trigger
        self.cluster = cluster
        self.n_train = n_train
        self.gap = gap
        self.n_test = n_test
        self.alpha = alpha
        self.shadow = shadow

        lo = t_alarm - gap - n_train + 1
        hi = t_alarm - gap
        self.model_s = nb_factory()
        n_seen = 0
        for t, inst in buffer:
            if lo <= t <= hi:
                self.model_s.train(inst)
                n_seen += 1
        assert n_seen == n_train, (
            f"S trained on {n_seen} instances, expected {n_train} "
            f"(buffer did not cover [{lo}, {hi}])"
        )
        self.model_r = nb_factory()

        self.b = 0  # S right, R wrong
        self.c = 0  # S wrong, R right
        self.p_value: float | None = None
        self.verdict: str | None = None
        self.t_decision: int | None = None
        self.finished = False
        # (t, detector) for every detector firing in (t_alarm, t_alarm+N+E],
        # plus any co-firing at t_alarm itself, added by the ensemble.
        self.cofired: list[tuple[int, str]] = []

    @property
    def t_end(self) -> int:
        return self.t_alarm + self.n_train + self.n_test

    @property
    def cofired_names(self) -> set[str]:
        return {name for _, name in self.cofired}

    def add_cofired(self, t: int, name: str) -> None:
        self.cofired.append((t, name))

    def step(self, t: int, instance) -> bool:
        """Feed the instance at time t. Returns True when the verification
        finished (verdict available)."""
        if self.finished or t <= self.t_alarm:
            return self.finished
        if t <= self.t_alarm + self.n_train:
            self.model_r.train(instance)
        elif t <= self.t_end:
            y = instance.y_index
            s_ok = self.model_s.predict(instance) == y
            r_ok = self.model_r.predict(instance) == y
            if s_ok and not r_ok:
                self.b += 1
            elif r_ok and not s_ok:
                self.c += 1
        if t == self.t_end:
            self.p_value = mcnemar_one_sided(self.b, self.c)
            self.verdict = verdict(self.b, self.c, self.alpha)
            self.t_decision = t
            self.finished = True
        return self.finished

    def record(self) -> dict:
        return {
            "t_alarm": self.t_alarm,
            "trigger": self.trigger,
            "cluster": self.cluster,
            "shadow": self.shadow,
            "N": self.n_train,
            "D": self.gap,
            "E": self.n_test,
            "b": self.b,
            "c": self.c,
            "p": self.p_value,
            "verdict": self.verdict,
            "cofired": sorted(self.cofired_names),
            "t_decision": self.t_decision,
        }
