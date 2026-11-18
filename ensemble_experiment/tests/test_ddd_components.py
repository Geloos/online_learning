import math

import pytest

from ensemble_experiment.bagging import majority
from ensemble_experiment.cap import AdaptiveCap
from ensemble_experiment.pool import ConceptPool
from ensemble_experiment.tracker import PrequentialTracker, SlidingAcc

CAP = dict(cap_init=500, cap_floor=100, cap_ceil=2000, cap_margin=1.5,
           cap_growth=1.25, cap_history=10, cap_min_history=3)


# --- trackers ---------------------------------------------------------------
def test_prequential_tracker_acc_std_reset():
    tr = PrequentialTracker()
    assert tr.acc == 0.0 and tr.std == 0.0
    for c in [1, 1, 1, 0]:
        tr.update(c)
    assert tr.acc == 0.75
    assert tr.std == pytest.approx(math.sqrt(0.75 * 0.25 / 4))
    cp = tr.copy()
    tr.reset()
    assert tr.n == 0 and cp.n == 4 and cp.acc == 0.75


def test_sliding_acc_window():
    s = SlidingAcc(3)
    assert s.acc is None
    for c in [1, 1, 0, 0]:
        s.update(c)
    assert s.acc == pytest.approx(1 / 3)  # window = [1, 0, 0]
    s.reset()
    assert s.acc is None


# --- cap ----------------------------------------------------------------------
def test_cap_fallback_before_min_history():
    cap = AdaptiveCap(**CAP)
    cap.on_natural_exit("fast", 100)
    cap.on_natural_exit("fast", 120)
    assert cap.get("fast") == 500  # only 2 < 3 natural exits
    assert cap.get("slow") == 500


def test_cap_p90_update_and_kinds_independent():
    cap = AdaptiveCap(**CAP)
    for s in [100, 200, 300]:
        cap.on_natural_exit("fast", s)
    # p90 of [100, 200, 300] = 280 -> 1.5 * 280 = 420
    assert cap.get("fast") == pytest.approx(420)
    assert cap.get("slow") == 500


def test_cap_clip_bounds():
    cap = AdaptiveCap(**CAP)
    for s in [10, 10, 10]:
        cap.on_natural_exit("fast", s)
    assert cap.get("fast") == 100  # floor
    for s in [5000, 5000, 5000]:
        cap.on_natural_exit("slow", s)
    assert cap.get("slow") == 2000  # ceil


def test_cap_forced_exit_grows_and_is_not_recorded():
    cap = AdaptiveCap(**CAP)
    cap.on_forced_exit("slow")
    assert cap.get("slow") == pytest.approx(625)
    assert len(cap.exit_times["slow"]) == 0
    for _ in range(20):
        cap.on_forced_exit("slow")
    assert cap.get("slow") == 2000


# --- pool ---------------------------------------------------------------------
def test_pool_fifo_eviction_and_replace():
    pool = ConceptPool(2)
    a, ev = pool.add("A", 1)
    assert ev is None
    b, _ = pool.add("B", 2)
    c, ev = pool.add("C", 3)
    assert ev == a and pool.ids() == [b, c]
    assert pool.replace(b, "B2", 4)
    assert pool.get(b)["ensemble"] == "B2"
    assert pool.ids() == [b, c]  # replace keeps the FIFO position
    assert not pool.replace(a, "X", 5)  # evicted


def test_pool_disabled():
    pool = ConceptPool(0)
    assert pool.add("A", 1) == (None, None)
    assert len(pool) == 0


# --- bagging ------------------------------------------------------------------
class _CountingTree:
    def __init__(self):
        self.calls = 0

    def trainOnInstance(self, ji):
        self.calls += 1


class _Inst:
    java_instance = object()


def test_learn_trains_each_tree_exactly_k_times():
    import numpy as np

    from ensemble_experiment.bagging import OnlineBaggingEnsemble

    trees = [_CountingTree() for _ in range(10)]
    ens = OnlineBaggingEnsemble(trees, np.random.default_rng(3))
    expected = np.random.default_rng(3).poisson(2.0, size=10)
    ens.learn(_Inst(), 2.0)
    assert [t.calls for t in trees] == [int(k) for k in expected]


def test_majority_vote_ties_and_abstain():
    assert majority([1, 1, 0]) == 1
    assert majority([1, 0]) == 0  # tie -> lowest label
    assert majority([None, 2, None]) == 2
    assert majority([None, None]) is None
