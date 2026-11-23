"""Extended DDD: its parts and the state machine, with fake ensembles and
signals (no JVM).
"""

from __future__ import annotations

import itertools
import math

import pytest

from ensemble_experiment import config
from ensemble_experiment.ddd import (
    AdaptiveCap,
    ConceptPool,
    ExtendedDDD,
    PrequentialTracker,
    SlidingAcc,
)
from ensemble_experiment.models import majority


# ===========================================================================
# Fakes
#
# Duck-typed fakes for JVM-free tests of the DDD state machine.
# ===========================================================================
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


# ===========================================================================
# Parts
# ===========================================================================
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

    from ensemble_experiment.models import OnlineBaggingEnsemble

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


# ===========================================================================
# State machine
#
# JVM-free tests of the ExtendedDDD state machine (fake ensembles and a
# scripted drift signal).
# ===========================================================================
def make_ddd(fire_at=(), log=None, **cfg_overrides):
    cfg = {**config.DDD, "min_steps": 5, "ref_window": 10,
           **cfg_overrides}
    factory = FakeFactory(pred=0, log=log)
    signal = StubSignal(fire_at)
    events = []
    ddd = ExtendedDDD(cfg, factory, signal, min_training_size=0,
                      listener=events.append)
    return ddd, factory, signal, events


def set_acc(ddd, voter, n, correct):
    tr = ddd.trackers[voter]
    tr.n, tr.n_correct = n, correct


def enter_after(ddd, t=0):
    """Fire the stub signal at step t (y=0, every fake predicts 0)."""
    ddd.signal.fire_at.add(t)
    ddd.step(t, FakeInst(0))
    assert ddd.mode == "after"


# --- weighted vote --------------------------------------------------------
def test_weighted_vote_skips_none_and_breaks_ties_low():
    w = {"nl": 0.25, "ol": 0.25, "oh": 0.5}
    assert ExtendedDDD.weighted_vote({"nl": 1, "ol": 1, "oh": 0}, w) == 0
    assert ExtendedDDD.weighted_vote({"nl": 1, "ol": None, "oh": None},
                                     w) == 1
    assert ExtendedDDD.weighted_vote({"nl": 1, "ol": 0, "oh": None},
                                     w) == 0  # tie -> lowest
    assert ExtendedDDD.weighted_vote({"nl": None, "ol": None, "oh": None},
                                     w) is None


def test_vote_weights_equal_when_all_zero_and_W_applies():
    ddd, *_ = make_ddd(W=2.0)
    enter_after(ddd)
    w = ddd.vote_weights()
    assert w == {"nl": pytest.approx(1 / 3), "ol": pytest.approx(1 / 3),
                 "oh": pytest.approx(1 / 3)}
    set_acc(ddd, "nl", 10, 5)
    set_acc(ddd, "ol", 10, 5)
    set_acc(ddd, "oh", 10, 10)
    w = ddd.vote_weights()  # raw: 0.5, 0.5*2, 1.0 -> sum 2.5
    assert w["ol"] == pytest.approx(0.4)
    assert w["nl"] == pytest.approx(0.2)
    assert w["oh"] == pytest.approx(0.4)


# --- detection ------------------------------------------------------------
def test_detection_from_before_reassigns_and_pools_a_copy():
    ddd, factory, signal, events = make_ddd()
    nl, nh = ddd.new_low, ddd.new_high
    for t in range(3):  # fill the main window: 3 correct
        ddd.step(t, FakeInst(0))
    enter_after(ddd, t=3)
    assert ddd.old_low is nl and ddd.old_high is nh
    assert ddd.new_low not in (nl, nh) and ddd.new_high not in (nl, nh)
    assert len(ddd.pool) == 1
    stored = ddd.pool.entries[0]["ensemble"]
    assert stored.copied_from is nl and stored is not nl
    assert ddd.acc_ref == 1.0  # main window was all correct
    assert signal.resets == 1  # choice (a)
    assert ddd.steps_since_drift == 0 and ddd.kind is None
    assert [e["type"] for e in events] == ["pool_add", "detection"]


def test_detection_with_empty_window_gives_acc_ref_none_and_slow():
    ddd, *_ = make_ddd()
    # Fire at t=0: the window gets the step-0 update before detection, so
    # use an explicitly emptied window to cover the None branch.
    ddd.main_window.reset()
    ddd.mode = "before"
    ddd.signal.fire_at.add(0)
    # bypass step(): call the handler directly with an empty window
    ddd._on_detection(0)
    assert ddd.acc_ref is None
    ddd._classify()
    assert ddd.kind == "slow"


@pytest.mark.parametrize("nl_beats_oh", [True, False])
def test_detection_in_after_mode_reassignment_and_interrupt(nl_beats_oh):
    ddd, factory, signal, _ = make_ddd()
    enter_after(ddd, t=0)
    acc_ref = ddd.acc_ref
    nl, oh, nh = ddd.new_low, ddd.old_high, ddd.new_high
    ddd.recalled = FakeEnsemble()
    ddd.recalled_pool_id = 0
    set_acc(ddd, "nl", 10, 9 if nl_beats_oh else 3)
    set_acc(ddd, "oh", 10, 5)
    pool_before = len(ddd.pool)
    # direct call: a step() would first overwrite the trackers
    ddd._on_detection(1)
    assert ddd.old_low is (nl if nl_beats_oh else oh)
    assert ddd.old_high is nh
    assert ddd.recalled is None and ddd.recalled_pool_id is None
    assert ddd.acc_ref == acc_ref  # kept in after-mode
    assert len(ddd.pool) == pool_before  # no pool add from after-mode
    assert ddd.episodes[-1]["exit_type"] == "interrupted"
    assert ddd.episodes[-1]["winner"] is None
    assert signal.resets == 2


# --- classification ---------------------------------------------------------
@pytest.mark.parametrize("acc_ref, ol_correct, kind", [
    (0.9, 7, "fast"),   # drop 0.2 > tau
    (0.75, 7, "slow"),  # drop 0.05
    (None, 0, "slow"),
])
def test_classify(acc_ref, ol_correct, kind):
    ddd, *_ = make_ddd()
    enter_after(ddd)
    ddd.acc_ref = acc_ref
    set_acc(ddd, "ol", 10, ol_correct)
    ddd._classify()
    assert ddd.kind == kind
    assert ddd.episode["kind"] == kind


# --- exit -------------------------------------------------------------------
def prepare_exit(ddd, accs, steps=10, kind="slow"):
    enter_after(ddd)
    for v, (n, c) in accs.items():
        set_acc(ddd, v, n, c)
    ddd.steps_since_drift = steps
    ddd.kind = kind
    ddd.episode["kind"] = kind


def test_exit_new_low_wins():
    ddd, _, signal, _ = make_ddd()
    prepare_exit(ddd, {"nl": (100, 90), "ol": (100, 50), "oh": (100, 50)})
    nl = ddd.new_low
    assert ddd._check_exit(10) == "natural_new_low"
    assert ddd.new_low is nl and ddd.mode == "before"
    assert ddd.old_low is None and ddd.old_high is None
    assert signal.resets == 1  # only the detection's reset
    assert ddd.cap.n_natural["slow"] == 1
    assert ddd.episodes[-1]["winner"] == "nl"


def test_exit_old_high_wins_and_resets_signal():
    ddd, _, signal, _ = make_ddd()
    prepare_exit(ddd, {"nl": (100, 50), "ol": (100, 50), "oh": (100, 95)})
    oh, nh = ddd.old_high, ddd.new_high
    assert ddd._check_exit(10) == "natural_old_high"
    assert ddd.new_low is oh and ddd.new_high is nh
    assert signal.resets == 2  # choice (b)


def test_exit_recalled_wins_and_updates_its_pool_entry():
    ddd, _, signal, events = make_ddd()
    prepare_exit(ddd, {"nl": (100, 50), "ol": (100, 50), "oh": (100, 50)})
    pid = ddd.pool.ids()[0]
    rc = FakeEnsemble()
    ddd.recalled, ddd.recalled_pool_id = rc, pid
    set_acc(ddd, "rc", 100, 95)
    assert ddd._check_exit(10) == "natural_recalled"
    assert ddd.new_low is rc
    stored = ddd.pool.get(pid)["ensemble"]
    assert stored.copied_from is rc and stored is not rc
    assert len(ddd.pool) == 1  # replaced, not duplicated
    assert events[-2]["type"] == "pool_replace"
    assert signal.resets == 2


def test_old_low_wins_only_on_forced_exit():
    ddd, *_ = make_ddd()
    # old-low clearly best, but no natural rule lets it win
    prepare_exit(ddd, {"nl": (100, 50), "ol": (100, 95), "oh": (100, 50)},
                 steps=10)
    assert ddd._check_exit(10) is None
    ddd.steps_since_drift = int(ddd.cap.get("slow"))
    ol = ddd.old_low
    assert ddd._check_exit(ddd.steps_since_drift) == "forced"
    assert ddd.new_low is ol
    assert ddd.episodes[-1]["winner"] == "ol"
    assert ddd.episodes[-1]["cap_used"] == 500
    assert ddd.cap.get("slow") == pytest.approx(625)  # grown
    assert len(ddd.cap.exit_times["slow"]) == 0  # not recorded


def test_no_forced_exit_in_plain_ddd():
    ddd, *_ = make_ddd(forced_exit=False)
    prepare_exit(ddd, {"nl": (100, 50), "ol": (100, 95), "oh": (100, 50)},
                 steps=10_000)
    assert ddd._check_exit(10_000) is None


# --- pool scoring and recall -----------------------------------------------
def test_recall_picks_best_scorer_ties_most_recent():
    ddd, factory, _, events = make_ddd(min_steps=3)
    # three pool entries predicting 1, 0, 0 (y = 0 throughout)
    for pred in (1, 0, 0):
        ddd.pool.add(FakeEnsemble(pred), 0)
    enter_after(ddd, t=0)  # adds a 4th entry (copy of new_low, pred 0)
    ids = ddd.pool.ids()
    for t in range(1, 4):
        ddd.step(t, FakeInst(0))
    assert all(ddd.pool_trackers[i].n == 3 for i in ids)
    # entries 1, 2, 3 all perfect -> the most recent (the just-added one);
    # everyone ties at 1.0, so nobody exits yet
    assert ddd.mode == "after"
    assert ddd.recalled_pool_id == ids[-1]
    rec = [e for e in events if e["type"] == "recall"]
    assert [e["id"] for e in rec] == [ids[-1]]


def test_recall_trains_a_copy_and_wins():
    ddd, *_ = make_ddd(min_steps=3)
    ddd.pool.add(FakeEnsemble(0), 0)  # the only correct entry
    good = ddd.pool.entries[0]
    good_ens = good["ensemble"]
    ddd.new_low.pred = 1  # current model is wrong
    enter_after(ddd, t=0)  # adds a second entry (copy of new_low, pred 1)
    for e in (ddd.new_low, ddd.old_low, ddd.old_high):
        e.pred = 1  # every live model wrong
    for t in range(1, 4):
        ddd.step(t, FakeInst(0))
    # step 3: recall entry 0 with its 3/3 scoring record, which then
    # dominates every live model -> natural_recalled exit
    ep = ddd.episodes[-1]
    assert ep["exit_type"] == "natural_recalled"
    assert ep["recalled_pool_id"] == good["id"]
    assert ep["recall_score"] == 1.0
    rc = ddd.new_low
    assert rc.copied_from is good_ens  # the COPY was used
    assert good_ens.learned == []  # stored ensembles never learn
    assert ddd.pool.get(good["id"])["ensemble"].copied_from is rc


# --- prequential order --------------------------------------------------------
def test_all_predictions_before_any_learning():
    log = []
    ddd, *_ = make_ddd(log=log)
    ddd.pool.add(FakeEnsemble(0, log), 0)
    for t in range(8):
        if t == 2:
            ddd.signal.fire_at.add(2)
        log.clear()
        ddd.step(t, FakeInst(0))
        kinds = [k for k, _ in log]
        first_learn = kinds.index("learn")
        assert "predict" not in kinds[first_learn:], (t, log)


def test_old_high_learns_with_low_lambda_after_drift():
    ddd, *_ = make_ddd()
    enter_after(ddd)
    oh = ddd.old_high
    ddd.step(1, FakeInst(0))
    assert oh.learned[-1] == ddd.cfg["lambda_low"]
    assert ddd.new_high.learned[-1] == ddd.cfg["lambda_high"]


def test_signal_fed_only_after_warmup():
    cfg = {**config.DDD, "min_steps": 5, "ref_window": 10}
    signal = StubSignal()
    ddd = ExtendedDDD(cfg, FakeFactory(), signal, min_training_size=3)
    for t in range(5):
        ddd.step(t, FakeInst(0))
    assert [t for t, _ in signal.seen] == [3, 4]
