"""Detectors and ensembles: the ensembles' state machines (StubDetector
fakes, no JVM), the full reset, mid-stream thresholds and the phase-2 gate
(some with a JVM).
"""

from __future__ import annotations

import numpy as np
import pytest

from drift_ensemble import config
from drift_ensemble.detectors import (
    ClusterSensitivityEnsemble,
    ClusterVoteEnsemble,
    gate_alarms,
    GatedDetector,
    get_threshold,
    make_detector,
    RawResetAllEnsemble,
    reset_detector,
    set_threshold,
    threshold_for_level,
    THRESHOLD_TARGETS,
)


# ===========================================================================
# Fakes
#
# Duck-typed fakes for JVM-free unit tests of the detector ensembles.
# ===========================================================================
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


# ===========================================================================
# RawResetAllEnsemble
#
# RawResetAllEnsemble with stub detectors (no JVM). The cluster ensembles
# have their own sections below.
# ===========================================================================
ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]


def test_raw_resetall_ensemble():
    detectors = {name: StubDetector({"STEPD": [5, 20],
                                     "SEED": [5]}.get(name, ()))
                 for name in ORDER}
    ens = RawResetAllEnsemble(detectors, ORDER)
    for t in range(30):
        ens.update(t, 0)
    # Every fire instant is an alarm and each alarm resets ALL detectors.
    assert ens.alarms == [5, 20]
    assert [f["detector"] for f in ens.raw_fires] == ["STEPD", "SEED",
                                                      "STEPD"]
    assert all(d.reset_count == 2 for d in detectors.values())


# ===========================================================================
# ClusterVoteEnsemble
#
# Unit tests for ClusterVoteEnsemble (no JVM; StubDetector fakes).
#
# StubDetector fires on scripted add_element call indices; the loops below
# call update(t, 0) once per t, so call index == t.
# ===========================================================================
CLUSTERS = {"SEED": "ABRUPT", "RDDM": "ABRUPT",
            "STEPD": "GRADUAL", "HDDMWeighted": "GRADUAL"}
WINDOW = 250
STEP = 0.1
W_INIT = 0.75


def make_vote(fire_at: dict[str, tuple]) -> ClusterVoteEnsemble:
    return ClusterVoteEnsemble(
        detectors={n: StubDetector(fire_at.get(n, ())) for n in ORDER},
        clusters=dict(CLUSTERS),
        order=ORDER,
        window=WINDOW,
        w_init=W_INIT,
        w_max=1.0,
        w_floor=0.5,
        step=STEP,
    )


def run(ens, until: int) -> None:
    for t in range(until):
        ens.update(t, 0)


def test_single_vote_no_alarm_even_after_expiry():
    ens = make_vote({"SEED": (100,)})
    run(ens, 500)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] is None  # expired
    assert all(w == W_INIT for w in ens.weights.values())


def test_pair_within_window_alarms_and_moves_weights():
    ens = make_vote({"SEED": (100,), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    delta = STEP * (1 - 50 / WINDOW)  # 0.08
    assert abs(ens.weights["SEED"] - (W_INIT + delta)) < 1e-12
    assert abs(ens.weights["RDDM"] - (W_INIT - delta)) < 1e-12
    # the other cluster is untouched
    assert ens.weights["STEPD"] == ens.weights["HDDMWeighted"] == W_INIT


def test_pair_outside_window_does_not_alarm():
    ens = make_vote({"SEED": (100,), "RDDM": (351,)})  # dt = 251 > WINDOW
    run(ens, 400)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] == ("RDDM", 351)  # became the new vote
    assert all(w == W_INIT for w in ens.weights.values())


def test_same_instant_pair_is_tie():
    ens = make_vote({"SEED": (100,), "RDDM": (100,)})
    run(ens, 150)
    assert ens.alarms == [100]
    assert all(w == W_INIT for w in ens.weights.values())


def test_same_instant_pair_with_earlier_pending_rewards_first():
    ens = make_vote({"SEED": (100, 150), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    delta = STEP * (1 - 50 / WINDOW)
    assert abs(ens.weights["SEED"] - (W_INIT + delta)) < 1e-12
    assert abs(ens.weights["RDDM"] - (W_INIT - delta)) < 1e-12


def test_full_weight_detector_alarms_alone_and_floor_holds():
    # Three pairings with dt=25 (delta 0.09) push SEED to the 1.0 cap and
    # RDDM onto the 0.5 floor; then SEED alarms alone.
    ens = make_vote({"SEED": (100, 300, 500, 700),
                    "RDDM": (125, 325, 525)})
    run(ens, 800)
    assert ens.alarms == [125, 325, 525, 700]
    assert ens.weights["SEED"] == 1.0
    assert ens.weights["RDDM"] == 0.5
    # the solo alarm changed no weights
    last_logged = [r for r in ens.weights_log if r["t"] == 700]
    assert last_logged == []


def test_alarm_resets_all_detectors_and_clears_other_cluster_pending():
    ens = make_vote({"STEPD": (90,), "SEED": (100,), "RDDM": (120,),
                    "HDDMWeighted": (130,)})
    run(ens, 350)  # before the 130-vote expires at t > 380
    assert ens.alarms == [120]
    assert all(d.reset_count == 1 for d in ens.detectors.values())
    # STEPD's pre-alarm vote was cleared; HDDM's post-alarm fire is the
    # only live vote until it expires.
    assert ens.pending["GRADUAL"] == ("HDDMWeighted", 130)


def test_vote_cross_cluster_fires_never_pair():
    ens = make_vote({"SEED": (100,), "STEPD": (120,)})
    run(ens, 200)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] == ("SEED", 100)
    assert ens.pending["GRADUAL"] == ("STEPD", 120)


# ===========================================================================
# ClusterSensitivityEnsemble
#
# Unit tests for ClusterSensitivityEnsemble and threshold_for_level
# (no JVM; StubDetector fakes + a recording apply_fn).
#
# StubDetector fires on scripted add_element call indices; the loops below
# call update(t, 0) once per t, so call index == t.
# ===========================================================================
BASE = {"STEPD": 0.01, "SEED": 0.01, "HDDMWeighted": 0.001, "RDDM": 2.0}
LATE = 1.0
L_MAX = 3.0


def fake_threshold(name, base, level):
    return (name, base, level)  # traceable stand-in


def make_sens(fire_at: dict[str, tuple]):
    applied = []
    ens = ClusterSensitivityEnsemble(
        detectors={n: StubDetector(fire_at.get(n, ())) for n in ORDER},
        clusters=dict(CLUSTERS),
        order=ORDER,
        base_thresholds=dict(BASE),
        window=WINDOW,
        late_step=LATE,
        l_max=L_MAX,
        threshold_fn=fake_threshold,
        apply_fn=lambda name, det, value: applied.append((name, value)),
    )
    return ens, applied


def test_late_voter_becomes_more_sensitive_first_untouched():
    ens, applied = make_sens({"SEED": (100,), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    assert ens.levels["RDDM"] == pytest.approx(LATE * 50 / WINDOW)  # 0.2
    assert ens.levels["SEED"] == 0.0
    assert applied == [("RDDM", ("RDDM", 2.0, ens.levels["RDDM"]))]
    assert ens.sens_log == [{"t": 150, "detector": "RDDM",
                             "level": ens.levels["RDDM"],
                             "threshold": applied[0][1], "reason": "late"}]
    # other cluster untouched
    assert ens.levels["STEPD"] == ens.levels["HDDMWeighted"] == 0.0


def test_same_instant_tie_changes_nothing():
    ens, applied = make_sens({"SEED": (100,), "RDDM": (100,)})
    run(ens, 500)
    assert ens.alarms == [100]
    assert applied == [] and ens.sens_log == []


def test_same_instant_pair_with_earlier_pending_rewards_partner():
    ens, _ = make_sens({"SEED": (100, 150), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    assert ens.levels["RDDM"] == pytest.approx(LATE * 50 / WINDOW)
    assert ens.levels["SEED"] == 0.0


def test_expired_lone_vote_changes_nothing():
    # No stricter rule: an unconfirmed vote just expires.
    ens, applied = make_sens({"STEPD": (100,)})
    run(ens, 400)
    assert ens.alarms == []
    assert ens.pending["GRADUAL"] is None
    assert all(lv == 0.0 for lv in ens.levels.values())
    assert applied == [] and ens.sens_log == []


def test_vote_wiped_by_other_cluster_alarm_changes_nothing():
    # STEPD votes, then an ABRUPT pair alarms and resets everything.
    ens, _ = make_sens({"STEPD": (90,), "SEED": (100,), "RDDM": (120,)})
    run(ens, 600)
    assert ens.alarms == [120]
    assert ens.levels["STEPD"] == 0.0
    assert all(d.reset_count == 1 for d in ens.detectors.values())
    assert all(p is None for p in ens.pending.values())


def test_pair_outside_window_does_not_alarm_or_adjust():
    ens, _ = make_sens({"SEED": (100,), "RDDM": (351,)})  # gap 251
    run(ens, 400)
    assert ens.alarms == []
    assert all(lv == 0.0 for lv in ens.levels.values())
    assert ens.pending["ABRUPT"] == ("RDDM", 351)


def test_levels_only_rise_and_cap_at_l_max():
    # RDDM confirms SEED a full window late 5 times: +1.0 each -> capped at 3.
    seed_fires = tuple(range(100, 100 + 5 * 600, 600))
    rddm_fires = tuple(t + WINDOW for t in seed_fires)
    ens, applied = make_sens({"SEED": seed_fires, "RDDM": rddm_fires})
    run(ens, 100 + 5 * 600)
    assert ens.alarms == list(rddm_fires)
    assert ens.levels["RDDM"] == L_MAX
    assert ens.levels["SEED"] == 0.0
    assert len(applied) == 3  # the 4th and 5th pairings hit the cap
    levels = [r["level"] for r in ens.sens_log]
    assert levels == sorted(levels)  # monotone non-decreasing


def test_sens_cross_cluster_fires_never_pair():
    ens, _ = make_sens({"SEED": (100,), "STEPD": (120,)})
    run(ens, 200)
    assert ens.alarms == []
    assert ens.pending == {"ABRUPT": ("SEED", 100),
                           "GRADUAL": ("STEPD", 120)}
    assert all(lv == 0.0 for lv in ens.levels.values())


# ---------------------------------------------------------------------------
# threshold_for_level (driven by config.SENS)
# ---------------------------------------------------------------------------
def test_threshold_level_zero_is_base():
    for name, base in BASE.items():
        assert threshold_for_level(name, base, 0.0) == pytest.approx(base)


def test_threshold_mul_and_cap():
    f = config.SENS["SCALE"]["STEPD"][2]
    assert threshold_for_level("STEPD", 0.01, 1.0) == pytest.approx(0.01 * f)
    assert threshold_for_level("STEPD", 0.01, -2.0) == pytest.approx(
        0.01 / f ** 2)
    assert threshold_for_level("STEPD", 0.3, 3.0) == config.SENS["PROB_MAX"]


def test_threshold_rddm_add_and_floor():
    a = config.SENS["SCALE"]["RDDM"][2]
    assert threshold_for_level("RDDM", 2.0, 2.0) == pytest.approx(2.0 + 2 * a)
    assert threshold_for_level("RDDM", 2.0, -1.0) > 2.0  # stricter
    assert threshold_for_level("RDDM", 1.2, 3.0) == \
        config.SENS["RDDM_LEVEL_MIN"]


def test_positive_level_is_always_more_sensitive():
    # mul thresholds rise (higher p-value/delta/confidence = more alarms);
    # RDDM's drift_level falls.
    for name in ORDER:
        lo = threshold_for_level(name, BASE[name], -1.0)
        hi = threshold_for_level(name, BASE[name], 1.0)
        if name == "RDDM":
            assert hi < BASE[name] < lo
        else:
            assert lo < BASE[name] < hi


# ===========================================================================
# Full reset
#
# reset_detector (used by ens_raw_resetall and the cluster ensembles) must
# really reset the MOA detector state: capymoa's own reset() only clears
# Python-side flags (regression test: without resetLearning() a reset-all
# ensemble behaved exactly like one without resets).
# ===========================================================================
def _fires(with_reset: bool) -> list[int]:
    from capymoa.drift.detectors import RDDM

    det = RDDM()
    fires = []
    seq = [0.0] * 2000 + [1.0] * 500 + [0.0] * 2000 + [1.0] * 500
    for t, v in enumerate(seq):
        det.add_element(v)
        if det.detected_change():
            fires.append(t)
        if t == 2100 and with_reset:
            reset_detector(det)
    return fires


@pytest.mark.jvm
def test_reset_detector_changes_moa_state():
    assert _fires(False) != _fires(True)


# ===========================================================================
# Mid-stream thresholds
#
# set_threshold really changes a live MOA detector's drift threshold
# mid-stream, and the change survives self-resets and reset_detector (guards
# the capymoa 0.9.0 / MOA private-field quirk used by ens_cluster_sens).
# ===========================================================================
SWITCH = 2000
# name -> (strict params, threshold kwarg, sensitive value)
CASES = {
    "STEPD": (dict(window_size=30, alpha_drift=1e-5), "alpha_drift", 0.05),
    "SEED": (dict(delta=1e-6, epsilon_prime=0.0025, block_size=32,
                  alpha=0.5), "delta", 0.05),
    "HDDMWeighted": (dict(drift_confidence=1e-6, lambda_=0.05,
                          test_type="One-sided"), "drift_confidence", 0.05),
    "RDDM": (dict(min_n_instances=30, drift_level=3.0), "drift_level", 1.5),
}


def _errors():
    rng = np.random.default_rng(1)
    p = np.r_[np.full(4000, .20), np.full(4000, .26),
              np.full(4000, .20), np.full(4000, .26)]
    return (rng.random(len(p)) < p).astype(float)


def _run(det, errors, hook=None) -> list[int]:
    alarms = []
    for t, e in enumerate(errors):
        if t == SWITCH and hook is not None:
            hook(det)
        det.add_element(float(e))
        if det.detected_change():
            alarms.append(t)
    return alarms


def _after(alarms):
    return sum(t >= SWITCH for t in alarms)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_set_threshold_updates_option_and_field(name):
    params, _, value = CASES[name]
    det = make_detector(name, params)
    set_threshold(det, name, value)
    assert get_threshold(det, name) == pytest.approx(value)
    option_name = THRESHOLD_TARGETS[name][0]
    assert float(str(det.get_params()[option_name])) == pytest.approx(value)
    # survives self-resets after alarms and an explicit full reset
    _run(det, _errors())
    assert get_threshold(det, name) == pytest.approx(value)
    reset_detector(det)
    assert get_threshold(det, name) == pytest.approx(value)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_strict_to_sensitive_mid_stream(name):
    params, _, value = CASES[name]
    errors = _errors()
    strict = _run(make_detector(name, params), errors)
    switched = _run(make_detector(name, params), errors,
                    lambda d: set_threshold(d, name, value))
    assert _after(switched) > _after(strict)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_sensitive_to_strict_mid_stream(name):
    params, key, value = CASES[name]
    errors = _errors()
    strict_value = params[key]
    sensitive = {**params, key: value}
    switched = _run(make_detector(name, sensitive), errors,
                    lambda d: set_threshold(d, name, strict_value))
    assert _after(switched) <= 1


# ===========================================================================
# The cluster's shared-minimum gate (tuning phase 2)
# ===========================================================================
def _replay(det, errors):
    alarms = []
    for t, e in enumerate(errors):
        if e < 0:
            continue
        det.add_element(float(e))
        if det.detected_change():
            alarms.append(t)
    return alarms


# ---------------------------------------------------------------------------
# GatedDetector / gate_alarms (no JVM)
# ---------------------------------------------------------------------------
def test_gate_drops_fires_before_the_minimum():
    # call indices 0-based: the n-th instance fed is index n-1
    det = GatedDetector(StubDetector(fire_at=[49, 99, 150, 260, 400]), 100)
    fired = [i for i in range(500)
             if (det.add_element(0.0), det.detected_change())[1]]
    # 49: 50 seen < 100 -> dropped; 99: 100 seen -> kept (gate restarts);
    # 150: 51 since -> dropped; 260: 161 -> kept; 400: 140 -> kept
    assert fired == [99, 260, 400]


def test_reset_restarts_the_gate():
    stub = StubDetector(fire_at=[120, 150])
    det = GatedDetector(stub, 100)
    for _ in range(100):
        det.add_element(0.0)
    reset_detector(det)  # stub has no moa_detector: Python reset only
    assert stub.reset_count == 1
    fired = [i for i in range(100, 300)
             if (det.add_element(0.0), det.detected_change())[1]]
    assert fired == []  # 120 is 21 after the reset, 150 is 51 after


def test_gate_alarms_matches_a_gated_replay():
    rng = np.random.default_rng(3)
    errors = np.concatenate([np.full(1000, -1), rng.integers(0, 2, 5000)])
    fire_at = sorted(rng.choice(5000, size=120, replace=False).tolist())
    ungated = _replay(StubDetector(fire_at), errors)
    for m in (0, 1, 37, 100, 500):
        gated = _replay(GatedDetector(StubDetector(fire_at), m), errors) \
            if m else ungated
        assert gate_alarms(ungated, errors, m) == gated


# ---------------------------------------------------------------------------
# Selection (no JVM)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real detectors (JVM)
# ---------------------------------------------------------------------------
@pytest.mark.jvm
@pytest.mark.parametrize("name,params,m", [
    ("STEPD", {"window_size": 50, "alpha_drift": 0.01}, 300),
    ("SEED", {"delta": 0.1, "epsilon_prime": 0.01, "block_size": 32,
              "alpha": 0.6}, 2500),
    ("HDDMWeighted", {"drift_confidence": 0.01, "lambda_": 0.05,
                      "test_type": "One-sided"}, 500),
    # (in phase 2 RDDM's min_n_instances = M, so its gate never drops a
    # fire; a smaller min_n here just exercises the gate)
    ("RDDM", {"min_n_instances": 30, "drift_level": 1.75}, 1000),
])
def test_gate_alarms_equals_gated_run_of_a_real_detector(name, params, m):
    from drift_ensemble.detectors import run_detector_on_errors

    rng = np.random.default_rng(7)
    rates = np.repeat([0.1, 0.4, 0.15, 0.5, 0.2], 2000)
    errors = np.concatenate([np.full(500, -1),
                             (rng.random(rates.size) < rates).astype(int)])
    ungated = run_detector_on_errors(name, params, errors)
    gated = run_detector_on_errors(name, params, errors, m)
    assert len(gated) < len(ungated)  # the gate really drops some fires
    assert gate_alarms(ungated, errors, m) == gated


@pytest.mark.jvm
def test_gated_detector_passes_moa_through():
    from drift_ensemble.detectors import (
        get_threshold,
        make_detector,
        set_threshold,
    )

    det = GatedDetector(make_detector("STEPD", {"window_size": 30,
                                                "alpha_drift": 0.003}), 100)
    set_threshold(det, "STEPD", 0.05)
    assert get_threshold(det, "STEPD") == pytest.approx(0.05)
    for _ in range(50):
        det.add_element(1.0)
    reset_detector(det)  # Python reset + MOA resetLearning, gate restarted
    assert det.n_seen == 0
    assert get_threshold(det, "STEPD") == pytest.approx(0.05)
