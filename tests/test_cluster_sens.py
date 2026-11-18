"""Unit tests for ClusterSensitivityEnsemble and threshold_for_level
(no JVM; StubDetector fakes + a recording apply_fn).

StubDetector fires on scripted add_element call indices; the loops below
call update(t, 0) once per t, so call index == t.
"""

import pytest

from drift_ensemble import config
from drift_ensemble.detectors import threshold_for_level
from drift_ensemble.ensemble import ClusterSensitivityEnsemble
from tests.fakes import StubDetector

ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]
CLUSTERS = {"SEED": "ABRUPT", "RDDM": "ABRUPT",
            "STEPD": "GRADUAL", "HDDMWeighted": "GRADUAL"}
BASE = {"STEPD": 0.01, "SEED": 0.01, "HDDMWeighted": 0.001, "RDDM": 2.0}
WINDOW = 250
LATE = 1.0
L_MAX = 3.0


def fake_threshold(name, base, level):
    return (name, base, level)  # traceable stand-in


def make_ens(fire_at: dict[str, tuple]):
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


def run(ens, until: int) -> None:
    for t in range(until):
        ens.update(t, 0)


def test_late_voter_becomes_more_sensitive_first_untouched():
    ens, applied = make_ens({"SEED": (100,), "RDDM": (150,)})
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
    ens, applied = make_ens({"SEED": (100,), "RDDM": (100,)})
    run(ens, 500)
    assert ens.alarms == [100]
    assert applied == [] and ens.sens_log == []


def test_same_instant_pair_with_earlier_pending_rewards_partner():
    ens, _ = make_ens({"SEED": (100, 150), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    assert ens.levels["RDDM"] == pytest.approx(LATE * 50 / WINDOW)
    assert ens.levels["SEED"] == 0.0


def test_expired_lone_vote_changes_nothing():
    # No stricter rule: an unconfirmed vote just expires.
    ens, applied = make_ens({"STEPD": (100,)})
    run(ens, 400)
    assert ens.alarms == []
    assert ens.pending["GRADUAL"] is None
    assert all(lv == 0.0 for lv in ens.levels.values())
    assert applied == [] and ens.sens_log == []


def test_vote_wiped_by_other_cluster_alarm_changes_nothing():
    # STEPD votes, then an ABRUPT pair alarms and resets everything.
    ens, _ = make_ens({"STEPD": (90,), "SEED": (100,), "RDDM": (120,)})
    run(ens, 600)
    assert ens.alarms == [120]
    assert ens.levels["STEPD"] == 0.0
    assert all(d.reset_count == 1 for d in ens.detectors.values())
    assert all(p is None for p in ens.pending.values())


def test_pair_outside_window_does_not_alarm_or_adjust():
    ens, _ = make_ens({"SEED": (100,), "RDDM": (351,)})  # gap 251
    run(ens, 400)
    assert ens.alarms == []
    assert all(lv == 0.0 for lv in ens.levels.values())
    assert ens.pending["ABRUPT"] == ("RDDM", 351)


def test_levels_only_rise_and_cap_at_l_max():
    # RDDM confirms SEED a full window late 5 times: +1.0 each -> capped at 3.
    seed_fires = tuple(range(100, 100 + 5 * 600, 600))
    rddm_fires = tuple(t + WINDOW for t in seed_fires)
    ens, applied = make_ens({"SEED": seed_fires, "RDDM": rddm_fires})
    run(ens, 100 + 5 * 600)
    assert ens.alarms == list(rddm_fires)
    assert ens.levels["RDDM"] == L_MAX
    assert ens.levels["SEED"] == 0.0
    assert len(applied) == 3  # the 4th and 5th pairings hit the cap
    levels = [r["level"] for r in ens.sens_log]
    assert levels == sorted(levels)  # monotone non-decreasing


def test_cross_cluster_fires_never_pair():
    ens, _ = make_ens({"SEED": (100,), "STEPD": (120,)})
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
