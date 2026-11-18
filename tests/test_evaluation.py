import math

import pytest

from drift_ensemble.evaluation import (
    compute_metrics,
    delay_curve,
    match_alarms,
    per_type,
)


def gt(*drifts):
    """drifts: (start, width, type)"""
    return [{"start": s, "end": s + w, "type": ty,
             "function_before": 1, "function_after": 2}
            for s, w, ty in drifts]


TRUTH = gt((1000, 0, "ABRUPT"), (3000, 0, "ABRUPT"))


def test_tp_with_delay():
    m = match_alarms([1100], TRUTH, max_delay=1000)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)
    assert m.delays == [100]


def test_alarm_at_exact_start_and_window_edge():
    m = match_alarms([1000, 4000], TRUTH, max_delay=1000)
    assert (m.tp, m.fp, m.fn) == (2, 0, 0)
    assert m.delays == [0, 1000]


def test_alarm_just_outside_window_is_fp():
    m = match_alarms([999, 4001], TRUTH, max_delay=1000)
    assert (m.tp, m.fp, m.fn) == (0, 2, 2)


def test_duplicates_counted_as_fp_or_ignored():
    m = match_alarms([1100, 1200, 1300], TRUTH, max_delay=1000,
                     duplicates_as_fp=True)
    assert (m.tp, m.fp, m.fn) == (1, 2, 1)
    assert m.duplicate_alarms == [1200, 1300]
    m2 = match_alarms([1100, 1200, 1300], TRUTH, max_delay=1000,
                      duplicates_as_fp=False)
    assert (m2.tp, m2.fp, m2.fn) == (1, 0, 1)


def test_fn_when_no_alarm():
    m = match_alarms([], TRUTH, max_delay=1000)
    assert (m.tp, m.fp, m.fn) == (0, 0, 2)


def test_gradual_window_includes_width():
    truth = gt((1000, 500, "GRADUAL"))
    # window = [1000, 1500 + max_delay]
    m = match_alarms([2400], truth, max_delay=1000)
    assert m.tp == 1 and m.delays == [1400]
    m2 = match_alarms([2501], truth, max_delay=1000)
    assert m2.tp == 0 and m2.fp == 1


def test_overlapping_windows_assert():
    truth = gt((1000, 0, "ABRUPT"), (1500, 0, "ABRUPT"))
    with pytest.raises(AssertionError):
        match_alarms([1100], truth, max_delay=1000)


def test_compute_metrics():
    r = compute_metrics([1100, 500], TRUTH, max_delay=1000,
                        n_instances=11000, min_training_size=1000)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["f1"] == pytest.approx(0.5)
    # F2 weights recall higher: (1+4)*P*R / (4P + R)
    assert r["f2"] == pytest.approx(5 * 0.25 / (2 + 0.5))
    assert r["mean_delay"] == 100.0
    assert r["far"] == pytest.approx(1 / 10.0)  # 1 FP per 10k monitored


def test_metrics_no_alarms():
    r = compute_metrics([], TRUTH, 1000, 11000, 1000)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    assert math.isnan(r["mean_delay"])


def test_delay_curve():
    # Single drift, alarm 150 after the start: detected for m >= 200 only.
    truth = gt((1000, 0, "ABRUPT"))
    dc = delay_curve([1150], truth, delay_grid=[100, 200, 300])
    recalls = [p["recall"] for p in dc["points"]]
    assert recalls == [0.0, 1.0, 1.0]
    assert dc["delay_auc"] == pytest.approx(2 / 3)


def test_per_type():
    truth = gt((1000, 0, "ABRUPT"), (3000, 500, "GRADUAL"),
               (6000, 0, "ABRUPT"))
    res = per_type([1100, 3600], truth, max_delay=1000)
    assert res["ABRUPT"]["n_drifts"] == 2
    assert res["ABRUPT"]["recall"] == 0.5
    assert res["ABRUPT"]["mean_delay"] == 100.0
    assert res["GRADUAL"]["recall"] == 1.0
    assert res["GRADUAL"]["mean_delay"] == 600.0
