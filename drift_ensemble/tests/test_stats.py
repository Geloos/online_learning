"""Scoring (alarm <-> drift matching) and the statistics.
"""

from __future__ import annotations

import json
import math

import pytest

from drift_ensemble import config, stats
from drift_ensemble.runs import load_runs
from drift_ensemble.stats import (
    compute_metrics,
    delay_curve,
    match_alarms,
    per_type,
)


# ===========================================================================
# Scoring
# ===========================================================================
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


# ===========================================================================
# Statistics
#
# stats.py: per-run metrics, delays, medians, delay curves and the
# statistics folder, on fake run folders (no JVM).
# ===========================================================================
# One abrupt drift at 2000 and one gradual drift over [5000, 7000].
RUN_TRUTH = [{"start": 2000, "end": 2000, "type": "ABRUPT"},
         {"start": 5000, "end": 7000, "type": "GRADUAL"}]


def _write_run(root, seed, alarms, method="single_SEED", mode="MIXED"):
    d = root / f"SEA,{mode},{seed},{method}"
    d.mkdir(parents=True)
    (d / "alarms.json").write_text(json.dumps({"alarms": alarms}))
    meta = {"method": method, "generator": "SEA", "mode": mode,
            "seed": seed, "sid": f"SEA,{mode},{seed}", "length": 10_000,
            "runtime_s": 0.1, "n_alarms": len(alarms),
            "ground_truth": RUN_TRUTH,
            "config": {"MIN_TRAINING_SIZE": 1000}}
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def runs(tmp_path):
    # seed 1: both drifts detected (delays 100 and 1000) + one FP at 9000
    #         -> P 2/3, R 1, F1 0.8
    # seed 2: abrupt drift detected (delay 300), gradual missed
    #         -> P 1, R 0.5, F1 2/3
    _write_run(tmp_path, 1, [2100, 6000, 9000])
    _write_run(tmp_path, 2, [2300])
    return load_runs(tmp_path)


def test_run_table(runs):
    df = stats.run_table(runs).set_index("seed")
    assert df.loc[1, ["tp", "fp", "fn"]].tolist() == [2, 1, 0]
    assert df.loc[1, "precision"] == pytest.approx(2 / 3)
    assert df.loc[1, "f1"] == pytest.approx(0.8)
    assert df.loc[2, ["tp", "fp", "fn"]].tolist() == [1, 0, 1]
    assert df.loc[2, "recall"] == pytest.approx(0.5)


def test_delay_table_has_only_detected_drifts(runs):
    df = stats.delay_table(runs)
    assert sorted(zip(df["seed"], df["type"], df["delay"])) == [
        (1, "ABRUPT", 100), (1, "GRADUAL", 1000), (2, "ABRUPT", 300)]


def test_method_summary_means_and_median_delay(runs):
    s = stats.method_summary(stats.run_table(runs), stats.delay_table(runs))
    assert len(s) == 1
    row = s.iloc[0]
    assert (row["mode"], row["method"], row["n_runs"]) == (
        "MIXED", "single_SEED", 2)
    assert (row["tp"], row["fp"], row["fn"]) == (3, 1, 1)
    assert row["precision_mean"] == pytest.approx((2 / 3 + 1) / 2)
    assert row["recall_mean"] == pytest.approx(0.75)
    assert row["f1_mean"] == pytest.approx((0.8 + 2 / 3) / 2)
    # sample std of two values a, b is |a - b| / sqrt(2)
    assert row["recall_std"] == pytest.approx(0.5 / 2 ** 0.5)
    assert row["n_detected"] == 3
    assert row["delay_median"] == 300  # delays 100, 1000, 300


def test_mean_differs_from_median_on_a_failed_generator(tmp_path):
    # 3 runs, one failing (F1 0): the mean counts the failure, a median
    # of [0, 0.8, 0.8] would not.
    for seed, alarms in [(1, [2100, 6000, 9000]), (2, [2100, 6000, 9000]),
                         (3, [])]:
        _write_run(tmp_path, seed, alarms)
    runs = load_runs(tmp_path)
    s = stats.method_summary(stats.run_table(runs), stats.delay_table(runs))
    assert s.iloc[0]["f1_mean"] == pytest.approx(1.6 / 3)


def test_delay_curve_zero_and_full_window(runs):
    curve = stats.delay_curve_table(runs, grid=[0, config.MAX_DELAY])
    at = curve.set_index("max_delay")
    # max delay 0: the abrupt drift needs an alarm exactly at 2000 (missed);
    # the gradual drift is still hit inside its transition (6000, seed 1).
    assert at.loc[0, "recall"] == pytest.approx((0.5 + 0.0) / 2)
    # at the full window the curve equals the scored metrics
    runs_df = stats.run_table(runs)
    for metric in stats.METRICS:
        assert at.loc[config.MAX_DELAY, metric] == pytest.approx(
            runs_df[metric].mean())


def test_run_statistics_writes_every_file(tmp_path):
    test_dir, out = tmp_path / "test", tmp_path / "out"
    for seed, alarms in [(1, [2100, 6000, 9000]), (2, [2300])]:
        _write_run(test_dir, seed, alarms)
        _write_run(test_dir, seed, alarms, mode="ABRUPT",
                   method="ens_raw_resetall")
    summary = stats.run_statistics(test_dir, out)
    assert len(summary) == 2
    names = {p.relative_to(out).as_posix() for p in out.rglob("*")
             if p.is_file()}
    assert {"runs.csv", "delays.csv", "metrics.csv",
            "delay_curve.csv"} <= names
    # ABRUPT has only an ensemble run, MIXED only a single-detector run
    assert {"ABRUPT/delay_curve.png", "ABRUPT/ensembles.png",
            "MIXED/delay_curve.png", "MIXED/singles.png"} <= names
    assert "ABRUPT/singles.png" not in names
    assert "MIXED/ensembles.png" not in names
    assert not (out / "GRADUAL").exists()  # no runs of that mode


def test_run_statistics_without_runs(tmp_path):
    assert stats.run_statistics(tmp_path / "none", tmp_path / "out").empty
