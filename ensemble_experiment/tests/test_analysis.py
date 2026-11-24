"""Statistics of the synthetic runs and of the real-data runs (fake run
folders, no JVM, no download).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ensemble_experiment import analysis, config
from ensemble_experiment import runs as ee_runs


# ===========================================================================
# Synthetic streams
#
# analysis.py: recovery after real drifts, median accuracy curves, and the
# statistics folder on fake run folders.
# ===========================================================================
# --- drift_bounds / recovery_instances -----------------------------------------
def test_drift_bounds_sorted_with_next_start():
    gt = [{"start": 600, "end": 700}, {"start": 100, "end": 100}]
    assert analysis.drift_bounds(gt, 1000) == [(100, 100, 600),
                                               (600, 700, 1000)]


def test_recovery_counts_accuracy_since_detection():
    c = np.ones(1000)
    c[11:41] = 0  # 30 wrong right after the detection at 10
    # accuracy since detection after n instances = (n - 30) / n >= 0.9
    # first at n = 300
    assert analysis.recovery_instances(c, 10, 1000, 0.9, 50) == 300


def test_recovery_not_checked_before_min_instances():
    # all right: 100% after 1 instance, but only checked from 50 on
    assert analysis.recovery_instances(np.ones(1000), 10, 1000, 0.95,
                                       50) == 50


def test_recovery_none_when_never_reached_before_next_drift():
    assert analysis.recovery_instances(np.zeros(1000), 10, 1000, 0.5,
                                       50) is None
    # fewer than min_instances before the next drift
    assert analysis.recovery_instances(np.ones(1000), 10, 40, 0.5,
                                       50) is None


# --- median curves ------------------------------------------------------------
def test_block_acc_non_overlapping_blocks():
    out = analysis.block_acc([1, 0, 1, 1, 0], 2)  # partial last block dropped
    assert out.tolist() == [0.5, 1.0]


def test_median_blocks_stream_is_blockwise_median():
    runs = [np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]),
            np.array([1, 0, 1, 1, 1, 1])]  # longer run is cut to 4
    assert analysis.median_blocks_stream(runs, 2).tolist() == [0.5, 1.0]


def test_median_blocks_aligned_uses_each_drift_in_range():
    a = np.ones(100)
    a[50:60] = 0
    b = np.ones(100)
    b[20:30] = 0
    # drifts at 50 (run a) and 20 (run b) line up at offset 0; the drift at
    # 95 does not fit the window and is skipped. Blocks of 5 from -10..+20.
    curve, n = analysis.median_blocks_aligned([(a, [50, 95]), (b, [20])],
                                              5, 10, 20)
    assert n == 2
    assert curve.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_same_positions():
    gt1 = [{"start": 10, "end": 10}]
    gt2 = [{"start": 10, "end": 20}]
    assert analysis._same_positions([(None, {"ground_truth": gt1})] * 2)
    assert not analysis._same_positions([(None, {"ground_truth": gt1}),
                                         (None, {"ground_truth": gt2})])


# --- fake run folders -------------------------------------------------------------
LEN = 12_000
GT = [{"start": 2000, "end": 2000, "type": "ABRUPT", "recurring": False},
      {"start": 6000, "end": 7000, "type": "GRADUAL", "recurring": False},
      {"start": 9000, "end": 9000, "type": "ABRUPT", "recurring": True}]
GT_SHIFTED = [{**g, "start": g["start"] + 500, "end": g["end"] + 500}
              for g in GT]
BEST = config.STATS["method"]


def _write_run(root, method, correct, episodes, seed=201, gt=GT):
    d = root / f"SEA,MIXED,{seed},{method}"
    d.mkdir(parents=True)
    np.save(d / "correct.npy", correct.astype(np.int8))
    pd.DataFrame(episodes, columns=["t_detect", "label", "drift_idx"]
                 ).to_csv(d / "episodes.csv", index=False)  # header always
    (d / "meta.json").write_text(json.dumps({
        "generator": "SEA", "mode": "MIXED", "seed": seed, "method": method,
        "length": LEN, "ground_truth": gt}))


def _ep(t, label, k):
    return {"t_detect": t, "label": label, "drift_idx": k}


@pytest.fixture
def fake_runs(tmp_path):
    runs = tmp_path / "runs"
    # The best method detects all 3 drifts.
    a = np.ones(LEN)
    a[2000:2101] = 0      # drift 0 at 2000, detected at 2040
    a[6500:6600] = 0      # inside drift 1's transition, detected at 6500
    a[9000::2] = 0        # after drift 2 it stays at 0.5: never recovers
    _write_run(runs, BEST, a, [
        _ep(2040, "TP", 0),
        _ep(4000, "FP", None),  # a false alarm: never used
        _ep(6500, "TP", 1),
        _ep(9010, "TP", 2),
    ])
    # Another method: misses drifts 0 and 2, ends every concept at 0.8.
    b = np.ones(LEN)
    b[np.arange(0, LEN, 5)] = 0
    _write_run(runs, "single_SEED", b, [_ep(6800, "TP", 1)])
    # A second run of the best method whose drifts sit 500 later.
    _write_run(runs, BEST, np.ones(LEN), [], seed=202, gt=GT_SHIFTED)
    return tmp_path


def _drifts(fake_runs, method=BEST):
    episodes, metas = analysis.load_runs(fake_runs / "runs", method)
    return analysis.drift_table(metas, episodes, config.RECOVERY, warmup=1000)


def test_drift_table_real_drifts_only(fake_runs):
    d = _drifts(fake_runs)
    assert len(d) == 6  # 2 runs x 3 real drifts; the FP is not a drift
    a = d[d["seed"] == 201].set_index("drift")
    # before = the stable period before the drift: warm-up end (1000) ..
    # 2000, then 2000 (drift 0 end) .. 6000 with 101 wrong, then 7000 .. 9000
    assert a["acc_before"].tolist() == pytest.approx([1.0, 3899 / 4000, 1.0])
    assert a.loc[0, "delay"] == 40
    assert a.loc[0, "acc_drift_to_detect"] == 0.0  # 2000..2040 all wrong
    # abrupt: 60 wrong after the detection (2041..2100); accuracy since the
    # detection (n - 60) / n >= 0.95 first at n = 1200
    assert a.loc[0, "rec_instances"] == 1200
    # gradual: 99 wrong after the detection (6501..6599); target 0.95 x
    # 3899/4000; (n - 99) / n reaches it first at n = 1339
    assert a.loc[1, "rec_instances"] == 1339
    # stays at 0.5 after drift 2: never back to 0.95
    assert a.loc[2, "acc_drift_to_detect"] == pytest.approx(5 / 11)
    assert np.isnan(a.loc[2, "rec_instances"])
    assert a["recovered"].tolist() == [True, True, False]
    b = _drifts(fake_runs, "single_SEED").set_index("drift")
    assert not b.loc[0, "detected"] and np.isnan(b.loc[0, "rec_instances"])
    assert b.loc[0, "recovered"] is None


def test_mode_summary_medians(fake_runs):
    s = analysis.mode_summary(_drifts(fake_runs))
    assert s["mode"].tolist() == ["MIXED"]
    r = s.iloc[0]
    assert r["drifts"] == 6 and r["detected"] == 3
    assert r["acc_before_median"] == 1.0  # of 1.0, 0.975, 1.0
    # drift -> detection: 0.0, 500/501, 5/11
    assert r["acc_drift_to_detect_median"] == pytest.approx(5 / 11)
    assert r["rec_instances_median"] == pytest.approx((1200 + 1339) / 2)
    assert r["recovered_pct"] == pytest.approx(200 / 3)


def test_run_analysis_writes_statistics(fake_runs):
    out = fake_runs / "statistics"
    s = analysis.run_analysis(fake_runs / "runs", out)
    assert (out / "recovery.csv").exists()
    assert (out / "accuracy_200_MIXED.png").exists()  # drift-aligned branch
    assert not (out / "accuracy_200_ABRUPT.png").exists()  # no ABRUPT runs
    drifts = pd.read_csv(out / "drifts.csv")
    assert set(drifts["method"]) == {BEST}
    assert s.iloc[0]["detected"] == 3


# ===========================================================================
# Real datasets
#
# Run list and statistics, on fake run folders (no JVM, no
# download).
# ===========================================================================
def test_one_run_per_dataset_and_method():
    specs = ee_runs.real_run_specs(["Electricity"])
    assert specs == [("Electricity", config.REAL_SEED, "ens_raw_resetall"),
                     ("Electricity", config.REAL_SEED, "no_change")]


def _fake_run(root, method, correct, alarms=None):
    d = root / ee_runs.real_run_name("Electricity", config.REAL_SEED, method)
    d.mkdir(parents=True)
    c = np.asarray(correct, dtype=np.int8)
    np.save(d / "correct.npy", c)
    if alarms is not None:
        (d / "alarms.json").write_text(json.dumps({"alarms": alarms}))
    (d / "meta.json").write_text(json.dumps({
        "dataset": "Electricity", "seed": config.REAL_SEED, "method": method,
        "length": len(c),
        "summary": {"acc": float(c[config.MIN_TRAINING_SIZE:].mean()),
                    "acc_all": float(c.mean()),
                    "n_detections": len(alarms or [])}}))


def test_statistics_table_and_files(tmp_path):
    runs_dir, out = tmp_path / "runs", tmp_path / "out"
    n = config.MIN_TRAINING_SIZE + 1000
    _fake_run(runs_dir, "ens_raw_resetall", [1] * (n - 500) + [0] * 500,
              alarms=[1500, 1700])
    _fake_run(runs_dir, "no_change", [0] * n)
    table = analysis.real_statistics(runs_dir, out).set_index("method")
    assert table.loc["ens_raw_resetall", "acc"] == pytest.approx(0.5)
    assert table.loc["ens_raw_resetall", "n_detections"] == 2
    assert table.loc["no_change", "acc"] == 0.0
    assert {p.name for p in (out / "Electricity").iterdir()} == {
        "accuracy.csv", "accuracy_over_time.png"}


def test_every_real_dataset_is_fully_specified():
    assert set(config.REAL_DATASETS) == {
        "Electricity", "Airlines", "Covtype", "Poker",  # OpenML
        "INSECTS_abrupt", "INSECTS_gradual", "INSECTS_incr_reoccurring",
        "NOAA", "Rialto", "GasSensor", "Posture"}  # USP, in data/
    for name, spec in config.REAL_DATASETS.items():
        assert {"file", "length", "md5", "acc_block"} <= set(spec) <= {
            "file", "length", "md5", "acc_block", "url"}
        assert spec["length"] > config.MIN_TRAINING_SIZE
        assert len(spec["md5"]) == 32
