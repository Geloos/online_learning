"""stats.py: per-run metrics, delays, medians, delay curves and the
statistics folder, on fake run folders (no JVM)."""

import json

import pytest

from drift_ensemble import config, stats
from drift_ensemble.analysis import load_runs

# One abrupt drift at 2000 and one gradual drift over [5000, 7000].
TRUTH = [{"start": 2000, "end": 2000, "type": "ABRUPT"},
         {"start": 5000, "end": 7000, "type": "GRADUAL"}]


def _write_run(root, seed, alarms, method="single_SEED", mode="MIXED"):
    d = root / f"SEA,{mode},{seed},{method}"
    d.mkdir(parents=True)
    (d / "alarms.json").write_text(json.dumps({"alarms": alarms}))
    meta = {"method": method, "generator": "SEA", "mode": mode,
            "seed": seed, "sid": f"SEA,{mode},{seed}", "length": 10_000,
            "runtime_s": 0.1, "n_alarms": len(alarms),
            "ground_truth": TRUTH,
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


def test_method_summary_medians(runs):
    s = stats.method_summary(stats.run_table(runs), stats.delay_table(runs))
    assert len(s) == 1
    row = s.iloc[0]
    assert (row["mode"], row["method"], row["n_runs"]) == (
        "MIXED", "single_SEED", 2)
    assert (row["tp"], row["fp"], row["fn"]) == (3, 1, 1)
    assert row["precision"] == pytest.approx((2 / 3 + 1) / 2)
    assert row["recall"] == pytest.approx(0.75)
    assert row["f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert row["delay"] == 300  # median over the 3 detected drifts


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
            runs_df[metric].median())


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
