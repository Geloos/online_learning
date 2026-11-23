"""real.py: run list and statistics, on fake run folders (no JVM, no
download)."""

import json

import numpy as np
import pytest

from ensemble_experiment import config, real


def test_one_run_per_dataset_and_method():
    specs = real.run_specs(["Electricity"])
    assert specs == [("Electricity", config.REAL_SEED, "ens_raw_resetall"),
                     ("Electricity", config.REAL_SEED, "no_change")]


def _fake_run(root, method, correct, alarms=None):
    d = root / real.run_name("Electricity", config.REAL_SEED, method)
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
    table = real.real_statistics(runs_dir, out).set_index("method")
    assert table.loc["ens_raw_resetall", "acc"] == pytest.approx(0.5)
    assert table.loc["ens_raw_resetall", "n_detections"] == 2
    assert table.loc["no_change", "acc"] == 0.0
    assert {p.name for p in (out / "Electricity").iterdir()} == {
        "accuracy.csv", "accuracy_over_time.png"}


def test_every_real_dataset_is_fully_specified():
    assert set(config.REAL_DATASETS) == {"Electricity", "Airlines",
                                         "Covtype", "Poker"}
    for name, spec in config.REAL_DATASETS.items():
        assert set(spec) == {"file", "length", "url", "md5", "acc_block"}
        assert spec["length"] > config.MIN_TRAINING_SIZE
        assert len(spec["md5"]) == 32
