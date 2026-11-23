"""screening.py: task lists, selection of the new detectors and the summary,
on fake rows (no JVM)."""

import json

import pandas as pd
import pytest

from drift_ensemble import config, screening
from drift_ensemble.detectors import iter_configs
from drift_ensemble.tuning import params_key

STUDY = {mode: {d: {"p": i} for i, d in enumerate(config.DETECTOR_ORDER)}
         for mode in config.PURE_MODES}


def test_screening_grid_sizes():
    sizes = {d: len(iter_configs(d, g))
             for d, g in config.SCREENING_GRIDS.items()}
    assert sizes == {"DDM": 49, "HDDMAverage": 16, "ADWIN": 10}


def test_every_grid_includes_the_capymoa_default():
    grids = {**config.TUNING_GRIDS, **config.SCREENING_GRIDS}
    for det, grid in grids.items():
        assert config.DEFAULT_PARAMS[det] in iter_configs(det, grid), det
    # RDDM's default (129, 2.258) is off its grid -> appended once
    rddm = iter_configs("RDDM")
    assert len(rddm) == 7 * 8 + 1
    assert rddm[-1] == config.DEFAULT_PARAMS["RDDM"]


def test_tuning_tasks_are_grid_plus_study_configs_without_duplicates():
    tasks = screening.tuning_tasks(STUDY)
    # 75 grid configs (defaults in grid) + 4 study configs (same per mode)
    # + the study's 4 defaults
    assert len(tasks) == len(set(tasks)) == 75 + 4 + 4
    assert ("STEPD", params_key({"p": 0})) in tasks
    assert ("RDDM", params_key(config.DEFAULT_PARAMS["RDDM"])) in tasks


def _rows(det, params, mode, f1s, precision=0.5):
    return [{"detector": det, "params": params_key(params), "mode": mode,
             "generator": "SEA", "seed": i, "f1": f, "precision": precision,
             "recall": f, "mean_delay": 100.0 * (i + 1), "f2": f,
             "far": 0.0, "delay_auc": f}
            for i, f in enumerate(f1s)]


def test_all_selected_picks_best_mean_f1_for_new_detectors():
    rows = []
    for mode in config.PURE_MODES:
        for det in screening.NEW_DETECTORS:
            rows += _rows(det, {"x": 1}, mode, [0.2, 0.4])
            rows += _rows(det, {"x": 2}, mode,
                          [0.9, 0.1] if mode == "ABRUPT" else [0.1, 0.1])
    sel = screening.all_selected(pd.DataFrame(rows), STUDY)
    assert sel["ABRUPT"]["DDM"] == {"x": 2}  # mean 0.5 > 0.3
    assert sel["GRADUAL"]["ADWIN"] == {"x": 1}
    assert sel["ABRUPT"]["SEED"] == STUDY["ABRUPT"]["SEED"]
    # 4 study configs + 2 configs x 3 new detectors + 7 defaults
    assert len(screening.test_tasks(sel)) == 4 + 2 * 3 + 7


def test_summarize_medians_and_order():
    sel = {mode: {**STUDY[mode], **{d: {"x": 1}
                                    for d in screening.NEW_DETECTORS}}
           for mode in config.PURE_MODES}
    tune, test = [], []
    for mode in config.PURE_MODES:
        for k, det in enumerate(config.DETECTOR_ORDER
                                + screening.NEW_DETECTORS):
            p = sel[mode][det]
            tune += _rows(det, p, mode, [0.5, 0.7])
            test += _rows(det, p, mode, [0.1 * k, 0.1 * k + 0.2, 0.9])
    s = screening.summarize(pd.DataFrame(tune), pd.DataFrame(test), sel)
    assert len(s) == 2 * 2 * 7  # mode x setting x detector
    assert list(s["setting"].unique()) == ["tuned", "default"]
    assert s[s["setting"] == "default"]["n_test"].eq(0).all()  # no rows
    a = s[(s["mode"] == "ABRUPT")
          & (s["setting"] == "tuned")].reset_index(drop=True)
    assert a.loc[0, "detector"] == "ADWIN"  # highest median test F1
    assert list(s["mode"].unique()) == ["ABRUPT", "GRADUAL"]
    row = a.set_index("detector").loc["STEPD"]
    assert row["tuning_f1"] == pytest.approx(0.6)
    assert row["test_f1"] == pytest.approx(0.2)  # median of 0, 0.2, 0.9
    assert row["test_delay"] == pytest.approx(200.0)
    assert bool(row["in_study"]) and json.loads(row["params"]) == {"p": 0}
