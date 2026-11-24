"""Tuning: config sanity and the detector screening."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from drift_ensemble import config, tuning
from drift_ensemble.detectors import iter_configs
from drift_ensemble.tuning import params_key


# ===========================================================================
# Config
# ===========================================================================
def test_tuning_and_test_seeds_disjoint():
    assert not set(config.TUNING_SEEDS) & set(config.TEST_SEEDS)


def test_grids_cover_the_study_detectors():
    assert set(config.DETECTOR_ORDER) == set(config.TUNING_GRIDS)
    assert config.TUNE_OBJECTIVE == "tf2"


# ===========================================================================
# Selection: configs and clusters by TF2 (no JVM)
# ===========================================================================
def _study_rows(scores):
    """scores: {(mode, detector): [(params, tf2, precision), ...]}"""
    rows = []
    for (mode, det), configs in scores.items():
        for params, score, precision in configs:
            rows += _rows(det, params, mode, [score, score], precision)
    return pd.DataFrame(rows)


def test_clusters_are_the_top_two_by_tf2_and_may_share_a_detector():
    from drift_ensemble.runs import members, methods, params_for

    scores = {
        ("ABRUPT", "HDDMWeighted"): [({"a": 1}, 0.6, 0.5),
                                     ({"a": 2}, 0.1, 0.9)],
        ("ABRUPT", "RDDM"): [({"r": 1}, 0.5, 0.5)],
        ("ABRUPT", "STEPD"): [({"s": 1}, 0.4, 0.5)],
        ("ABRUPT", "SEED"): [({"e": 1}, 0.3, 0.5)],
        ("GRADUAL", "HDDMWeighted"): [({"a": 3}, 0.7, 0.5)],
        # STEPD and SEED tie on TF2: SEED wins on precision
        ("GRADUAL", "STEPD"): [({"s": 2}, 0.5, 0.4)],
        ("GRADUAL", "SEED"): [({"e": 2}, 0.5, 0.6)],
        ("GRADUAL", "RDDM"): [({"r": 2}, 0.2, 0.5)],
    }
    selected, summary = tuning.select_params(_study_rows(scores))
    assert selected["ABRUPT"]["HDDMWeighted"] == {"a": 1}  # best TF2
    clusters, summary = tuning.select_clusters(summary)
    assert clusters == {"ABRUPT": ["HDDMWeighted", "RDDM"],
                        "GRADUAL": ["SEED", "HDDMWeighted"]}
    g = summary[summary["mode"] == "GRADUAL"].set_index("detector")
    assert g["rank"].to_dict() == {"HDDMWeighted": 1, "SEED": 2,
                                   "STEPD": 3, "RDDM": 4}
    assert g["in_cluster"].sum() == 2

    selected["clusters"] = clusters
    assert list(members(selected)) == [
        "HDDMWeighted_ABRUPT", "RDDM_ABRUPT",
        "SEED_GRADUAL", "HDDMWeighted_GRADUAL"]
    assert params_for("HDDMWeighted_ABRUPT", selected) == {"a": 1}
    assert params_for("HDDMWeighted_GRADUAL", selected) == {"a": 3}
    assert methods(selected) == [
        "single_HDDMWeighted_ABRUPT", "single_RDDM_ABRUPT",
        "single_SEED_GRADUAL", "single_HDDMWeighted_GRADUAL",
        "ens_raw_resetall", "ens_cluster_vote", "ens_cluster_sens"]


def test_grid_sizes():
    import math

    sizes = {name: math.prod(len(v) for v in grid.values())
             for name, grid in config.TUNING_GRIDS.items()}
    assert sizes == {"STEPD": 36, "SEED": 700, "HDDMWeighted": 50,
                     "RDDM": 56}


# ===========================================================================
# Screening
#
# Screening: task lists, selection of the new detectors and the summary,
# on fake rows (no JVM).
# ===========================================================================
STUDY = {mode: {d: {"p": i} for i, d in enumerate(config.DETECTOR_ORDER)}
         for mode in config.PURE_MODES}


def test_screening_grid_sizes():
    sizes = {d: len(iter_configs(d, g))
             for d, g in config.SCREENING_GRIDS.items()}
    # KSWIN: 5 x 4 x 3 = 60, minus (window 100, stat 100) x 5 alphas
    assert sizes == {"DDM": 49, "HDDMAverage": 10, "ADWIN": 7, "KSWIN": 55}


def test_every_grid_includes_the_capymoa_default():
    grids = {**config.TUNING_GRIDS, **config.SCREENING_GRIDS}
    for det, grid in grids.items():
        assert config.DEFAULT_PARAMS[det] in iter_configs(det, grid), det
    # RDDM's default (129, 2.258) is off its grid -> appended once
    rddm = iter_configs("RDDM")
    assert len(rddm) == 7 * 8 + 1
    assert rddm[-1] == config.DEFAULT_PARAMS["RDDM"]


def test_tuning_tasks_are_grid_plus_study_configs_without_duplicates():
    tasks = tuning.screening_tuning_tasks(STUDY)
    # 121 grid configs (defaults in grid) + 4 study configs (same per mode)
    # + the study's 4 defaults
    assert len(tasks) == len(set(tasks)) == 121 + 4 + 4
    assert ("STEPD", params_key({"p": 0})) in tasks
    assert ("RDDM", params_key(config.DEFAULT_PARAMS["RDDM"])) in tasks


def test_only_tasks_drops_configs_no_longer_in_a_grid():
    tasks = tuning.screening_tuning_tasks(STUDY)
    rows = (_rows("ADWIN", {"delta": 0.1}, "ABRUPT", [0.4])
            + _rows("ADWIN", {"delta": 1e-6}, "ABRUPT", [0.9]))  # dropped
    kept = tuning.only_tasks(pd.DataFrame(rows), tasks)
    assert kept["params"].tolist() == [params_key({"delta": 0.1})]


def _rows(det, params, mode, scores, precision=0.5):
    return [{"detector": det, "params": params_key(params), "mode": mode,
             "generator": "SEA", "seed": i, "tf2": f, "precision": precision,
             "recall": f, "ndt": 0.5, "mean_delay": 100.0 * (i + 1),
             "far": 0.0}
            for i, f in enumerate(scores)]


def test_all_selected_picks_best_mean_tf2_for_new_detectors():
    rows = []
    for mode in config.PURE_MODES:
        for det in tuning.SCREENING_DETECTORS:
            rows += _rows(det, {"x": 1}, mode, [0.2, 0.4])
            rows += _rows(det, {"x": 2}, mode,
                          [0.9, 0.1] if mode == "ABRUPT" else [0.1, 0.1])
    sel = tuning.screening_selected(pd.DataFrame(rows), STUDY)
    assert sel["ABRUPT"]["DDM"] == {"x": 2}  # mean 0.5 > 0.3
    assert sel["GRADUAL"]["ADWIN"] == {"x": 1}
    assert sel["ABRUPT"]["SEED"] == STUDY["ABRUPT"]["SEED"]
    # 4 study configs + 2 configs x 4 new detectors + 8 defaults
    assert len(tuning.screening_test_tasks(sel)) == 4 + 2 * 4 + 8


def test_summarize_means_and_order():
    sel = {mode: {**STUDY[mode], **{d: {"x": 1}
                                    for d in tuning.SCREENING_DETECTORS}}
           for mode in config.PURE_MODES}
    tune, test = [], []
    for mode in config.PURE_MODES:
        for k, det in enumerate(config.DETECTOR_ORDER
                                + tuning.SCREENING_DETECTORS):
            p = sel[mode][det]
            tune += _rows(det, p, mode, [0.5, 0.7])
            test += _rows(det, p, mode, [0.1 * k, 0.1 * k + 0.2, 0.9])
    s = tuning.screening_summary(pd.DataFrame(tune), pd.DataFrame(test), sel)
    assert len(s) == 2 * 2 * 8  # mode x setting x detector
    assert list(s["setting"].unique()) == ["tuned", "default"]
    assert s[s["setting"] == "default"]["n_test"].eq(0).all()  # no rows
    a = s[(s["mode"] == "ABRUPT")
          & (s["setting"] == "tuned")].reset_index(drop=True)
    assert a.loc[0, "detector"] == "KSWIN"  # highest mean test TF2
    assert list(s["mode"].unique()) == ["ABRUPT", "GRADUAL"]
    row = a.set_index("detector").loc["STEPD"]
    assert row["tuning_tf2"] == pytest.approx(0.6)
    assert row["test_tf2"] == pytest.approx(1.1 / 3)  # mean of 0, 0.2, 0.9
    assert row["test_ndt"] == pytest.approx(0.5)
    assert row["test_precision"] == pytest.approx(0.5)
    assert row["test_delay"] == pytest.approx(200.0)  # median
    assert bool(row["in_study"]) and json.loads(row["params"]) == {"p": 0}
