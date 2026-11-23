"""Tuning: config sanity, phase-2 tasks and selection, which tuning each
method uses, and the detector screening.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from drift_ensemble import config, tuning
from drift_ensemble.detectors import iter_configs, own_minimum
from drift_ensemble.tuning import params_key


# ===========================================================================
# Config
# ===========================================================================
def test_tuning_and_test_seeds_disjoint():
    assert not set(config.TUNING_SEEDS) & set(config.TEST_SEEDS)


def test_delay_grid():
    assert config.DELAY_GRID == [50, 100, 150, 200, 250, 300, 350, 400,
                                 450, 500]
    assert config.DELAY_GRID[-1] == config.MAX_DELAY


def test_clusters_cover_all_detectors():
    members = [m for ms in config.CLUSTER_MEMBERS.values() for m in ms]
    assert sorted(members) == sorted(config.DETECTOR_ORDER)
    assert set(members) == set(config.TUNING_GRIDS)


def test_grid_sizes():
    import math

    sizes = {name: math.prod(len(v) for v in grid.values())
             for name, grid in config.TUNING_GRIDS.items()}
    assert sizes == {"STEPD": 36, "SEED": 700, "HDDMWeighted": 50,
                     "RDDM": 56}


# ===========================================================================
# Phase 2
# ===========================================================================
# ---------------------------------------------------------------------------
# GatedDetector / gate_alarms (no JVM)
# ---------------------------------------------------------------------------


def test_own_minimum_and_fitting_minimums():
    grid = [50, 100, 200, 2000]
    assert own_minimum("STEPD", {"window_size": 100}) == 200
    assert own_minimum("SEED", {"block_size": 32}) == 64
    assert own_minimum("RDDM", {"min_n_instances": 500}) == 500
    assert own_minimum("HDDMWeighted", {"lambda_": 0.05}) == 0
    assert tuning.fitting_minimums("STEPD", {"window_size": 50}, grid) == [
        100, 200, 2000]
    assert tuning.fitting_minimums("RDDM", {"min_n_instances": 200}, grid) == [
        200]
    assert tuning.fitting_minimums("HDDMWeighted", {}, grid) == grid


def test_tasks_cover_each_cluster_with_fitting_configs():
    tasks = tuning.phase2_tasks()
    assert len(tasks) == len(set(tasks))
    for cluster, name, pjson in tasks:
        assert name in config.CLUSTER_MEMBERS[cluster]
        assert tuning.fitting_minimums(name, json.loads(pjson))
    rddm = [json.loads(p) for c, n, p in tasks if n == "RDDM"]
    if rddm:  # min_n_instances only takes the grid's values
        assert ({p["min_n_instances"] for p in rddm}
                == set(config.MIN_INSTANCES_GRID))
        levels = {p["drift_level"] for p in rddm}
        assert len(rddm) == len(levels) * len(config.MIN_INSTANCES_GRID)


# ---------------------------------------------------------------------------
# Selection (no JVM)
# ---------------------------------------------------------------------------
def _phase2_rows(cluster, m, name, params, f1, precision=0.5):
    return [{"cluster": cluster, "min_instances": m, "detector": name,
             "params": json.dumps(params), "generator": g, "mode": cluster,
             "seed": 1, "f1": f1, "f2": f1, "precision": precision,
             "recall": f1, "mean_delay": 10.0, "far": 0.0, "delay_auc": f1}
            for g in ("A", "B")]


def test_selection_picks_the_m_with_the_best_mean_of_both_members():
    rows = []
    for cluster, (a, b) in config.CLUSTER_MEMBERS.items():
        # M=100: a 0.9 / b 0.1 -> 0.5;  M=500: a 0.6 / b 0.6 -> 0.6
        rows += _phase2_rows(cluster, 100, a, {"x": 1}, 0.9)
        rows += _phase2_rows(cluster, 100, b, {"y": 1}, 0.1)
        rows += _phase2_rows(cluster, 500, a, {"x": 2}, 0.6)
        rows += _phase2_rows(cluster, 500, a, {"x": 3}, 0.2)  # worse config
        rows += _phase2_rows(cluster, 500, b, {"y": 2}, 0.6)
        rows += _phase2_rows(cluster, 1000, a, {"x": 4}, 0.99)  # b has none
    chosen, summary = tuning.select_min_instances(pd.DataFrame(rows), "f1")
    for cluster, (a, b) in config.CLUSTER_MEMBERS.items():
        assert chosen["min_instances"][cluster] == 500
        assert chosen[cluster] == {a: {"x": 2}, b: {"y": 2}}
        s = summary[summary["cluster"] == cluster]
        assert set(s["min_instances"]) == {100, 500}  # 1000 skipped
        assert s.loc[s["chosen"], "cluster_score"].iloc[0] == \
            pytest.approx(0.6)


def test_selection_tie_goes_to_precision_then_smaller_m():
    rows = []
    for cluster, (a, b) in config.CLUSTER_MEMBERS.items():
        for m in (200, 300):
            rows += _phase2_rows(cluster, m, a, {"x": m}, 0.5)
            rows += _phase2_rows(cluster, m, b, {"y": m}, 0.5)
    chosen, _ = tuning.select_min_instances(pd.DataFrame(rows), "f1")
    assert set(chosen["min_instances"].values()) == {200}


def test_only_the_two_vote_methods_use_phase_2():
    from drift_ensemble.runs import method_params

    phase1, phase2 = {"phase": 1}, {"phase": 2}
    used = {m: method_params(m, phase1, phase2)["phase"]
            for m in config.METHODS}
    assert used == {
        "single_STEPD": 1, "single_SEED": 1, "single_HDDMWeighted": 1,
        "single_RDDM": 1, "ens_raw_resetall": 1,
        "ens_cluster_vote": 2, "ens_cluster_sens": 2}


def test_min_instances_for_reads_the_cluster_minimum():
    from drift_ensemble.runs import CLUSTER_OF, min_instances_for

    name = config.DETECTOR_ORDER[0]
    params = {"min_instances": {CLUSTER_OF[name]: 300}}
    assert min_instances_for(name, params) == 300
    assert min_instances_for(name, {}) == 0  # phase-1 dict: no gate


# ---------------------------------------------------------------------------
# Real detectors (JVM)
# ---------------------------------------------------------------------------


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
    tasks = tuning.screening_tuning_tasks(STUDY)
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
        for det in tuning.SCREENING_DETECTORS:
            rows += _rows(det, {"x": 1}, mode, [0.2, 0.4])
            rows += _rows(det, {"x": 2}, mode,
                          [0.9, 0.1] if mode == "ABRUPT" else [0.1, 0.1])
    sel = tuning.screening_selected(pd.DataFrame(rows), STUDY)
    assert sel["ABRUPT"]["DDM"] == {"x": 2}  # mean 0.5 > 0.3
    assert sel["GRADUAL"]["ADWIN"] == {"x": 1}
    assert sel["ABRUPT"]["SEED"] == STUDY["ABRUPT"]["SEED"]
    # 4 study configs + 2 configs x 3 new detectors + 7 defaults
    assert len(tuning.screening_test_tasks(sel)) == 4 + 2 * 3 + 7


def test_summarize_medians_and_order():
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
