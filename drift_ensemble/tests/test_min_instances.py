"""Tuning phase 2: the cluster's shared minimum (GatedDetector,
gate_alarms, the phase-2 task list and the selection of M)."""

import json

import numpy as np
import pandas as pd
import pytest

from drift_ensemble import config
from drift_ensemble import min_instances as p2
from drift_ensemble.detectors import GatedDetector, gate_alarms, own_minimum
from drift_ensemble.ensemble import reset_detector
from drift_ensemble.tests.fakes import StubDetector


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


def test_own_minimum_and_fitting_minimums():
    grid = [50, 100, 200, 2000]
    assert own_minimum("STEPD", {"window_size": 100}) == 200
    assert own_minimum("SEED", {"block_size": 32}) == 64
    assert own_minimum("RDDM", {"min_n_instances": 500}) == 500
    assert own_minimum("HDDMWeighted", {"lambda_": 0.05}) == 0
    assert p2.fitting_minimums("STEPD", {"window_size": 50}, grid) == [
        100, 200, 2000]
    assert p2.fitting_minimums("RDDM", {"min_n_instances": 200}, grid) == [
        200]
    assert p2.fitting_minimums("HDDMWeighted", {}, grid) == grid


def test_tasks_cover_each_cluster_with_fitting_configs():
    tasks = p2.all_tasks()
    assert len(tasks) == len(set(tasks))
    for cluster, name, pjson in tasks:
        assert name in config.CLUSTER_MEMBERS[cluster]
        assert p2.fitting_minimums(name, json.loads(pjson))
    rddm = [json.loads(p) for c, n, p in tasks if n == "RDDM"]
    if rddm:  # min_n_instances only takes the grid's values
        assert ({p["min_n_instances"] for p in rddm}
                == set(config.MIN_INSTANCES_GRID))
        levels = {p["drift_level"] for p in rddm}
        assert len(rddm) == len(levels) * len(config.MIN_INSTANCES_GRID)


# ---------------------------------------------------------------------------
# Selection (no JVM)
# ---------------------------------------------------------------------------
def _rows(cluster, m, name, params, f1, precision=0.5):
    return [{"cluster": cluster, "min_instances": m, "detector": name,
             "params": json.dumps(params), "generator": g, "mode": cluster,
             "seed": 1, "f1": f1, "f2": f1, "precision": precision,
             "recall": f1, "mean_delay": 10.0, "far": 0.0, "delay_auc": f1}
            for g in ("A", "B")]


def test_selection_picks_the_m_with_the_best_mean_of_both_members():
    rows = []
    for cluster, (a, b) in config.CLUSTER_MEMBERS.items():
        # M=100: a 0.9 / b 0.1 -> 0.5;  M=500: a 0.6 / b 0.6 -> 0.6
        rows += _rows(cluster, 100, a, {"x": 1}, 0.9)
        rows += _rows(cluster, 100, b, {"y": 1}, 0.1)
        rows += _rows(cluster, 500, a, {"x": 2}, 0.6)
        rows += _rows(cluster, 500, a, {"x": 3}, 0.2)  # worse config
        rows += _rows(cluster, 500, b, {"y": 2}, 0.6)
        rows += _rows(cluster, 1000, a, {"x": 4}, 0.99)  # b has none
    chosen, summary = p2.select_min_instances(pd.DataFrame(rows), "f1")
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
            rows += _rows(cluster, m, a, {"x": m}, 0.5)
            rows += _rows(cluster, m, b, {"y": m}, 0.5)
    chosen, _ = p2.select_min_instances(pd.DataFrame(rows), "f1")
    assert set(chosen["min_instances"].values()) == {200}


def test_only_the_two_vote_methods_use_phase_2():
    from drift_ensemble.workflow import method_params

    phase1, phase2 = {"phase": 1}, {"phase": 2}
    used = {m: method_params(m, phase1, phase2)["phase"]
            for m in config.METHODS}
    assert used == {
        "single_STEPD": 1, "single_SEED": 1, "single_HDDMWeighted": 1,
        "single_RDDM": 1, "ens_raw_resetall": 1,
        "ens_cluster_vote": 2, "ens_cluster_sens": 2}


def test_min_instances_for_reads_the_cluster_minimum():
    from drift_ensemble.workflow import CLUSTER_OF, min_instances_for

    name = config.DETECTOR_ORDER[0]
    params = {"min_instances": {CLUSTER_OF[name]: 300}}
    assert min_instances_for(name, params) == 300
    assert min_instances_for(name, {}) == 0  # phase-1 dict: no gate


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
