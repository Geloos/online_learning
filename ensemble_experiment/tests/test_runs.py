"""RECURRING streams, and full runs with capymoa (JVM).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import streams
from ensemble_experiment import config
from ensemble_experiment.runs import run_specs


# ===========================================================================
# RECURRING streams
#
# The shared streams.py as used by the DDD experiment: the RECURRING mode,
# concept ids, recurrence flags and the run grid.
# ===========================================================================
SIZES = config.STREAM_SIZES


def plan(generator, mode, seed, **overrides):
    return streams.plan_stream(generator, mode, seed,
                               **{**SIZES, **overrides})


@pytest.mark.parametrize("gen", config.GENERATORS)
def test_recurring_plan_concepts_recur(gen):
    p = plan(gen, "RECURRING", 201)
    fns = p["functions"]
    assert len(set(p["concepts"])) == config.RECURRING_N_CONCEPTS
    assert set(fns) <= set(p["concepts"])
    assert all(fns[i] != fns[i + 1] for i in range(len(fns) - 1))
    assert all(f in streams.FUNCTIONS[gen] for f in fns)
    # some concept comes back after a different one
    assert any(fns[j] == fns[i] for i in range(len(fns))
               for j in range(i + 2, len(fns)))
    gt = p["ground_truth"]
    assert all(gt[i]["function_after"] == gt[i + 1]["function_before"]
               for i in range(len(gt) - 1))


def test_recurring_plan_deterministic_and_seed_dependent():
    a = plan("Agrawal", "RECURRING", 205)
    assert a == plan("Agrawal", "RECURRING", 205)
    assert a != plan("Agrawal", "RECURRING", 206)


def test_recurring_widths_half_half_and_same_layout_as_mixed():
    p = plan("SEA", "RECURRING", 203)
    types = [d["type"] for d in p["ground_truth"]]
    n = SIZES["n_drifts"]
    assert types.count("ABRUPT") == n // 2
    assert types.count("GRADUAL") == n - n // 2
    ref = plan("SEA", "MIXED", 203)
    assert set(p) == set(ref) | {"concepts"}
    assert set(p["ground_truth"][0]) == set(ref["ground_truth"][0])
    pos = 0
    for d, w in zip(p["ground_truth"], p["widths"]):
        pos += SIZES["drift_every_n"]
        assert d["start"] == pos and d["end"] == pos + w
        pos += w
    assert p["length"] == pos + SIZES["drift_every_n"]


def test_stagger_uses_all_three_functions_as_concepts():
    assert sorted(plan("STAGGER", "RECURRING", 201)["concepts"]) == [1, 2, 3]


def test_recurring_needs_a_valid_concept_count():
    with pytest.raises(ValueError):
        plan("STAGGER", "RECURRING", 1, n_concepts=4)
    with pytest.raises(ValueError):
        plan("SEA", "RECURRING", 1, n_concepts=None)


def test_concept_ids_switch_points():
    p = plan("SEA", "RECURRING", 202)
    ids = streams.concept_ids(p)
    assert len(ids) == p["length"]
    assert ids[0] == p["functions"][0]
    for d in p["ground_truth"]:
        switch = d["start"] + (d["end"] - d["start"]) // 2
        assert ids[switch - 1] == d["function_before"]
        assert ids[switch] == d["function_after"]


def test_mark_recurring_flags():
    p = {"functions": [1, 2, 1, 3, 2],
         "ground_truth": [{"function_after": f} for f in [2, 1, 3, 2]]}
    assert [d["recurring"] for d in streams.mark_recurring(p)] == [
        False, True, False, True]


def test_pure_modes_ignore_n_concepts():
    a = plan("Agrawal", "GRADUAL", 201)
    b = plan("Agrawal", "GRADUAL", 201, n_concepts=None)
    assert a == b and "concepts" not in a
    assert len(a["ground_truth"]) == SIZES["n_drifts"]


def test_run_specs_grid_and_seed_disjointness():
    specs = run_specs()
    assert len(specs) == (len(config.GENERATORS) * len(config.MODES)
                          * len(config.SEEDS) * len(config.METHODS))
    assert len(set(specs)) == len(specs)
    assert "ens_raw" not in config.METHODS


def test_matching_windows_do_not_overlap_on_experiment_streams():
    from drift_ensemble.stats import match_alarms

    for mode in config.MODES:
        p = plan("SEA", mode, 201)
        match_alarms([], p["ground_truth"], config.MAX_DELAY)  # asserts
        assert np.all(np.diff([d["start"] for d in p["ground_truth"]]) > 0)


# ===========================================================================
# With a JVM
#
# Integration tests that need capymoa and a JVM.
# ===========================================================================
def _cluster_params(m=200):
    """A fixed cluster_params.json-style dict (capymoa defaults, shared
    minimum m), so these tests don't depend on a tuning run."""
    from drift_ensemble import config as base

    out = {"min_instances": {c: m for c in base.CLUSTER_MEMBERS}}
    for c, members in base.CLUSTER_MEMBERS.items():
        out[c] = {n: dict(base.DEFAULT_PARAMS[n]) for n in members}
        if "RDDM" in out[c]:
            out[c]["RDDM"]["min_n_instances"] = m
    return out


def _instances(n=600, seed=201):
    from streams import make_stream

    stream, _, _ = make_stream("Agrawal", "ABRUPT", seed, n_drifts=1,
                               drift_every_n=n, gradual_width=100)
    return stream.get_schema(), [stream.next_instance() for _ in range(n)]


def _votes(tree, insts):
    return [list(tree.getVotesForInstance(i.java_instance)) for i in insts]


@pytest.mark.jvm
def test_frozen_copy_is_independent():
    from ensemble_experiment.models import EnsembleFactory

    schema, insts = _instances()
    ens = EnsembleFactory(schema, config.DDD, seed=1)()
    for inst in insts[:300]:
        ens.learn(inst, 1.0)
    copy = ens.frozen_copy()
    probe = insts[300:350]
    before = [_votes(t, probe) for t in ens.trees]
    assert [_votes(t, probe) for t in copy.trees] == before
    for inst in insts[300:]:
        copy.learn(inst, 1.0)
    assert [_votes(t, probe) for t in ens.trees] == before
    assert [_votes(t, probe) for t in copy.trees] != before


@pytest.mark.jvm
def test_trees_use_nb_adaptive_leaves_and_grace_period():
    from ensemble_experiment.models import EnsembleFactory

    schema, _ = _instances(10)
    ens = EnsembleFactory(schema, config.DDD, seed=1)()
    assert len(ens.trees) == config.DDD["n_learners"]
    opts = {str(o.getName()): str(o.getValueAsCLIString())
            for o in ens.trees[0].getOptions().getOptionArray()}
    assert opts["leafprediction"] == "NBAdaptive"
    assert opts["gracePeriod"] == str(config.DDD["grace_period"])


@pytest.mark.jvm
def test_ensemble_predicts_after_learning_and_is_reproducible():
    from ensemble_experiment.models import EnsembleFactory

    schema, insts = _instances(300)

    def run():
        ens = EnsembleFactory(schema, config.DDD, seed=7)()
        assert ens.predict(insts[0]) is None  # nothing learned yet
        for inst in insts[:200]:
            ens.learn(inst, 1.0)
        return [ens.predict(i) for i in insts[200:]]

    preds = run()
    assert all(p in (0, 1) for p in preds)
    assert run() == preds


def _phase1_params():
    """A fixed selected_params.json-style dict (no shared minimum)."""
    p = _cluster_params()
    del p["min_instances"]
    return p


# (not ens_raw_resetall: STEPD with capymoa's default params fires every
# 60 instances on an all-zero error stream, a MOA STEPD edge case)
@pytest.mark.parametrize("method", ["single_HDDMWeighted",
                                    "ens_cluster_sens"])
@pytest.mark.jvm
def test_signal_fires_and_resets(method):
    from drift_ensemble.runs import method_params
    from ensemble_experiment.models import make_signal

    sig = make_signal(method, method_params(method, _phase1_params(),
                                            _cluster_params()))
    errs = [0] * 3000 + [1] * 1000
    fired = [t for t, e in enumerate(errs) if sig.update(t, e)]
    assert fired, "no alarm on a 0 -> 1 error jump"
    assert fired[0] >= 3000
    assert sig.alarms == fired
    sig.reset()  # must not raise; statistics wiped
    assert not any(sig.update(4000 + t, 0) for t in range(200))


@pytest.mark.jvm
def test_smoke_run_writes_all_files(tmp_path):
    from ensemble_experiment.runs import run_one

    out = run_one("Agrawal", "RECURRING", 201, "ens_cluster_sens",
                  selected_params=_cluster_params(),
                  overrides={"n_drifts": 2, "drift_every_n": 1000,
                             "gradual_width": 200},
                  out_root=tmp_path, min_training_size=300, log_every=50)
    for name in ("steps.csv", "episodes.csv", "correct.npy", "alarms.json",
                 "meta.json"):
        assert (out / name).exists(), name
    meta = json.loads((out / "meta.json").read_text())
    s = meta["summary"]
    assert 0.0 <= s["acc"] <= 1.0
    assert meta["length"] == 3200
    assert len(np.load(out / "correct.npy")) == 3200
    assert meta["detector_params"] == _cluster_params()
    # re-run is skipped (resumable)
    mtime = (out / "meta.json").stat().st_mtime
    run_one("Agrawal", "RECURRING", 201, "ens_cluster_sens",
            overrides={"n_drifts": 2, "drift_every_n": 1000,
                       "gradual_width": 200},
            out_root=tmp_path, min_training_size=300, log_every=50)
    assert (out / "meta.json").stat().st_mtime == mtime


def test_each_method_reads_only_its_own_params_file(tmp_path, monkeypatch):
    from ensemble_experiment import config, models

    phase1 = tmp_path / "selected_params.json"
    phase1.write_text('{"phase": 1}')
    monkeypatch.setattr(config, "SELECTED_PARAMS_PATH", phase1)
    monkeypatch.setattr(config, "CLUSTER_PARAMS_PATH",
                        tmp_path / "cluster_params.json")  # not there
    assert models.load_method_params("ens_raw_resetall") == {"phase": 1}
    assert models.load_method_params("single_SEED") == {"phase": 1}
    with pytest.raises(FileNotFoundError, match="phase2"):
        models.load_method_params("ens_cluster_vote")
