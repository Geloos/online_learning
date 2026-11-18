"""The shared streams.py as used by the DDD experiment: the RECURRING mode,
concept ids, recurrence flags and the run grid."""

import numpy as np
import pytest

from ensemble_experiment import config
from ensemble_experiment.runner import run_specs
import streams

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
    from drift_ensemble.evaluation import match_alarms

    for mode in config.MODES:
        p = plan("SEA", mode, 201)
        match_alarms([], p["ground_truth"], config.MAX_DELAY)  # asserts
        assert np.all(np.diff([d["start"] for d in p["ground_truth"]]) > 0)
