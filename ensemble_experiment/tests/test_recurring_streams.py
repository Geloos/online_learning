import numpy as np
import pytest

from ensemble_experiment import config
from ensemble_experiment.streams import (
    concept_ids,
    mark_recurring,
    plan_for,
    plan_recurring,
    run_specs,
)
from drift_ensemble.streams import plan_stream

OV = config.STREAM_OVERRIDES


@pytest.mark.parametrize("gen", config.GENERATORS)
def test_recurring_plan_concepts_recur(gen):
    p = plan_recurring(gen, 201, **OV)
    fns = p["functions"]
    assert len(set(p["concepts"])) == config.RECURRING_N_CONCEPTS
    assert set(fns) <= set(p["concepts"])
    assert all(fns[i] != fns[i + 1] for i in range(len(fns) - 1))
    assert all(f in config.FUNCTIONS[gen] for f in fns)
    # some concept comes back after a different one
    assert any(fns[j] == fns[i] for i in range(len(fns))
               for j in range(i + 2, len(fns)))
    gt = p["ground_truth"]
    assert all(gt[i]["function_after"] == gt[i + 1]["function_before"]
               for i in range(len(gt) - 1))


def test_recurring_plan_deterministic_and_seed_dependent():
    a = plan_recurring("Agrawal", 205, **OV)
    assert a == plan_recurring("Agrawal", 205, **OV)
    assert a != plan_recurring("Agrawal", 206, **OV)


def test_recurring_widths_half_half_and_layout_matches_plan_stream():
    p = plan_recurring("SEA", 203, **OV)
    types = [d["type"] for d in p["ground_truth"]]
    n = OV["n_drifts"]
    assert types.count("ABRUPT") == n // 2
    assert types.count("GRADUAL") == n - n // 2
    # same position arithmetic and keys as the detector study's plans
    ref = plan_stream("SEA", "MIXED", 203, **OV)
    assert set(p) >= set(ref)
    assert set(p["ground_truth"][0]) == set(ref["ground_truth"][0])
    pos = 0
    for d, w in zip(p["ground_truth"], p["widths"]):
        pos += OV["drift_every_n"]
        assert d["start"] == pos and d["end"] == pos + w
        pos += w
    assert p["length"] == pos + OV["drift_every_n"]


def test_stagger_uses_all_three_functions_as_concepts():
    p = plan_recurring("STAGGER", 201, **OV)
    assert sorted(p["concepts"]) == [1, 2, 3]


def test_too_many_concepts_rejected():
    with pytest.raises(ValueError):
        plan_recurring("STAGGER", 1, n_concepts=4, **OV)


def test_concept_ids_switch_points():
    p = plan_recurring("SEA", 202, **OV)
    ids = concept_ids(p)
    assert len(ids) == p["length"]
    assert ids[0] == p["functions"][0]
    for d in p["ground_truth"]:
        switch = d["start"] + (d["end"] - d["start"]) // 2
        assert ids[switch - 1] == d["function_before"]
        assert ids[switch] == d["function_after"]


def test_mark_recurring_flags():
    p = {"functions": [1, 2, 1, 3, 2],
         "ground_truth": [{"function_after": f} for f in [2, 1, 3, 2]]}
    assert [d["recurring"] for d in mark_recurring(p)] == [
        False, True, False, True]


def test_plan_for_delegates_pure_modes_with_overrides():
    p = plan_for("Agrawal", "GRADUAL", 201)
    assert p == plan_stream("Agrawal", "GRADUAL", 201, **OV)
    assert len(p["ground_truth"]) == OV["n_drifts"]


def test_run_specs_grid_and_seed_disjointness():
    specs = run_specs()
    assert len(specs) == (len(config.GENERATORS) * len(config.MODES)
                          * len(config.SEEDS) * len(config.METHODS))
    assert len(set(specs)) == len(specs)
    assert "ens_raw" not in config.METHODS


def test_matching_windows_do_not_overlap_on_experiment_streams():
    from drift_ensemble.evaluation import match_alarms

    for mode in config.MODES:
        p = plan_for("SEA", mode, 201)
        match_alarms([], p["ground_truth"], config.MAX_DELAY)  # asserts
    assert np.all(np.diff([d["start"] for d in p["ground_truth"]]) > 0)
