"""The shared streams.py as used by the drift-ensemble experiment (its
stream sizes from drift_ensemble.config). RECURRING is covered by
ensemble_experiment/tests/test_recurring_streams.py."""

import numpy as np
import pytest

from drift_ensemble import config
from drift_ensemble.config import test_stream_specs as _test_specs
from drift_ensemble.config import tuning_stream_specs as _tuning_specs
import streams
from streams import make_stream, stream_id


def plan(generator, mode, seed, **overrides):
    return streams.plan_stream(generator, mode, seed,
                               **{**config.STREAM_SIZES, **overrides})


# ---------------------------------------------------------------------------
# Plan-level tests (no JVM)
# ---------------------------------------------------------------------------
def test_abrupt_plan_positions_and_length():
    p = plan("SEA", "ABRUPT", 1)
    gt = p["ground_truth"]
    assert len(gt) == config.N_DRIFTS
    assert [d["start"] for d in gt] == [
        (i + 1) * config.DRIFT_EVERY_N for i in range(config.N_DRIFTS)]
    assert all(d["end"] == d["start"] for d in gt)
    assert all(d["type"] == "ABRUPT" for d in gt)
    assert p["length"] == config.N_DRIFTS * config.DRIFT_EVERY_N + \
        config.DRIFT_EVERY_N


def test_gradual_plan_widths_and_length():
    p = plan("Agrawal", "GRADUAL", 2)
    gt = p["ground_truth"]
    assert all(d["end"] - d["start"] == config.GRADUAL_WIDTH for d in gt)
    assert all(d["type"] == "GRADUAL" for d in gt)
    # start positions shift by the widths of earlier drifts
    assert gt[0]["start"] == config.DRIFT_EVERY_N
    assert gt[1]["start"] == 2 * config.DRIFT_EVERY_N + config.GRADUAL_WIDTH
    assert p["length"] == (config.N_DRIFTS * config.DRIFT_EVERY_N
                           + config.N_DRIFTS * config.GRADUAL_WIDTH
                           + config.DRIFT_EVERY_N)


def test_mixed_plan_split():
    p = plan("Agrawal", "MIXED", 3)
    types = [d["type"] for d in p["ground_truth"]]
    assert types.count("ABRUPT") == config.N_DRIFTS // 2  # 15 / 15
    assert types.count("GRADUAL") == config.N_DRIFTS - config.N_DRIFTS // 2
    # the permutation actually mixes the two kinds
    assert len(set(types[:10])) > 1


def test_same_seed_same_plan_different_seed_differs():
    a = plan("SEA", "MIXED", 5)
    b = plan("SEA", "MIXED", 5)
    c = plan("SEA", "MIXED", 6)
    assert a == b
    assert a != c


def test_adjacent_functions_differ_and_chain():
    for gen in config.GENERATORS:
        p = plan(gen, "ABRUPT", 4)
        fns = p["functions"]
        assert all(fns[i] != fns[i + 1] for i in range(len(fns) - 1))
        assert all(f in streams.FUNCTIONS[gen] for f in fns)
        gt = p["ground_truth"]
        assert all(gt[i]["function_after"] == gt[i + 1]["function_before"]
                   for i in range(len(gt) - 1))
        assert gt[0]["function_before"] == streams.INIT_FUNCTION


def test_plan_overrides():
    p = plan("SEA", "ABRUPT", 1, n_drifts=4, drift_every_n=1500)
    assert [d["start"] for d in p["ground_truth"]] == [1500, 3000, 4500,
                                                       6000]
    assert p["length"] == 7500


def test_stream_sets():
    tuning = _tuning_specs()
    test = _test_specs()
    assert len(tuning) == 18  # 3 generators x 2 modes x 3 seeds
    assert len(test) == 45  # 3 x 2 x 5 + 3 x MIXED x 5
    tuning_seeds = {s for _, _, s in tuning}
    test_seeds = {s for _, _, s in test}
    assert not tuning_seeds & test_seeds
    assert len({stream_id(*s) for s in tuning + test}) == 63


# ---------------------------------------------------------------------------
# JVM tests
# ---------------------------------------------------------------------------
@pytest.mark.jvm
def test_same_seed_identical_instances():
    def read(n):
        stream, _, _ = make_stream("SEA", "ABRUPT", 3, n_drifts=1,
                                   drift_every_n=200, gradual_width=0)
        xs, ys = [], []
        for _ in range(n):
            inst = stream.next_instance()
            xs.append(np.asarray(inst.x, dtype=float))
            ys.append(int(inst.y_index))
        return np.array(xs), ys

    x1, y1 = read(400)
    x2, y2 = read(400)
    assert y1 == y2
    assert np.array_equal(x1, x2)


@pytest.mark.jvm
def test_all_generators_build_and_run():
    for gen in config.GENERATORS:
        stream, gt, p = make_stream(gen, "GRADUAL", 2, n_drifts=1,
                                    drift_every_n=150, gradual_width=50)
        assert gt[0] == {"start": 150, "end": 200, "type": "GRADUAL",
                         "function_before": gt[0]["function_before"],
                         "function_after": gt[0]["function_after"]}
        ys = {int(stream.next_instance().y_index)
              for _ in range(p["length"])}
        assert len(ys) >= 1
