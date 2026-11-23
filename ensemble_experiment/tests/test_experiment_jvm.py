"""Integration tests that need capymoa and a JVM."""

import json

import numpy as np
import pytest

from ensemble_experiment import config  # noqa: F401  (JAVA_HOME first)

pytestmark = pytest.mark.jvm


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


def test_frozen_copy_is_independent():
    from ensemble_experiment.bagging import EnsembleFactory

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


def test_trees_use_nb_adaptive_leaves_and_grace_period():
    from ensemble_experiment.bagging import EnsembleFactory

    schema, _ = _instances(10)
    ens = EnsembleFactory(schema, config.DDD, seed=1)()
    assert len(ens.trees) == config.DDD["n_learners"]
    opts = {str(o.getName()): str(o.getValueAsCLIString())
            for o in ens.trees[0].getOptions().getOptionArray()}
    assert opts["leafprediction"] == "NBAdaptive"
    assert opts["gracePeriod"] == str(config.DDD["grace_period"])


def test_ensemble_predicts_after_learning_and_is_reproducible():
    from ensemble_experiment.bagging import EnsembleFactory

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
def test_signal_fires_and_resets(method):
    from drift_ensemble.workflow import method_params
    from ensemble_experiment.detection import make_signal

    sig = make_signal(method, method_params(method, _phase1_params(),
                                            _cluster_params()))
    errs = [0] * 3000 + [1] * 1000
    fired = [t for t, e in enumerate(errs) if sig.update(t, e)]
    assert fired, "no alarm on a 0 -> 1 error jump"
    assert fired[0] >= 3000
    assert sig.alarms == fired
    sig.reset()  # must not raise; statistics wiped
    assert not any(sig.update(4000 + t, 0) for t in range(200))


def test_smoke_run_writes_all_files(tmp_path):
    from ensemble_experiment.runner import run_one

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
