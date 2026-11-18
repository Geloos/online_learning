"""Cached HT errors must reproduce a live prequential run exactly: the same
stream seed gives the same HT, so detector alarms replayed from the cache
equal the alarms of a detector attached to a live HT run."""

import pytest

from drift_ensemble import config
from drift_ensemble.cache import cache_stream, load_cache
from drift_ensemble.detectors import run_detector_on_errors
from drift_ensemble.workflow import run_single_live

OVERRIDES = dict(n_drifts=3, drift_every_n=1500)  # length 6000
SPEC = ("SEA", "ABRUPT", 1)


@pytest.mark.jvm
def test_cached_replay_equals_live_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    sid = cache_stream(*SPEC, **OVERRIDES)
    errors, ground_truth, meta = load_cache(sid)
    assert meta["length"] == 6000
    assert (errors[:config.MIN_TRAINING_SIZE] == -1).all()
    assert set(errors[config.MIN_TRAINING_SIZE:].tolist()) <= {0, 1}

    params = {"min_n_instances": 30, "drift_level": 1.9}
    replay = run_detector_on_errors("RDDM", params, errors)
    live = run_single_live("RDDM", params, *SPEC, **OVERRIDES)
    assert replay == live
    assert len(replay) > 0  # the test is vacuous if nothing ever fires
