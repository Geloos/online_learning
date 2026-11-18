"""set_threshold really changes a live MOA detector's drift threshold
mid-stream, and the change survives self-resets and reset_detector (guards
the capymoa 0.9.0 / MOA private-field quirk used by ens_cluster_sens)."""

import numpy as np
import pytest

from drift_ensemble import config  # noqa: F401  (JAVA_HOME before capymoa)
from drift_ensemble.detectors import (
    THRESHOLD_TARGETS,
    get_threshold,
    make_detector,
    set_threshold,
)
from drift_ensemble.ensemble import reset_detector

SWITCH = 2000
# name -> (strict params, threshold kwarg, sensitive value)
CASES = {
    "STEPD": (dict(window_size=30, alpha_drift=1e-5), "alpha_drift", 0.05),
    "SEED": (dict(delta=1e-6, epsilon_prime=0.0025, block_size=32,
                  alpha=0.5), "delta", 0.05),
    "HDDMWeighted": (dict(drift_confidence=1e-6, lambda_=0.05,
                          test_type="One-sided"), "drift_confidence", 0.05),
    "RDDM": (dict(min_n_instances=30, drift_level=3.0), "drift_level", 1.5),
}


def _errors():
    rng = np.random.default_rng(1)
    p = np.r_[np.full(4000, .20), np.full(4000, .26),
              np.full(4000, .20), np.full(4000, .26)]
    return (rng.random(len(p)) < p).astype(float)


def _run(det, errors, hook=None) -> list[int]:
    alarms = []
    for t, e in enumerate(errors):
        if t == SWITCH and hook is not None:
            hook(det)
        det.add_element(float(e))
        if det.detected_change():
            alarms.append(t)
    return alarms


def _after(alarms):
    return sum(t >= SWITCH for t in alarms)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_set_threshold_updates_option_and_field(name):
    params, _, value = CASES[name]
    det = make_detector(name, params)
    set_threshold(det, name, value)
    assert get_threshold(det, name) == pytest.approx(value)
    option_name = THRESHOLD_TARGETS[name][0]
    assert float(str(det.get_params()[option_name])) == pytest.approx(value)
    # survives self-resets after alarms and an explicit full reset
    _run(det, _errors())
    assert get_threshold(det, name) == pytest.approx(value)
    reset_detector(det)
    assert get_threshold(det, name) == pytest.approx(value)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_strict_to_sensitive_mid_stream(name):
    params, _, value = CASES[name]
    errors = _errors()
    strict = _run(make_detector(name, params), errors)
    switched = _run(make_detector(name, params), errors,
                    lambda d: set_threshold(d, name, value))
    assert _after(switched) > _after(strict)


@pytest.mark.jvm
@pytest.mark.parametrize("name", list(CASES))
def test_sensitive_to_strict_mid_stream(name):
    params, key, value = CASES[name]
    errors = _errors()
    strict_value = params[key]
    sensitive = {**params, key: value}
    switched = _run(make_detector(name, sensitive), errors,
                    lambda d: set_threshold(d, name, strict_value))
    assert _after(switched) <= 1
