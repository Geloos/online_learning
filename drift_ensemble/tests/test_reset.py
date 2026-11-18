"""reset_detector (used by ens_raw_resetall and the cluster ensembles) must
really reset the MOA detector state: capymoa's own reset() only clears
Python-side flags (regression test: without resetLearning() a reset-all
ensemble behaved exactly like one without resets)."""

import pytest

from drift_ensemble import config  # noqa: F401  (JAVA_HOME before capymoa)
from drift_ensemble.ensemble import reset_detector


def _fires(with_reset: bool) -> list[int]:
    from capymoa.drift.detectors import RDDM

    det = RDDM()
    fires = []
    seq = [0.0] * 2000 + [1.0] * 500 + [0.0] * 2000 + [1.0] * 500
    for t, v in enumerate(seq):
        det.add_element(v)
        if det.detected_change():
            fires.append(t)
        if t == 2100 and with_reset:
            reset_detector(det)
    return fires


@pytest.mark.jvm
def test_reset_detector_changes_moa_state():
    assert _fires(False) != _fires(True)
