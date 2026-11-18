import pytest

from drift_ensemble.verifier import (
    CONFIRMED,
    FALSE_ALARM,
    INCONCLUSIVE,
    mcnemar_one_sided,
    verdict,
)

ALPHA = 0.05


def test_confirmed_case():
    p = mcnemar_one_sided(10, 25)
    assert p == pytest.approx(0.0083, abs=5e-4)
    assert verdict(10, 25, ALPHA) == CONFIRMED


def test_inconclusive_case():
    p = mcnemar_one_sided(20, 30)
    assert p == pytest.approx(0.101, abs=5e-3)
    assert verdict(20, 30, ALPHA) == INCONCLUSIVE


def test_false_alarm_case():
    assert verdict(30, 28, ALPHA) == FALSE_ALARM


def test_empty_counts_is_false_alarm():
    assert mcnemar_one_sided(0, 0) == 1.0
    assert verdict(0, 0, ALPHA) == FALSE_ALARM
