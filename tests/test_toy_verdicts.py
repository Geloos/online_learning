"""Toy-stream verdicts with REAL capymoa NaiveBayes models (PHASE 10):
a forced alarm on a stationary stream must come back FALSE_ALARM, and a
forced alarm right after a label flip must come back CONFIRMED."""

import numpy as np
import pytest

from drift_ensemble import config  # noqa: F401  (JAVA_HOME before capymoa)
from drift_ensemble.verifier import CONFIRMED, FALSE_ALARM, Verification

N = D = E = 200
ALPHA = 0.05


def collect_instances(X, y):
    from capymoa.stream import NumpyStream

    stream = NumpyStream(X, y.astype(int), dataset_name="toy")
    schema = stream.get_schema()
    insts = [stream.next_instance() for _ in range(len(X))]
    return insts, schema


def nb_factory_for(schema):
    from capymoa.classifier import NaiveBayes

    return lambda: NaiveBayes(schema=schema)


def run_verification(t_alarm, insts, schema, shadow=False):
    buffer = [(t, insts[t]) for t in range(t_alarm + 1)]
    v = Verification(t_alarm, "STEPD", "ABRUPT", N, D, E, ALPHA, shadow,
                     nb_factory_for(schema), buffer)
    for t in range(t_alarm + 1, t_alarm + N + E + 1):
        v.step(t, insts[t])
    assert v.finished
    return v


@pytest.mark.jvm
def test_stationary_forced_alarm_is_false_alarm():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1500, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    insts, schema = collect_instances(X, y)
    v = run_verification(1000, insts, schema)
    # S and R were trained on the same concept: R has no real edge.
    assert v.verdict == FALSE_ALARM, (v.b, v.c, v.p_value)


@pytest.mark.jvm
def test_label_flip_forced_alarm_is_confirmed():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(2500, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[2000:] = 1 - y[2000:]  # abrupt concept change at t = 2000
    insts, schema = collect_instances(X, y)
    v = run_verification(2000, insts, schema)
    # S learned the old concept, R the new one: R must beat S clearly.
    assert v.c > v.b
    assert v.verdict == CONFIRMED, (v.b, v.c, v.p_value)
