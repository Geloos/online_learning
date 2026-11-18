import pytest

from drift_ensemble.verifier import CONFIRMED, FALSE_ALARM, Verification

from .fakes import FakeInstance, ModelFactory, RecordingModel


def make_buffer(upto, start=0):
    return [(t, FakeInstance(t)) for t in range(start, upto + 1)]


def run_verification(v, upto):
    for t in range(v.t_alarm + 1, upto + 1):
        v.step(t, FakeInstance(t))


def test_exact_index_windows():
    # N=3, D=2, E=4, alarm at t=10:
    #   S trains on [10-2-3+1 .. 10-2] = [6, 7, 8], then frozen
    #   R trains on 11..13
    #   test on 14..17, decision at 17, nobody trains during the test
    s, r = RecordingModel(right=True), RecordingModel(right=True)
    v = Verification(t_alarm=10, trigger="STEPD", cluster="ABRUPT",
                     n_train=3, gap=2, n_test=4, alpha=0.05, shadow=False,
                     nb_factory=ModelFactory([s, r]),
                     buffer=make_buffer(10))
    assert s.trained == [6, 7, 8]
    run_verification(v, 17)
    assert r.trained == [11, 12, 13]
    assert s.predicted == [14, 15, 16, 17]
    assert r.predicted == [14, 15, 16, 17]
    assert s.trained == [6, 7, 8]  # S frozen, R never trains in test phase
    assert r.trained == [11, 12, 13]
    assert v.finished and v.t_decision == 17


def test_decision_exactly_at_t_end():
    s, r = RecordingModel(True), RecordingModel(True)
    v = Verification(10, "STEPD", "ABRUPT", 3, 2, 4, 0.05, False,
                     ModelFactory([s, r]), make_buffer(10))
    run_verification(v, 16)
    assert not v.finished
    assert v.step(17, FakeInstance(17))
    assert v.finished


def test_b_c_counting_and_verdicts():
    # S always right, R always wrong -> b=E, c=0 -> FALSE_ALARM.
    v = Verification(10, "STEPD", "ABRUPT", 3, 2, 6, 0.05, False,
                     ModelFactory([RecordingModel(True),
                                   RecordingModel(False)]),
                     make_buffer(10))
    run_verification(v, v.t_end)
    assert (v.b, v.c) == (6, 0)
    assert v.verdict == FALSE_ALARM

    # S always wrong, R always right -> c=E=6 -> p = 2^-6 < 0.05 -> CONFIRMED.
    v2 = Verification(10, "STEPD", "ABRUPT", 3, 2, 6, 0.05, False,
                      ModelFactory([RecordingModel(False),
                                    RecordingModel(True)]),
                      make_buffer(10))
    run_verification(v2, v2.t_end)
    assert (v2.b, v2.c) == (0, 6)
    assert v2.p_value == pytest.approx(2 ** -6)
    assert v2.verdict == CONFIRMED


def test_buffer_must_cover_s_window():
    with pytest.raises(AssertionError):
        Verification(10, "STEPD", "ABRUPT", 3, 2, 4, 0.05, False,
                     ModelFactory([RecordingModel(True),
                                   RecordingModel(True)]),
                     buffer=make_buffer(10, start=8))  # misses t=6, 7


def test_alarm_too_early_asserts():
    with pytest.raises(AssertionError):
        Verification(4, "STEPD", "ABRUPT", 3, 2, 4, 0.05, False,
                     ModelFactory([RecordingModel(True),
                                   RecordingModel(True)]),
                     buffer=make_buffer(4))


def test_record_fields():
    v = Verification(10, "SEED", "ABRUPT", 3, 2, 4, 0.05, True,
                     ModelFactory([RecordingModel(True),
                                   RecordingModel(False)]),
                     make_buffer(10))
    v.add_cofired(12, "RDDM")
    run_verification(v, v.t_end)
    rec = v.record()
    assert rec["t_alarm"] == 10 and rec["trigger"] == "SEED"
    assert rec["shadow"] is True
    assert rec["cofired"] == ["RDDM"]
    assert rec["verdict"] == FALSE_ALARM
    assert rec["t_decision"] == v.t_end
