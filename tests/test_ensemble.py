"""State-machine tests for VerifiedDetectorEnsemble with stub detectors and
scripted NB models (no JVM)."""

from drift_ensemble.ensemble import (
    CooldownEnsemble,
    RawResetAllEnsemble,
    VerifiedDetectorEnsemble,
)

from .fakes import (
    FakeInstance,
    ModelFactory,
    RecordingModel,
    StubDetector,
    confirmed_factory,
    false_alarm_factory,
)

ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]
CLUSTERS = {"STEPD": "ABRUPT", "SEED": "ABRUPT",
            "HDDMWeighted": "GRADUAL", "RDDM": "GRADUAL"}
# Tiny verification: N=2, D=1, E=4 -> decision at t_alarm + 6.
VERIFY = {"ABRUPT": {"N": 2, "D": 1, "E": 4, "alpha": 0.05},
          "GRADUAL": {"N": 2, "D": 1, "E": 4, "alpha": 0.05}}
# E=5 so that b=0, c=5 -> p = 2^-5 = 0.03125 < 0.05 -> CONFIRMED.
VERIFY_E5 = {"ABRUPT": {"N": 2, "D": 1, "E": 5, "alpha": 0.05},
             "GRADUAL": {"N": 2, "D": 1, "E": 5, "alpha": 0.05}}


def make_ens(fire_at, nb_factory, verify=VERIFY, **kw):
    detectors = {name: StubDetector(fire_at.get(name, ())) for name in ORDER}
    ens = VerifiedDetectorEnsemble(
        detectors=detectors, clusters=CLUSTERS, order=ORDER,
        verify_cfg=verify, nb_factory=nb_factory,
        w_init=1.0, beta=0.5, w_min=0.25, **kw)
    return ens, detectors


def drive(ens, upto):
    for t in range(upto + 1):
        ens.update(t, FakeInstance(t), 0)


def drive_from(ens, start, upto):
    for t in range(start, upto + 1):
        ens.update(t, FakeInstance(t), 0)


def test_two_false_alarms_mute_then_shadow():
    ens, _ = make_ens({"STEPD": [5, 20, 35]}, false_alarm_factory())
    drive(ens, 45)
    # Alarms at 5 and 20 output; the fire at 35 comes from a muted detector.
    assert ens.ensemble_alarms == [5, 20]
    assert ens.weights["STEPD"] == 0.25 and ens.muted["STEPD"]
    assert [v["shadow"] for v in ens.verifications] == [False, False, True]
    assert all(v["verdict"] == "FALSE_ALARM" for v in ens.verifications)
    # Muted trigger of the shadow test is not penalized further.
    assert ens.weights["STEPD"] == 0.25


def test_muted_detector_unmuted_by_confirmed_cofire():
    ens, _ = make_ens({"SEED": [10], "STEPD": [12]}, confirmed_factory(),
                      verify=VERIFY_E5)
    ens.weights["STEPD"] = 0.25
    ens.muted["STEPD"] = True
    drive(ens, 20)  # decision at t = 10 + 2 + 5 = 17
    assert ens.ensemble_alarms == [10]
    v = ens.verifications[0]
    assert v["verdict"] == "CONFIRMED" and v["cofired"] == ["STEPD"]
    assert ens.weights["STEPD"] == 1.0 and not ens.muted["STEPD"]
    assert ens.weights["SEED"] == 1.0 and not ens.muted["SEED"]


def test_only_muted_fire_gives_shadow_and_no_alarm():
    ens, _ = make_ens({"RDDM": [8]}, false_alarm_factory())
    ens.muted["RDDM"] = True
    ens.weights["RDDM"] = 0.25
    drive(ens, 20)
    assert ens.ensemble_alarms == []
    assert len(ens.verifications) == 1
    assert ens.verifications[0]["shadow"] is True
    assert ens.weights["RDDM"] == 0.25  # muted trigger never penalized


def test_no_alarm_and_no_new_verification_while_running():
    ens, _ = make_ens({"SEED": [10], "HDDMWeighted": [13]},
                      false_alarm_factory())
    drive(ens, 25)
    assert ens.ensemble_alarms == [10]
    assert len(ens.verifications) == 1
    assert ens.verifications[0]["cofired"] == ["HDDMWeighted"]
    # Both fires were logged raw.
    assert [f["detector"] for f in ens.raw_fires] == ["SEED", "HDDMWeighted"]


def test_cofire_on_final_verification_instance_counts():
    # Verification from alarm at 10 finishes at t=16 (N+E=6); a fire at 16
    # still joins the co-fired list and starts nothing new.
    ens, _ = make_ens({"SEED": [10], "RDDM": [16]}, confirmed_factory(),
                      verify=VERIFY_E5)
    drive(ens, 25)  # decision at 17
    assert ens.ensemble_alarms == [10]
    assert len(ens.verifications) == 1
    assert "RDDM" in ens.verifications[0]["cofired"]
    assert ens.weights["RDDM"] == 1.0  # rewarded by the CONFIRMED verdict


def test_same_instant_cofire_and_tie_order():
    ens, _ = make_ens({"SEED": [10], "RDDM": [10]}, false_alarm_factory())
    drive(ens, 20)
    # SEED precedes RDDM in the fixed order -> SEED is the trigger.
    assert ens.verifications[0]["trigger"] == "SEED"
    assert ens.verifications[0]["cofired"] == ["RDDM"]
    assert ens.ties == [{"t": 10, "fired": ["SEED", "RDDM"]}]
    # FALSE_ALARM penalizes only the trigger, never co-fired detectors.
    assert ens.weights["SEED"] == 0.5
    assert ens.weights["RDDM"] == 1.0


def test_trigger_is_first_unmuted_in_order():
    ens, _ = make_ens({"STEPD": [10], "SEED": [10]}, false_alarm_factory())
    ens.muted["STEPD"] = True
    ens.weights["STEPD"] = 0.25
    drive(ens, 20)
    v = ens.verifications[0]
    assert v["trigger"] == "SEED" and v["shadow"] is False
    assert v["cofired"] == ["STEPD"]


def test_reset_all_on_confirm():
    # CONFIRMED verdict (decision at t = 10 + 2 + 5 = 17) resets everyone.
    ens, dets = make_ens({"SEED": [10]}, confirmed_factory(),
                         verify=VERIFY_E5, reset_all_on_confirm=True)
    drive(ens, 16)
    assert all(d.reset_count == 0 for d in dets.values())  # not at alarm
    drive_from(ens, 17, 20)
    assert all(d.reset_count == 1 for d in dets.values())


def test_no_reset_on_false_alarm():
    ens, dets = make_ens({"SEED": [10]}, false_alarm_factory(),
                         reset_all_on_confirm=True)
    drive(ens, 20)  # FALSE_ALARM decided at 16
    assert all(d.reset_count == 0 for d in dets.values())


def test_no_reset_by_default():
    ens, dets = make_ens({"SEED": [10]}, false_alarm_factory())
    drive(ens, 20)
    assert all(d.reset_count == 0 for d in dets.values())


def test_weights_log_records_changes():
    ens, _ = make_ens({"STEPD": [5]}, false_alarm_factory())
    drive(ens, 15)  # decision at 11
    assert ens.weights_log == [
        {"t": 11, "detector": "STEPD", "w": 0.5, "muted": False}]


def test_raw_resetall_ensemble():
    detectors = {name: StubDetector({"STEPD": [5, 20],
                                     "SEED": [5]}.get(name, ()))
                 for name in ORDER}
    ens = RawResetAllEnsemble(detectors, ORDER)
    for t in range(30):
        ens.update(t, 0)
    # Every fire instant is an alarm (no window, no verification) and each
    # alarm resets ALL detectors.
    assert ens.alarms == [5, 20]
    assert [f["detector"] for f in ens.raw_fires] == ["STEPD", "SEED",
                                                      "STEPD"]
    assert all(d.reset_count == 2 for d in detectors.values())


def test_cooldown_ensemble():
    detectors = {name: StubDetector({"STEPD": [5, 6, 9, 20]}.get(name, ()))
                 for name in ORDER}
    ens = CooldownEnsemble(detectors, CLUSTERS, ORDER, VERIFY)
    for t in range(30):
        ens.update(t, 0)
    # Window after the alarm at 5: silent through 5 + N + E = 11.
    assert ens.alarms == [5, 20]
    assert len(ens.raw_fires) == 4
