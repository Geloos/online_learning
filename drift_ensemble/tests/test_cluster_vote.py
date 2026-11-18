"""Unit tests for ClusterVoteEnsemble (no JVM; StubDetector fakes).

StubDetector fires on scripted add_element call indices; the loops below
call update(t, 0) once per t, so call index == t.
"""

from drift_ensemble.ensemble import ClusterVoteEnsemble
from drift_ensemble.tests.fakes import StubDetector

ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]
CLUSTERS = {"SEED": "ABRUPT", "RDDM": "ABRUPT",
            "STEPD": "GRADUAL", "HDDMWeighted": "GRADUAL"}
WINDOW = 250
STEP = 0.1
W_INIT = 0.75


def make_ens(fire_at: dict[str, tuple]) -> ClusterVoteEnsemble:
    return ClusterVoteEnsemble(
        detectors={n: StubDetector(fire_at.get(n, ())) for n in ORDER},
        clusters=dict(CLUSTERS),
        order=ORDER,
        window=WINDOW,
        w_init=W_INIT,
        w_max=1.0,
        w_floor=0.5,
        step=STEP,
    )


def run(ens, until: int) -> None:
    for t in range(until):
        ens.update(t, 0)


def test_single_vote_no_alarm_even_after_expiry():
    ens = make_ens({"SEED": (100,)})
    run(ens, 500)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] is None  # expired
    assert all(w == W_INIT for w in ens.weights.values())


def test_pair_within_window_alarms_and_moves_weights():
    ens = make_ens({"SEED": (100,), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    delta = STEP * (1 - 50 / WINDOW)  # 0.08
    assert abs(ens.weights["SEED"] - (W_INIT + delta)) < 1e-12
    assert abs(ens.weights["RDDM"] - (W_INIT - delta)) < 1e-12
    # the other cluster is untouched
    assert ens.weights["STEPD"] == ens.weights["HDDMWeighted"] == W_INIT


def test_pair_outside_window_does_not_alarm():
    ens = make_ens({"SEED": (100,), "RDDM": (351,)})  # dt = 251 > WINDOW
    run(ens, 400)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] == ("RDDM", 351)  # became the new vote
    assert all(w == W_INIT for w in ens.weights.values())


def test_same_instant_pair_is_tie():
    ens = make_ens({"SEED": (100,), "RDDM": (100,)})
    run(ens, 150)
    assert ens.alarms == [100]
    assert all(w == W_INIT for w in ens.weights.values())


def test_same_instant_pair_with_earlier_pending_rewards_first():
    ens = make_ens({"SEED": (100, 150), "RDDM": (150,)})
    run(ens, 200)
    assert ens.alarms == [150]
    delta = STEP * (1 - 50 / WINDOW)
    assert abs(ens.weights["SEED"] - (W_INIT + delta)) < 1e-12
    assert abs(ens.weights["RDDM"] - (W_INIT - delta)) < 1e-12


def test_full_weight_detector_alarms_alone_and_floor_holds():
    # Three pairings with dt=25 (delta 0.09) push SEED to the 1.0 cap and
    # RDDM onto the 0.5 floor; then SEED alarms alone.
    ens = make_ens({"SEED": (100, 300, 500, 700),
                    "RDDM": (125, 325, 525)})
    run(ens, 800)
    assert ens.alarms == [125, 325, 525, 700]
    assert ens.weights["SEED"] == 1.0
    assert ens.weights["RDDM"] == 0.5
    # the solo alarm changed no weights
    last_logged = [r for r in ens.weights_log if r["t"] == 700]
    assert last_logged == []


def test_alarm_resets_all_detectors_and_clears_other_cluster_pending():
    ens = make_ens({"STEPD": (90,), "SEED": (100,), "RDDM": (120,),
                    "HDDMWeighted": (130,)})
    run(ens, 350)  # before the 130-vote expires at t > 380
    assert ens.alarms == [120]
    assert all(d.reset_count == 1 for d in ens.detectors.values())
    # STEPD's pre-alarm vote was cleared; HDDM's post-alarm fire is the
    # only live vote until it expires.
    assert ens.pending["GRADUAL"] == ("HDDMWeighted", 130)


def test_cross_cluster_fires_never_pair():
    ens = make_ens({"SEED": (100,), "STEPD": (120,)})
    run(ens, 200)
    assert ens.alarms == []
    assert ens.pending["ABRUPT"] == ("SEED", 100)
    assert ens.pending["GRADUAL"] == ("STEPD", 120)
