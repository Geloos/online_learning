"""RawResetAllEnsemble with stub detectors (no JVM). The cluster ensembles
have their own files (test_cluster_vote.py, test_cluster_sens.py)."""

from drift_ensemble.ensemble import RawResetAllEnsemble

from .fakes import StubDetector

ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]


def test_raw_resetall_ensemble():
    detectors = {name: StubDetector({"STEPD": [5, 20],
                                     "SEED": [5]}.get(name, ()))
                 for name in ORDER}
    ens = RawResetAllEnsemble(detectors, ORDER)
    for t in range(30):
        ens.update(t, 0)
    # Every fire instant is an alarm and each alarm resets ALL detectors.
    assert ens.alarms == [5, 20]
    assert [f["detector"] for f in ens.raw_fires] == ["STEPD", "SEED",
                                                      "STEPD"]
    assert all(d.reset_count == 2 for d in detectors.values())
