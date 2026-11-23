"""What extended DDD drives: online-bagging ensembles of Hoeffding trees
(the learners) and the drift signal (one of experiment 1's detection
methods, with the same tuned params as there).
"""

from __future__ import annotations

import json

import numpy as np

from drift_ensemble.detectors import reset_detector
from drift_ensemble.runs import (
    build_detector,
    build_method_ensemble,
    method_params,
)
from ensemble_experiment import config


# ===========================================================================
# Online bagging
#
# Online bagging (Oza & Russell) over CapyMOA / MOA Hoeffding trees.
#
# Each tree is trained k ~ Poisson(lambda) times on an instance, as k
# separate training calls (k = 0 skips the tree). MOA's OzaBag instead
# trains once with instance weight k; that is NOT equivalent for an
# NB-adaptive Hoeffding tree (split checks and the NB/MC score counters see
# different sequences — measured 93-100% prediction agreement), so the
# literal repeats of the spec are used. The trees are the raw Java learners
# behind capymoa's HoeffdingTree, driven directly to avoid wrapper overhead.
#
# Frozen copies use MOA's own copy() (a serialization deep copy). Python's
# copy.deepcopy must NOT be used: it does not duplicate the Java objects.
# ===========================================================================
def _tree_vote(tree, ji) -> int | None:
    """Class index predicted by one Java tree, None if it abstains (no
    votes yet, or all votes zero). Ties -> lowest index (MOA maxIndex)."""
    from moa.core import Utils

    votes = tree.getVotesForInstance(ji)
    if len(votes) == 0:
        return None
    idx = int(Utils.maxIndex(votes))
    if votes[idx] <= 0.0:
        return None
    return idx


def majority(preds) -> int | None:
    """Unweighted majority over non-None predictions; ties -> lowest
    label; None if every voter abstains."""
    counts: dict[int, int] = {}
    for p in preds:
        if p is not None:
            counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return min(label for label, c in counts.items() if c == best)


class OnlineBaggingEnsemble:
    def __init__(self, trees: list, rng: np.random.Generator):
        self.trees = trees
        self.rng = rng

    @classmethod
    def new(cls, schema, *, n_learners: int, grace_period: int,
            leaf_prediction: str, rng: np.random.Generator):
        from capymoa.classifier import HoeffdingTree

        trees = [
            HoeffdingTree(schema=schema, grace_period=grace_period,
                          leaf_prediction=leaf_prediction,
                          random_seed=i + 1).moa_learner
            for i in range(n_learners)
        ]
        return cls(trees, rng)

    def predict(self, inst) -> int | None:
        ji = inst.java_instance
        return majority(_tree_vote(tree, ji) for tree in self.trees)

    def learn(self, inst, lam: float) -> None:
        ji = inst.java_instance
        ks = self.rng.poisson(lam, size=len(self.trees))
        for tree, k in zip(self.trees, ks):
            for _ in range(int(k)):
                tree.trainOnInstance(ji)

    def frozen_copy(self) -> "OnlineBaggingEnsemble":
        """Independent deep copy (trees via MOA copy()); it gets its own
        RNG stream, spawned deterministically from this ensemble's."""
        return OnlineBaggingEnsemble([t.copy() for t in self.trees],
                                     self.rng.spawn(1)[0])


class EnsembleFactory:
    """Callable handing out fresh ensembles; every ensemble gets its own
    child RNG spawned from one SeedSequence, so a run is reproducible."""

    def __init__(self, schema, cfg: dict, seed: int):
        self.schema = schema
        self.cfg = cfg
        self.seed_seq = np.random.SeedSequence(seed)

    def __call__(self) -> OnlineBaggingEnsemble:
        rng = np.random.default_rng(self.seed_seq.spawn(1)[0])
        return OnlineBaggingEnsemble.new(
            self.schema,
            n_learners=self.cfg["n_learners"],
            grace_period=self.cfg["grace_period"],
            leaf_prediction=self.cfg["leaf_prediction"],
            rng=rng,
        )


# ===========================================================================
# Drift signals
#
# Drift signals for DDD built from the detector study's methods.
#
# A signal exposes ``update(t, err) -> bool`` (True = drift alarm at t),
# ``reset()`` and the logs ``alarms``, ``raw_fires`` and ``logs``. The
# detectors run with the params tuned for their cluster, exactly as in the
# detector study: tuning phase 1 (selected_params.json) for the singles and
# ens_raw_resetall, phase 2 (cluster_params.json: re-tuned params behind the
# cluster's shared minimum) for the two-vote methods.
#
# reset() wipes every detector's statistics (and pending cluster votes) but
# keeps what an ensemble has learned about its detectors: cluster_vote
# weights and cluster_sens sensitivity levels survive a DDD reset.
# ===========================================================================
def _load(path, step: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the drift-ensemble experiment's "
            f"`python main.py {step}` first")
    return json.loads(path.read_text())


def load_method_params(method: str) -> dict:
    """The tuned params ``method`` runs with: phase 2 for the two-vote
    methods, phase 1 for the singles and ens_raw_resetall. Only the file
    the method needs is read."""
    if method_params(method, "phase1", "phase2") == "phase2":
        return _load(config.CLUSTER_PARAMS_PATH, "phase2")
    return _load(config.SELECTED_PARAMS_PATH, "select")


class SingleSignal:
    """One detector; every fire is an alarm."""

    def __init__(self, name: str, detector):
        self.name = name
        self.detector = detector
        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []
        self.logs: dict[str, list] = {}

    def update(self, t: int, err: int) -> bool:
        self.detector.add_element(float(err))
        if self.detector.detected_change():
            self.alarms.append(t)
            self.raw_fires.append({"t": t, "detector": self.name})
            return True
        return False

    def reset(self) -> None:
        reset_detector(self.detector)


class EnsembleSignal:
    """Wraps an error-driven detector ensemble of drift_ensemble.detectors
    (RawResetAllEnsemble / ClusterVoteEnsemble / ClusterSensitivity...)."""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    @property
    def alarms(self) -> list[int]:
        return self.ensemble.alarms

    @property
    def raw_fires(self) -> list[dict]:
        return self.ensemble.raw_fires

    @property
    def logs(self) -> dict[str, list]:
        out = {}
        for key in ("weights_log", "sens_log"):
            if hasattr(self.ensemble, key):
                out[key] = getattr(self.ensemble, key)
        return out

    def update(self, t: int, err: int) -> bool:
        n_before = len(self.ensemble.alarms)
        self.ensemble.update(t, int(err))
        return len(self.ensemble.alarms) > n_before

    def reset(self) -> None:
        for det in self.ensemble.detectors.values():
            reset_detector(det)
        pending = getattr(self.ensemble, "pending", None)
        if pending is not None:
            for c in pending:
                pending[c] = None


def make_signal(method: str, selected_params: dict):
    if method not in config.METHODS:
        raise ValueError(f"unknown detection method: {method!r}")
    if method.startswith("single_"):
        name = method[len("single_"):]
        return SingleSignal(name, build_detector(name, selected_params))
    return EnsembleSignal(build_method_ensemble(method, selected_params))
