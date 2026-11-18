"""Online bagging (Oza & Russell) over CapyMOA / MOA Hoeffding trees.

Each tree is trained k ~ Poisson(lambda) times on an instance, as k
separate training calls (k = 0 skips the tree). MOA's OzaBag instead
trains once with instance weight k; that is NOT equivalent for an
NB-adaptive Hoeffding tree (split checks and the NB/MC score counters see
different sequences — measured 93-100% prediction agreement), so the
literal repeats of the spec are used. The trees are the raw Java learners
behind capymoa's HoeffdingTree, driven directly to avoid wrapper overhead.

Frozen copies use MOA's own copy() (a serialization deep copy). Python's
copy.deepcopy must NOT be used: it does not duplicate the Java objects.
"""

from __future__ import annotations

import numpy as np

from ensemble_experiment import config  # noqa: F401  (sets JAVA_HOME first)


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
