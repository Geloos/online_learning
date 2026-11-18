"""Extended DDD state machine (Minku & Yao, "DDD: A New Ensemble Approach
For Dealing With Concept Drift", IEEE TKDE 2011), with three extensions:

* drift-type awareness: min_steps after a detection, the drift is
  classified "fast" or "slow" from the accuracy drop of the old-low
  ensemble against the main model's pre-drift accuracy,
* an adaptive per-kind cap on the after-drift mode (cap.AdaptiveCap),
* a concept pool of frozen past ensembles, one of which may be recalled as
  a fifth voter (pool.ConceptPool).

The drift signal is any object with ``update(t, err) -> bool`` and
``reset()`` (ensemble_experiment.detection). Ensembles come from a
``make_ensemble()`` factory and need ``predict(inst)``, ``learn(inst,
lam)`` and ``frozen_copy()`` — both injected so the state machine is
unit-testable without a JVM. The ground truth is never seen here.

Two choices beyond the spec, both because the WATCHED model changes:
  (a) every detection resets the whole drift signal (the new-low ensemble
      it watches is replaced by a fresh one);
  (b) an exit whose winner is not the current new-low also resets it.
"""

from __future__ import annotations

from ensemble_experiment.cap import AdaptiveCap
from ensemble_experiment.pool import ConceptPool
from ensemble_experiment.tracker import PrequentialTracker, SlidingAcc

VOTERS = ("nl", "ol", "oh", "rc")  # new-low, old-low, old-high, recalled
WINNER_ATTR = {"nl": "new_low", "ol": "old_low", "oh": "old_high",
               "rc": "recalled"}


class ExtendedDDD:
    def __init__(self, cfg: dict, make_ensemble, signal, *,
                 min_training_size: int = 0, listener=None):
        self.cfg = cfg
        self.make_ensemble = make_ensemble
        self.signal = signal
        self.min_training_size = min_training_size
        self.listener = listener  # optional callable(event dict)

        self.mode = "before"
        self.new_low = make_ensemble()
        self.new_high = make_ensemble()
        self.old_low = None
        self.old_high = None
        self.recalled = None
        self.recalled_pool_id: int | None = None

        self.trackers = {v: PrequentialTracker() for v in VOTERS}
        self.pool_trackers: dict[int, PrequentialTracker] = {}
        self.main_window = SlidingAcc(cfg["ref_window"])

        self.acc_ref: float | None = None
        self.steps_since_drift = 0
        self.kind: str | None = None
        self.drop: float | None = None

        self.pool = ConceptPool(cfg["pool_size"])
        self.cap = AdaptiveCap.from_cfg(cfg)

        self.episode: dict | None = None  # open after-mode episode
        self.episodes: list[dict] = []  # finished episodes
        self.last_weights: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        if self.listener is not None:
            self.listener(event)

    def active_voters(self) -> list[str]:
        if self.mode == "before":
            return ["nl"]
        return ["nl", "ol", "oh"] + (["rc"] if self.recalled is not None
                                     else [])

    def n_active_ensembles(self) -> int:
        """Ensembles held (and trained): 2 before a drift, 4-5 after."""
        if self.mode == "before":
            return 2
        return 4 + (self.recalled is not None)

    def _ensemble(self, voter: str):
        return getattr(self, WINNER_ATTR[voter])

    def vote_weights(self) -> dict[str, float]:
        """Normalized accuracy weights of the after-mode voters; old-low's
        accuracy is multiplied by W. All-zero -> equal weights."""
        voters = self.active_voters()
        raw = {v: self.trackers[v].acc for v in voters}
        raw["ol"] *= self.cfg["W"]
        s = sum(raw.values())
        if s == 0:
            return {v: 1.0 / len(voters) for v in voters}
        return {v: raw[v] / s for v in voters}

    @staticmethod
    def weighted_vote(preds: dict, weights: dict) -> int | None:
        """Sum of weights per predicted class; argmax, ties -> lowest
        label. Voters predicting None are skipped."""
        scores: dict[int, float] = {}
        for v, p in preds.items():
            if p is not None:
                scores[p] = scores.get(p, 0.0) + weights[v]
        if not scores:
            return None
        best = max(scores.values())
        return min(label for label, s in scores.items() if s == best)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _component_predictions(self, inst) -> dict:
        return {v: self._ensemble(v).predict(inst)
                for v in self.active_voters()}

    def predict(self, inst) -> int | None:
        preds = self._component_predictions(inst)
        if self.mode == "before":
            return preds["nl"]
        return self.weighted_vote(preds, self.vote_weights())

    # ------------------------------------------------------------------
    # One prequential step: predict with everything, then update, then learn
    # ------------------------------------------------------------------
    def step(self, t: int, inst) -> dict:
        y = int(inst.y_index)

        # 1. Every prediction for step t before any component learns.
        preds = self._component_predictions(inst)
        if self.mode == "before":
            system_pred = preds["nl"]
            self.last_weights = {"nl": 1.0}
        else:
            self.last_weights = self.vote_weights()
            system_pred = self.weighted_vote(preds, self.last_weights)
        scoring = (self.mode == "after"
                   and self.steps_since_drift < self.cfg["min_steps"]
                   and self.pool_trackers)
        pool_preds = ({e["id"]: e["ensemble"].predict(inst)
                       for e in self.pool.entries
                       if e["id"] in self.pool_trackers}
                      if scoring else {})

        if self.mode == "before":
            self.main_window.update(preds["nl"] == y)
        else:
            for v, p in preds.items():
                self.trackers[v].update(p is not None and p == y)
            for pid, p in pool_preds.items():
                self.pool_trackers[pid].update(p is not None and p == y)

        # 2. Drift detection on new-low's error.
        detected = False
        exited = None
        if t >= self.min_training_size:
            err = int(preds["nl"] is None or preds["nl"] != y)
            detected = bool(self.signal.update(t, err))
        if detected:
            self._on_detection(t)
        elif self.mode == "after":
            # 3.
            self.steps_since_drift += 1
            # 4.
            if self.steps_since_drift >= self.cfg["min_steps"]:
                if self.steps_since_drift == self.cfg["min_steps"]:
                    self._classify()
                    self._recall(t)
                exited = self._check_exit(t)

        # 5. Learning.
        lo, hi = self.cfg["lambda_low"], self.cfg["lambda_high"]
        self.new_low.learn(inst, lo)
        self.new_high.learn(inst, hi)
        if self.mode == "after":
            self.old_low.learn(inst, lo)
            self.old_high.learn(inst, lo)  # old-high switches to LOW
            if self.recalled is not None:
                self.recalled.learn(inst, lo)

        return {
            "pred": system_pred,
            "correct": system_pred is not None and system_pred == y,
            "detected": detected,
            "exited": exited,
        }

    # ------------------------------------------------------------------
    # 3.1 Detection
    # ------------------------------------------------------------------
    def _on_detection(self, t: int) -> None:
        was = self.mode
        if self.episode is not None:  # detection during after-mode
            self._close_episode(t, "interrupted", winner=None)

        pool_add_id = evicted = None
        if was == "before":
            # The main model of the ending concept goes to the pool,
            # stored BEFORE the reassignment below.
            pool_add_id, evicted = self.pool.add(self.new_low.frozen_copy(),
                                                 t)
            if pool_add_id is not None:
                self._emit({"type": "pool_add", "t": t, "id": pool_add_id,
                            "evicted": evicted})

        # Reassignment (Minku & Yao, Algorithm 2).
        if was == "before" or (self.trackers["nl"].acc
                               > self.trackers["oh"].acc):
            self.old_low = self.new_low
        else:
            self.old_low = self.old_high
        self.old_high = self.new_high
        self.new_low = self.make_ensemble()
        self.new_high = self.make_ensemble()

        # An interrupted episode loses its recalled model.
        self.recalled = None
        self.recalled_pool_id = None

        if was == "before":
            self.acc_ref = self.main_window.acc  # None if the window is empty
        # (in after-mode the existing acc_ref is kept)

        for tr in self.trackers.values():
            tr.reset()
        self.pool_trackers = {pid: PrequentialTracker()
                              for pid in self.pool.ids()}
        self.steps_since_drift = 0
        self.kind = None
        self.drop = None
        self.signal.reset()  # choice (a): the watched model was replaced
        self.mode = "after"

        self.episode = {
            "t_detect": t,
            "mode_at_detection": was,
            "pool_add_id": pool_add_id,
            "pool_evicted_id": evicted,
            "acc_ref": self.acc_ref,
            "kind": None,
            "drop": None,
            "recalled_pool_id": None,
            "recall_score": None,
        }
        self._emit({"type": "detection", "t": t, "mode_before": was})

    # ------------------------------------------------------------------
    # 3.2 / 3.3 Classification and recall (at steps_since_drift == min_steps)
    # ------------------------------------------------------------------
    def _classify(self) -> None:
        if self.acc_ref is None:
            self.kind = "slow"
            self.drop = None
        else:
            self.drop = self.acc_ref - self.trackers["ol"].acc
            self.kind = "fast" if self.drop > self.cfg["tau"] else "slow"
        self.episode["kind"] = self.kind
        self.episode["drop"] = self.drop

    def _recall(self, t: int) -> None:
        candidates = [e for e in self.pool.entries
                      if e["id"] in self.pool_trackers]
        if not candidates:
            return
        # Highest scoring accuracy; ties -> most recent (latest in FIFO
        # order, i.e. the last maximum).
        best = None
        for e in candidates:
            acc = self.pool_trackers[e["id"]].acc
            if best is None or acc >= best[1]:
                best = (e, acc)
        entry, acc = best
        self.recalled = entry["ensemble"].frozen_copy()  # train the COPY
        self.recalled_pool_id = entry["id"]
        self.trackers["rc"] = self.pool_trackers[entry["id"]].copy()
        self.episode["recalled_pool_id"] = entry["id"]
        self.episode["recall_score"] = acc
        self._emit({"type": "recall", "t": t, "id": entry["id"]})

    # ------------------------------------------------------------------
    # 3.4 Exit
    # ------------------------------------------------------------------
    def _dominates(self, v: str, others: list[str]) -> bool:
        a = self.trackers[v]
        return all(a.acc - a.std > self.trackers[x].acc + self.trackers[x].std
                   for x in others)

    def _check_exit(self, t: int) -> str | None:
        voters = self.active_voters()
        tr = self.trackers
        has_rc = "rc" in voters

        winner = exit_type = None
        nl = tr["nl"].acc
        if (nl > tr["ol"].acc and nl > tr["oh"].acc
                and (not has_rc or nl > tr["rc"].acc)):
            winner, exit_type = "nl", "natural_new_low"
        elif self._dominates("oh", [v for v in voters if v != "oh"]):
            winner, exit_type = "oh", "natural_old_high"
        elif has_rc and self._dominates("rc",
                                        [v for v in voters if v != "rc"]):
            winner, exit_type = "rc", "natural_recalled"
        elif (self.cfg.get("forced_exit", True)
              and self.steps_since_drift >= self.cap.get(self.kind)):
            # Old-low may win here (likely a false alarm). Ties -> the
            # first voter in VOTERS order.
            best = max(tr[v].acc for v in voters)
            winner = next(v for v in voters if tr[v].acc == best)
            exit_type = "forced"
        if winner is None:
            return None

        cap_used = self.cap.get(self.kind)
        if exit_type == "forced":
            self.cap.on_forced_exit(self.kind)
        else:
            self.cap.on_natural_exit(self.kind, self.steps_since_drift)

        main = self._ensemble(winner)
        if winner == "rc":
            # Update the stored concept (no duplicate entry).
            if self.pool.replace(self.recalled_pool_id,
                                 self.recalled.frozen_copy(), t):
                self._emit({"type": "pool_replace", "t": t,
                            "id": self.recalled_pool_id})
        if winner != "nl":
            self.signal.reset()  # choice (b): the watched model changes
        self.new_low = main  # new_high is kept

        self.episode["cap_used"] = cap_used
        self._close_episode(t, exit_type, winner)
        self.old_low = None
        self.old_high = None
        self.recalled = None
        self.recalled_pool_id = None
        for v in VOTERS:
            self.trackers[v].reset()
        self.pool_trackers = {}
        self.main_window.reset()
        self.mode = "before"
        self._emit({"type": "exit", "t": t, "exit_type": exit_type,
                    "winner": winner})
        return exit_type

    def _close_episode(self, t: int, exit_type: str, winner) -> None:
        ep = self.episode
        ep["t_exit"] = t
        ep["steps_after"] = self.steps_since_drift
        ep["exit_type"] = exit_type
        ep["winner"] = winner
        ep.setdefault("cap_used", None)
        ep["cap_fast_after"] = self.cap.get("fast")
        ep["cap_slow_after"] = self.cap.get("slow")
        self.episodes.append(ep)
        self.episode = None

    def finish(self, t_end: int) -> None:
        """Close an episode still open when the stream ends."""
        if self.episode is not None:
            self._close_episode(t_end, "unfinished", winner=None)
