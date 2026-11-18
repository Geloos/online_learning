"""Verified heterogeneous detector ensemble (PHASE 8).

The ensemble is written against duck-typed components so the state machine is
unit-testable without a JVM:

* detectors need ``add_element(float)`` and ``detected_change() -> bool``
  (and ``reset()`` in the *_resetall and cluster-vote methods),
* the NB factory and instances follow the contract in verifier.py.

The Hoeffding Tree lives outside: the caller runs the prequential loop and
passes each instance plus the HT 0/1 error to ``update``.
"""

from __future__ import annotations

from collections import deque

from drift_ensemble.verifier import CONFIRMED, FALSE_ALARM, Verification


def reset_detector(det) -> None:
    """Fully reset a detector. capymoa's reset() only clears the Python-side
    flags and history; the underlying MOA detector keeps its statistics, so
    it must be reset separately via resetLearning() (verified on capymoa
    0.9.0 — without it, the *_resetall methods are no-ops)."""
    det.reset()
    moa = getattr(det, "moa_detector", None)
    if moa is not None:
        moa.resetLearning()


class VerifiedDetectorEnsemble:
    """OR-ensemble of drift detectors with McNemar verification and
    mute/unmute weighting. Detection only — nothing is adapted or reset,
    except the optional reset_all_on_confirm ablation (ens_verified_resetall):
    a CONFIRMED verdict then resets ALL detectors, on the grounds that their
    statistics span a real concept change and are stale. False alarms never
    reset anything — they only mute the trigger."""

    def __init__(
        self,
        detectors: dict[str, object],  # name -> detector instance
        clusters: dict[str, str],  # name -> 'ABRUPT' | 'GRADUAL'
        order: list[str],  # fixed tie-break order
        verify_cfg: dict,  # cluster -> {'N', 'D', 'E', 'alpha'}
        nb_factory,  # () -> fresh NB model
        w_init: float = 1.0,
        beta: float = 0.5,
        w_min: float = 0.25,
        reset_all_on_confirm: bool = False,
    ):
        assert set(order) == set(detectors) == set(clusters)
        self.detectors = detectors
        self.clusters = clusters
        self.order = list(order)
        self.verify_cfg = verify_cfg
        self.nb_factory = nb_factory
        self.beta = beta
        self.w_min = w_min
        self.reset_all_on_confirm = reset_all_on_confirm

        self.weights = {name: w_init for name in order}
        self.muted = {name: False for name in order}
        self.verification: Verification | None = None

        buflen = max(cfg["N"] + cfg["D"] for cfg in verify_cfg.values())
        self.buffer: deque = deque(maxlen=buflen)

        # Logs
        self.raw_fires: list[dict] = []  # every detector fire, muted or not
        self.ensemble_alarms: list[int] = []  # the ensemble's alarm output
        self.verifications: list[dict] = []  # finished verification records
        self.weights_log: list[dict] = []  # every weight/mute change
        self.ties: list[dict] = []  # >1 detector fired on the same instance

    # ------------------------------------------------------------------
    def observe(self, t: int, instance) -> None:
        """Buffer-only update for the warm-up period (t < MIN_TRAINING_SIZE),
        so the pre-alarm buffer is full when the first alarm can occur."""
        self.buffer.append((t, instance))

    def update(self, t: int, instance, err: int) -> None:
        """One prequential step after HT predicted: ``err`` is HT's 0/1 error
        on ``instance``. Called for every instance once detection is active."""
        # 1. Buffer the instance; advance a running verification.
        self.buffer.append((t, instance))
        finished = False
        if self.verification is not None:
            finished = self.verification.step(t, instance)

        # 2. Feed the error to ALL detectors, muted ones included.
        fired = []
        for name in self.order:
            det = self.detectors[name]
            det.add_element(float(err))
            if det.detected_change():
                fired.append(name)
        for name in fired:
            self.raw_fires.append(
                {"t": t, "detector": name, "muted_at_fire": self.muted[name]}
            )
        if len(fired) > 1:
            self.ties.append({"t": t, "fired": list(fired)})

        # 3. While a verification is running (including the instance on which
        #    it finishes): fires only join its co-fired list. No alarm output,
        #    no new verification.
        if self.verification is not None:
            if fired:
                for name in fired:
                    self.verification.add_cofired(t, name)
        # 4. Otherwise one firing detector is enough to act.
        elif fired:
            unmuted = [n for n in fired if not self.muted[n]]
            if unmuted:
                trigger, shadow = unmuted[0], False
                self.ensemble_alarms.append(t)
            else:  # all fired detectors are muted -> shadow verification
                trigger, shadow = fired[0], True
            cluster = self.clusters[trigger]
            cfg = self.verify_cfg[cluster]
            self.verification = Verification(
                t_alarm=t,
                trigger=trigger,
                cluster=cluster,
                n_train=cfg["N"],
                gap=cfg["D"],
                n_test=cfg["E"],
                alpha=cfg["alpha"],
                shadow=shadow,
                nb_factory=self.nb_factory,
                buffer=self.buffer,
            )
            for name in fired:
                if name != trigger:
                    self.verification.add_cofired(t, name)

        # 5. Apply the verdict of a verification that finished at t.
        if finished:
            self._apply_verdict(t)

    # ------------------------------------------------------------------
    def _apply_verdict(self, t: int) -> None:
        v = self.verification
        self.verification = None
        self.verifications.append(v.record())
        if v.verdict == CONFIRMED:
            # Trigger and every co-fired detector: full weight, unmuted.
            for name in sorted({v.trigger} | v.cofired_names,
                               key=self.order.index):
                self.weights[name] = 1.0
                self.muted[name] = False
                self._log_weight(t, name)
            # Ablation: a confirmed concept change makes every detector's
            # statistics stale -> wipe them all (shadow confirms included).
            if self.reset_all_on_confirm:
                self._reset_detectors()
        elif v.verdict == FALSE_ALARM:
            # Only the trigger is penalized, and only if it is not muted
            # (a shadow trigger stays where it is). Co-fired detectors are
            # never penalized.
            name = v.trigger
            if not self.muted[name]:
                self.weights[name] *= self.beta
                if self.weights[name] <= self.w_min:
                    self.muted[name] = True
                self._log_weight(t, name)
        # INCONCLUSIVE: nothing changes.

    def _log_weight(self, t: int, name: str) -> None:
        self.weights_log.append(
            {"t": t, "detector": name,
             "w": self.weights[name], "muted": self.muted[name]}
        )

    def _reset_detectors(self) -> None:
        for det in self.detectors.values():
            reset_detector(det)


class RawResetAllEnsemble:
    """Ablation: OR of all detectors with the classic trusting response —
    every fire is an alarm and resets ALL detectors ("drift detected, okay,
    reset"). No verification, no muting, no silence window. The counterpart
    "reset only the signaler" needs no class of its own: it is ens_raw,
    because capymoa detectors restart their own statistics after firing
    (DETECTOR_SELF_RESETS). Error-driven only (no instances)."""

    def __init__(self, detectors, order):
        assert set(order) == set(detectors)
        self.detectors = detectors
        self.order = list(order)
        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []

    def update(self, t: int, err: int) -> None:
        fired = []
        for name in self.order:
            det = self.detectors[name]
            det.add_element(float(err))
            if det.detected_change():
                fired.append(name)
        for name in fired:
            self.raw_fires.append({"t": t, "detector": name})
        if fired:
            self.alarms.append(t)
            for det in self.detectors.values():
                reset_detector(det)


def _cluster_partners(order, clusters) -> dict[str, str]:
    """name -> the other member of its cluster (clusters have exactly 2)."""
    partner = {}
    for name in order:
        mates = [n for n in order
                 if n != name and clusters[n] == clusters[name]]
        assert len(mates) == 1, "clusters must have exactly 2 members"
        partner[name] = mates[0]
    return partner


def _reset_all(ens) -> None:
    """Alarm response of the cluster ensembles: wipe every detector's
    statistics; pre-reset pending votes are stale."""
    for det in ens.detectors.values():
        reset_detector(det)
    for c in ens.pending:
        ens.pending[c] = None


class ClusterVoteEnsemble:
    """Two votes from the SAME cluster within ``window`` instances signal a
    drift; every alarm resets ALL detectors. Dynamic speed weights: the
    earlier voter of a confirming pair gains ``step`` scaled by how much
    earlier it fired, the later voter loses the same amount; weights are
    clamped to [w_floor, w_max]. A detector whose weight reached ``w_max``
    may signal drift for its cluster alone. Error-driven only (no
    instances, no verification)."""

    def __init__(self, detectors, clusters, order, *, window, w_init,
                 w_max, w_floor, step):
        assert set(order) == set(detectors) == set(clusters)
        self.detectors = detectors
        self.clusters = clusters
        self.order = list(order)
        self.window = window
        self.w_max = w_max
        self.w_floor = w_floor
        self.step = step

        self.partner = _cluster_partners(order, clusters)

        self.weights = {name: w_init for name in order}
        # cluster -> (detector, t_fire) | None
        self.pending: dict[str, tuple[str, int] | None] = {
            c: None for c in set(clusters.values())
        }

        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []
        self.weights_log: list[dict] = []

    def _log_weight(self, t: int, name: str) -> None:
        self.weights_log.append(
            {"t": t, "detector": name,
             "w": self.weights[name], "muted": False}
        )

    def _reward_pair(self, t: int, *, first: str, t_first: int) -> None:
        """The earlier voter gains, its partner loses, scaled by speed."""
        second = self.partner[first]
        delta = self.step * (1.0 - (t - t_first) / self.window)
        self.weights[first] = min(self.w_max, self.weights[first] + delta)
        self.weights[second] = max(self.w_floor,
                                   self.weights[second] - delta)
        self._log_weight(t, first)
        self._log_weight(t, second)

    def update(self, t: int, err: int) -> None:
        fired = []
        for name in self.order:
            det = self.detectors[name]
            det.add_element(float(err))
            if det.detected_change():
                fired.append(name)
        for name in fired:
            self.raw_fires.append({"t": t, "detector": name})

        # Expire stale votes before processing this instance's fires.
        for c, p in self.pending.items():
            if p is not None and t - p[1] > self.window:
                self.pending[c] = None

        alarmed: set[str] = set()
        for cluster in sorted({self.clusters[n] for n in fired}):
            members = [n for n in fired if self.clusters[n] == cluster]
            p = self.pending[cluster]
            if len(members) == 2:
                # Same-instant pair -> alarm. If one of them already had a
                # pending vote from earlier, that vote is the (faster) first
                # signal; otherwise it is a tie and weights are untouched.
                alarmed.add(cluster)
                if p is not None:
                    self._reward_pair(t, first=p[0], t_first=p[1])
                self.pending[cluster] = None
            else:
                name, mate = members[0], self.partner[members[0]]
                if p is not None and p[0] == mate:
                    # Second vote confirms the pending one -> alarm.
                    alarmed.add(cluster)
                    self._reward_pair(t, first=mate, t_first=p[1])
                    self.pending[cluster] = None
                elif self.weights[name] >= self.w_max:
                    # Solo privilege: a full-weight detector alarms alone.
                    alarmed.add(cluster)
                elif p is None:
                    self.pending[cluster] = (name, t)
                # else: p holds this same detector -> the first vote stands.

        if alarmed:
            self.alarms.append(t)
            _reset_all(self)


class ClusterSensitivityEnsemble:
    """Same alarm rule as ClusterVoteEnsemble — two votes from the SAME
    cluster within ``window`` instances signal a drift and every alarm resets
    ALL detectors — but instead of vote weights each detector adapts its own
    drift threshold, cluster by cluster: the LATER voter of a confirming
    pair becomes more sensitive by ``late_step * gap / window`` levels (a
    same-instant tie changes nothing; the earlier voter is untouched).
    Unconfirmed votes simply expire — nothing ever makes a detector
    stricter.

    Levels start at 0 (the tuned threshold) and are capped at ``l_max``.
    ``threshold_fn(name, base, level)`` maps a level to a threshold and
    ``apply_fn(name, det, value)`` pushes it into the detector (injected so
    the state machine is unit-testable without a JVM). Error-driven only."""

    def __init__(self, detectors, clusters, order, *, base_thresholds,
                 window, late_step, l_max, threshold_fn, apply_fn):
        assert set(order) == set(detectors) == set(clusters) \
            == set(base_thresholds)
        self.detectors = detectors
        self.clusters = clusters
        self.order = list(order)
        self.base_thresholds = dict(base_thresholds)
        self.window = window
        self.late_step = late_step
        self.l_max = l_max
        self.threshold_fn = threshold_fn
        self.apply_fn = apply_fn

        self.partner = _cluster_partners(order, clusters)
        self.levels = {name: 0.0 for name in order}
        # cluster -> (detector, t_fire) | None
        self.pending: dict[str, tuple[str, int] | None] = {
            c: None for c in set(clusters.values())
        }

        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []
        self.sens_log: list[dict] = []

    def threshold(self, name: str) -> float:
        return self.threshold_fn(name, self.base_thresholds[name],
                                 self.levels[name])

    def _raise(self, t: int, name: str, delta: float) -> None:
        new = min(self.l_max, self.levels[name] + delta)
        if new <= self.levels[name]:
            return  # already at the cap: nothing to apply or log
        self.levels[name] = new
        value = self.threshold(name)
        self.apply_fn(name, self.detectors[name], value)
        self.sens_log.append({"t": t, "detector": name, "level": new,
                              "threshold": value, "reason": "late"})

    def _reward_late(self, t: int, *, first: str, t_first: int) -> None:
        """The partner of ``first`` confirmed late -> more sensitive."""
        gap = t - t_first
        if gap > 0:
            self._raise(t, self.partner[first],
                        self.late_step * gap / self.window)

    def update(self, t: int, err: int) -> None:
        fired = []
        for name in self.order:
            det = self.detectors[name]
            det.add_element(float(err))
            if det.detected_change():
                fired.append(name)
        for name in fired:
            self.raw_fires.append({"t": t, "detector": name})

        # Expire stale votes (no penalty).
        for c, p in self.pending.items():
            if p is not None and t - p[1] > self.window:
                self.pending[c] = None

        alarmed: set[str] = set()
        for cluster in sorted({self.clusters[n] for n in fired}):
            members = [n for n in fired if self.clusters[n] == cluster]
            p = self.pending[cluster]
            if len(members) == 2:
                # Same-instant pair -> alarm. A pending vote from one of
                # them is the earlier signal: its partner was late.
                alarmed.add(cluster)
                if p is not None:
                    self._reward_late(t, first=p[0], t_first=p[1])
                self.pending[cluster] = None
            else:
                name, mate = members[0], self.partner[members[0]]
                if p is not None and p[0] == mate:
                    alarmed.add(cluster)
                    self._reward_late(t, first=mate, t_first=p[1])
                    self.pending[cluster] = None
                elif p is None:
                    self.pending[cluster] = (name, t)
                # else: p holds this same detector -> the first vote stands.

        if alarmed:
            self.alarms.append(t)
            _reset_all(self)


class CooldownEnsemble:
    """Ablation: OR of all detectors plus the SAME 'ignore alarms for N+E
    instances after an alarm' window as the full method — but no
    verification and no muting. Separates the effect of verification from
    the effect of the cooldown window. Error-driven only (no instances)."""

    def __init__(self, detectors, clusters, order, verify_cfg):
        assert set(order) == set(detectors) == set(clusters)
        self.detectors = detectors
        self.clusters = clusters
        self.order = list(order)
        self.verify_cfg = verify_cfg
        self.cooldown_until = -1  # alarms suppressed while t <= cooldown_until
        self.alarms: list[int] = []
        self.raw_fires: list[dict] = []

    def update(self, t: int, err: int) -> None:
        fired = []
        for name in self.order:
            det = self.detectors[name]
            det.add_element(float(err))
            if det.detected_change():
                fired.append(name)
        for name in fired:
            self.raw_fires.append({"t": t, "detector": name})
        if fired and t > self.cooldown_until:
            trigger = fired[0]
            cfg = self.verify_cfg[self.clusters[trigger]]
            self.alarms.append(t)
            # Matches the full method: a verification started at t keeps the
            # ensemble silent for t+1 .. t+N+E inclusive.
            self.cooldown_until = t + cfg["N"] + cfg["E"]
