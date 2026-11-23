"""Error-driven drift-detector ensembles.

The ensembles are written against duck-typed detectors so the state
machines are unit-testable without a JVM: a detector needs
``add_element(float)``, ``detected_change() -> bool`` and ``reset()``.

The Hoeffding Tree lives outside: the caller runs the prequential loop (or
replays the cached HT errors) and passes each 0/1 error to ``update``.
"""

from __future__ import annotations


def reset_detector(det) -> None:
    """Fully reset a detector. capymoa's reset() only clears the Python-side
    flags and history; the underlying MOA detector keeps its statistics, so
    it must be reset separately via resetLearning() (verified on capymoa
    0.9.0 — without it, the *_resetall methods are no-ops)."""
    det.reset()
    moa = getattr(det, "moa_detector", None)
    if moa is not None:
        moa.resetLearning()


class RawResetAllEnsemble:
    """OR of all detectors with the classic trusting response — every fire
    is an alarm and resets ALL detectors ("drift detected, okay, reset").
    Error-driven only (no instances)."""

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
    may signal drift for its cluster alone. Error-driven only."""

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
            {"t": t, "detector": name, "w": self.weights[name]}
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
