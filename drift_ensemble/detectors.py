"""Detectors and the ensembles built from them.

  * detectors  factory, tuning grids, cached-error replay, mid-stream
               threshold control, the cluster's shared-minimum gate
               (GatedDetector, tuning phase 2)
  * ensembles  RawResetAllEnsemble, ClusterVoteEnsemble,
               ClusterSensitivityEnsemble (duck-typed detectors, so they
               are unit-testable without a JVM)
"""

from __future__ import annotations

import itertools

import numpy as np

from drift_ensemble import config


# ===========================================================================
# Detectors
#
# Detector factory, tuning grids and cached-error replay.
#
# capymoa 0.9.0 constructor kwargs match the grid names from Cerqueira et al.
# one-to-one (verified against the installed package):
#
#     STEPD(window_size, alpha_drift, ...)
#     SEED(delta, block_size, epsilon_prime, alpha, ...)
#     HDDMWeighted(drift_confidence, lambda_, test_type, ...)
#     RDDM(min_n_instances, drift_level, ...)
#
# Screening-only detectors (config.SCREENING_GRIDS, not in the roster):
#
#     DDM(min_n_instances, out_control_level, ...)
#     HDDMAverage(drift_confidence, test_type, ...)
#     ADWIN(delta)
# ===========================================================================
DETECTOR_NAMES = list(config.TUNING_GRIDS)


def _detector_classes():
    from capymoa.drift.detectors import (
        ADWIN,
        DDM,
        RDDM,
        SEED,
        STEPD,
        HDDMAverage,
        HDDMWeighted,
    )

    return {
        "STEPD": STEPD,
        "SEED": SEED,
        "HDDMWeighted": HDDMWeighted,
        "RDDM": RDDM,
        "DDM": DDM,
        "HDDMAverage": HDDMAverage,
        "ADWIN": ADWIN,
    }


def make_detector(name: str, params: dict):
    """Fresh capymoa detector with the given (grid-named) parameters."""
    cls = _detector_classes()[name]
    return cls(**params)


def iter_configs(name: str, grid: dict | None = None) -> list[dict]:
    """All grid configurations for a detector, in deterministic order
    (``grid`` defaults to config.TUNING_GRIDS[name]).

    Grids larger than SEED_RANDOM_CONFIGS (only SEED: 700 combos) are
    subsampled without replacement with a fixed seed. capymoa's default
    config (config.DEFAULT_PARAMS) is appended when the grid lacks it.
    """
    grid = config.TUNING_GRIDS[name] if grid is None else grid
    keys = list(grid)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*(grid[k] for k in keys))]
    if name == "SEED" and len(combos) > config.SEED_RANDOM_CONFIGS:
        rng = np.random.default_rng(config.GRID_SAMPLE_SEED)
        idx = rng.choice(len(combos), size=config.SEED_RANDOM_CONFIGS,
                         replace=False)
        combos = [combos[i] for i in sorted(idx)]
    default = config.DEFAULT_PARAMS.get(name)
    if (default is not None and set(default) == set(keys)
            and default not in combos):
        combos.append(dict(default))
    return combos


# ---------------------------------------------------------------------------
# Mid-stream threshold control (ens_cluster_sens)
# ---------------------------------------------------------------------------
# capymoa 0.9.0 / MOA quirk (verified by probe): the detectors copy their
# options into private Java fields, so changing only the MOA Option has no
# effect mid-stream. STEPD and RDDM read the options once (prepareForUse) and
# keep the field across resetLearning(); HDDM_W and SEED re-read the options
# on every resetLearning()/self-reset. Setting BOTH the Option and the field
# therefore changes the threshold immediately and keeps it across every
# reset path, without touching the detector's accumulated statistics.
THRESHOLD_TARGETS = {  # name -> (MOA option name, Java field path)
    "STEPD": ("alphaDrift", "alphaDrift"),
    "SEED": ("deltaSEED", "seed.DELTA"),
    "HDDMWeighted": ("driftConfidence", "driftConfidence"),
    "RDDM": ("driftLevel", "driftLevel"),
}


def threshold_for_level(name: str, base: float, level: float) -> float:
    """Drift threshold of detector ``name`` at sensitivity ``level``
    (0 = the tuned ``base``; positive = more sensitive)."""
    _, kind, factor = config.SENS["SCALE"][name]
    if kind == "mul":
        return min(config.SENS["PROB_MAX"], base * factor ** level)
    if kind == "add":
        return max(config.SENS["RDDM_LEVEL_MIN"], base + factor * level)
    raise ValueError(f"unknown scale kind {kind!r} for {name}")


def _declared_field(jobj, fname: str):
    cls = jobj.getClass()
    while cls is not None:
        try:
            field = cls.getDeclaredField(fname)
        except Exception:  # NoSuchFieldException -> look in the superclass
            cls = cls.getSuperclass()
            continue
        field.setAccessible(True)
        return field
    raise AttributeError(f"{jobj.getClass().getName()} has no field {fname}")


def _resolve_field(jobj, path: str):
    """(owner object, Field) for a dotted private-field path."""
    *parents, last = path.split(".")
    for part in parents:
        jobj = _declared_field(jobj, part).get(jobj)
    return jobj, _declared_field(jobj, last)


def get_threshold(det, name: str) -> float:
    """Current threshold as seen by the detector's Java code."""
    owner, field = _resolve_field(det.moa_detector,
                                  THRESHOLD_TARGETS[name][1])
    return float(field.getDouble(owner))


def set_threshold(det, name: str, value: float) -> None:
    """Change detector ``name``'s drift threshold mid-stream (Option and
    cached field together — see THRESHOLD_TARGETS)."""
    option_name, path = THRESHOLD_TARGETS[name]
    for opt in det.moa_detector.getOptions().getOptionArray():
        if str(opt.getName()) == option_name:
            opt.setValueViaCLIString(repr(float(value)))
            break
    else:
        raise KeyError(f"{name} has no option {option_name}")
    owner, field = _resolve_field(det.moa_detector, path)
    field.setDouble(owner, float(value))


# ---------------------------------------------------------------------------
# Tuning phase 2: the cluster's shared minimum (config.MIN_INSTANCES_GRID)
# ---------------------------------------------------------------------------
def own_minimum(name: str, params: dict) -> int:
    """Instances a detector needs after a reset before it can fire at all,
    set by its own params (probe 2026-11-23: 0s then 1s right after
    ensemble.reset_detector). HDDMWeighted has no explicit minimum."""
    if name == "STEPD":
        return 2 * int(params["window_size"])
    if name == "SEED":
        return 2 * int(params["block_size"])
    if name == "RDDM":
        return int(params["min_n_instances"])
    return 0


class GatedDetector:
    """A detector whose fires are ignored until it has seen
    ``min_instances`` instances since its last reset or its last accepted
    fire (the cluster's shared minimum). The wrapped detector keeps
    updating meanwhile; a fire it raises inside the gate is dropped (its
    own self-reset still happens inside MOA). ``moa_detector`` is passed
    through, so reset_detector and set_threshold work unchanged."""

    def __init__(self, detector, min_instances: int):
        self.detector = detector
        self.min_instances = int(min_instances)
        self.n_seen = 0  # instances since the last reset / accepted fire
        self._fired = False

    @property
    def moa_detector(self):
        return self.detector.moa_detector

    def add_element(self, element: float) -> None:
        self.detector.add_element(element)
        self.n_seen += 1
        self._fired = (self.detector.detected_change()
                       and self.n_seen >= self.min_instances)
        if self._fired:
            self.n_seen = 0

    def detected_change(self) -> bool:
        return self._fired

    def reset(self) -> None:
        self.detector.reset()
        self.n_seen = 0
        self._fired = False


def gate_alarms(alarms, errors, min_instances: int) -> list[int]:
    """The alarms a GatedDetector would keep, computed from the ungated
    alarms of a replay WITHOUT explicit resets (single detector on cached
    errors). Exact there: the wrapped detector sees the same input either
    way, so only the gate's bookkeeping differs. ``errors`` < 0 (warm-up)
    are not fed to the detector and do not count."""
    if min_instances <= 0:
        return list(alarms)
    fed = np.cumsum(np.asarray(errors) >= 0)  # instances fed up to t
    kept: list[int] = []
    last = 0
    for t in alarms:
        if fed[t] - last >= min_instances:
            kept.append(int(t))
            last = int(fed[t])
    return kept


def run_detector_on_errors(name: str, params: dict, errors,
                           min_instances: int = 0) -> list[int]:
    """Feed a cached 0/1 error sequence (entries < 0 are the warm-up period
    and are skipped) to a fresh detector; return the alarm indices. With
    ``min_instances`` > 0 the detector is gated (GatedDetector)."""
    det = make_detector(name, params)
    if min_instances > 0:
        det = GatedDetector(det, min_instances)
    alarms: list[int] = []
    for t, e in enumerate(errors):
        if e < 0:
            continue
        det.add_element(float(e))
        if det.detected_change():
            alarms.append(t)
    return alarms


# ===========================================================================
# Ensembles
#
# Error-driven drift-detector ensembles.
#
# The ensembles are written against duck-typed detectors so the state
# machines are unit-testable without a JVM: a detector needs
# ``add_element(float)``, ``detected_change() -> bool`` and ``reset()``.
#
# The Hoeffding Tree lives outside: the caller runs the prequential loop (or
# replays the cached HT errors) and passes each 0/1 error to ``update``.
# ===========================================================================
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
