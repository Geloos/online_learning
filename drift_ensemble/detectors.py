"""Detector factory, tuning grids and cached-error replay.

capymoa 0.9.0 constructor kwargs match the grid names from Cerqueira et al.
one-to-one (verified against the installed package):

    STEPD(window_size, alpha_drift, ...)
    SEED(delta, block_size, epsilon_prime, alpha, ...)
    HDDMWeighted(drift_confidence, lambda_, test_type, ...)
    RDDM(min_n_instances, drift_level, ...)

Screening-only detectors (config.SCREENING_GRIDS, not in the roster):

    DDM(min_n_instances, out_control_level, ...)
    HDDMAverage(drift_confidence, test_type, ...)
    ADWIN(delta)
"""

from __future__ import annotations

import itertools

import numpy as np

from drift_ensemble import config  # noqa: F401  (sets JAVA_HOME first)

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


def run_detector_on_errors(name: str, params: dict, errors) -> list[int]:
    """Feed a cached 0/1 error sequence (entries < 0 are the warm-up period
    and are skipped) to a fresh detector; return the alarm indices."""
    det = make_detector(name, params)
    alarms: list[int] = []
    for t, e in enumerate(errors):
        if e < 0:
            continue
        det.add_element(float(e))
        if det.detected_change():
            alarms.append(t)
    return alarms
