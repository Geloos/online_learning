"""Drift signals for DDD built from the detector study's methods.

A signal exposes ``update(t, err) -> bool`` (True = drift alarm at t),
``reset()`` and the logs ``alarms``, ``raw_fires`` and ``logs``. The
detectors run with the params tuned for their cluster
(results/tuning/selected_params.json), exactly as in the detector study.

reset() wipes every detector's statistics (and pending cluster votes) but
keeps what an ensemble has learned about its detectors: cluster_vote
weights and cluster_sens sensitivity levels survive a DDD reset.
"""

from __future__ import annotations

import json

from ensemble_experiment import config
from drift_ensemble.ensemble import reset_detector
from drift_ensemble.workflow import (
    build_method_ensemble,
    make_detector,
    params_for,
)


def load_selected_params(path=None) -> dict:
    path = config.SELECTED_PARAMS_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the detector study's "
            f"scripts/03_select_params.py first")
    return json.loads(path.read_text())


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
    """Wraps an error-driven detector ensemble of drift_ensemble.ensemble
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
        return SingleSignal(name, make_detector(
            name, params_for(name, selected_params)))
    return EnsembleSignal(build_method_ensemble(method, selected_params))
