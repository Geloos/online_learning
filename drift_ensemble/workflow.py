"""Prequential test-run workflow.

Every method runs on every test stream:

  single_STEPD / single_SEED / single_HDDMWeighted / single_RDDM
      one detector with its cluster's selected params; every raw fire is an
      alarm (replayed from the cached HT errors)
  ens_raw        OR of all 4; each detector only self-resets
  ens_raw_resetall   OR of all 4; every alarm resets ALL detectors
                 ("drift detected, okay, reset")
  ens_cluster_vote   two votes from the same cluster within VOTE["WINDOW"]
                 instances (or one vote from a full-weight detector) signal
                 drift and reset ALL detectors; speed-based dynamic weights
  ens_cluster_sens   same pair rule as ens_cluster_vote, no weights; the
                 later voter of a pair becomes more sensitive (thresholds
                 are changed mid-stream, per cluster — see config.SENS)

The HT error sequence is deterministic given the stream, so methods read it
from the per-stream cache (tests/test_cache_replay.py proves equality with a
live HT run).
"""

from __future__ import annotations

import time

from drift_ensemble import config
from drift_ensemble.cache import load_cache
from drift_ensemble.detectors import (
    make_detector,
    run_detector_on_errors,
    set_threshold,
    threshold_for_level,
)
from drift_ensemble.ensemble import (
    ClusterSensitivityEnsemble,
    ClusterVoteEnsemble,
    RawResetAllEnsemble,
)
from streams import make_stream, stream_id

CLUSTER_OF = {
    name: cluster
    for cluster, members in config.CLUSTER_MEMBERS.items()
    for name in members
}


def params_for(name: str, selected_params: dict) -> dict:
    """A detector always runs with the params tuned for its own cluster."""
    return selected_params[CLUSTER_OF[name]][name]


def build_detectors(selected_params: dict) -> dict:
    return {
        name: make_detector(name, params_for(name, selected_params))
        for name in config.DETECTOR_ORDER
    }


def build_method_ensemble(method: str, selected_params: dict):
    """Fresh error-driven detector ensemble for an ensemble method of the
    roster (ens_raw_resetall, ens_cluster_vote, ens_cluster_sens), wired
    with its config. Each object exposes update(t, err) and .alarms."""
    if method == "ens_raw_resetall":
        return RawResetAllEnsemble(
            detectors=build_detectors(selected_params),
            order=config.DETECTOR_ORDER,
        )
    if method == "ens_cluster_vote":
        return ClusterVoteEnsemble(
            detectors=build_detectors(selected_params),
            clusters=dict(CLUSTER_OF),
            order=config.DETECTOR_ORDER,
            window=config.VOTE["WINDOW"],
            w_init=config.VOTE["W_INIT"],
            w_max=config.VOTE["W_MAX"],
            w_floor=config.VOTE["W_FLOOR"],
            step=config.VOTE["STEP"],
        )
    if method == "ens_cluster_sens":
        sens = config.SENS
        return ClusterSensitivityEnsemble(
            detectors=build_detectors(selected_params),
            clusters=dict(CLUSTER_OF),
            order=config.DETECTOR_ORDER,
            base_thresholds={
                name: float(params_for(name, selected_params)[
                    sens["SCALE"][name][0]])
                for name in config.DETECTOR_ORDER
            },
            window=sens["WINDOW"],
            late_step=sens["LATE_STEP"],
            l_max=sens["L_MAX"],
            threshold_fn=threshold_for_level,
            apply_fn=lambda name, det, value: set_threshold(det, name,
                                                            value),
        )
    raise ValueError(f"not an error-driven ensemble method: {method!r}")


def run_single_live(name: str, params: dict, generator: str, mode: str,
                    seed: int, *, min_training_size: int | None = None,
                    **plan_overrides) -> list[int]:
    """Reference implementation: live prequential HT + one detector. Only
    used by tests to prove the cached replay is identical."""
    from capymoa.classifier import HoeffdingTree

    if min_training_size is None:
        min_training_size = config.MIN_TRAINING_SIZE
    stream, _, plan = make_stream(
        generator, mode, seed, **{**config.STREAM_SIZES, **plan_overrides})
    ht = HoeffdingTree(schema=stream.get_schema())
    det = make_detector(name, params)
    alarms: list[int] = []
    for t in range(plan["length"]):
        inst = stream.next_instance()
        if t >= min_training_size:
            pred = ht.predict(inst)
            err = int(pred is None or int(pred) != int(inst.y_index))
            det.add_element(float(err))
            if det.detected_change():
                alarms.append(t)
        ht.train(inst)
    return alarms


def run_method(method: str, generator: str, mode: str, seed: int,
               selected_params: dict) -> dict:
    """Run one method on one (cached) test stream; returns a result dict with
    at least 'alarms', 'ground_truth', 'length', 'runtime_s'."""
    sid = stream_id(generator, mode, seed)
    errors, ground_truth, meta = load_cache(sid)
    t0 = time.perf_counter()

    if method.startswith("single_"):
        name = method[len("single_"):]
        alarms = run_detector_on_errors(
            name, params_for(name, selected_params), errors)
        result = {"alarms": alarms}
    elif method == "ens_raw":
        per_det = {
            name: run_detector_on_errors(
                name, params_for(name, selected_params), errors)
            for name in config.DETECTOR_ORDER
        }
        fires = [{"t": t, "detector": n} for n, ts in per_det.items()
                 for t in ts]
        fires.sort(key=lambda r: (r["t"], config.DETECTOR_ORDER.index(
            r["detector"])))
        result = {"alarms": sorted({t for ts in per_det.values()
                                    for t in ts}),
                  "raw_fires": fires}
    elif method in ("ens_raw_resetall", "ens_cluster_vote",
                    "ens_cluster_sens"):
        ens = build_method_ensemble(method, selected_params)
        for t, e in enumerate(errors):
            if e >= 0:
                ens.update(t, int(e))
        result = {"alarms": ens.alarms, "raw_fires": ens.raw_fires}
        if method == "ens_cluster_vote":
            result["weights_log"] = ens.weights_log
            result["final_weights"] = dict(ens.weights)
        elif method == "ens_cluster_sens":
            result["sens_log"] = ens.sens_log
            result["final_levels"] = dict(ens.levels)
    else:
        raise ValueError(f"unknown method: {method!r}")

    result.setdefault("ground_truth", ground_truth)
    result.setdefault("length", meta["length"])
    result["runtime_s"] = time.perf_counter() - t0
    return result
