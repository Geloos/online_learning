"""Running the methods on the test streams.

  * cache        one Hoeffding-tree pass per stream -> cached 0/1 errors +
                 ground truth; every detector config replays them
  * workflow     which tuned params a method uses (phase 1 or 2), building
                 its detectors, one method on one cached stream
  * run folders  saving a run (alarms.json, weights.csv, sensitivity.csv,
                 raw_fires.csv, meta.json last) and loading finished runs
"""

from __future__ import annotations

import csv
import json
import time

import numpy as np

from drift_ensemble import config
from drift_ensemble.detectors import (
    ClusterSensitivityEnsemble,
    ClusterVoteEnsemble,
    GatedDetector,
    make_detector,
    RawResetAllEnsemble,
    run_detector_on_errors,
    set_threshold,
    threshold_for_level,
)
from streams import make_stream, stream_id


# ===========================================================================
# HT error cache
#
# HT error cache.
#
# HT is never reset and the detectors never influence it, so HT's 0/1 error
# sequence on a stream is the same whatever detector is attached. It is
# computed ONCE per stream and reused for every detector configuration.
#
# errors[t] = -1 during the warm-up (t < MIN_TRAINING_SIZE), else 0/1.
# ===========================================================================
def stream_cache_dir(sid: str):
    return config.CACHE_DIR / sid


def is_cached(sid: str) -> bool:
    d = stream_cache_dir(sid)
    return (d / "meta.json").exists() and (d / "errors.npy").exists()


def cache_stream(
    generator: str,
    mode: str,
    seed: int,
    *,
    force: bool = False,
    min_training_size: int | None = None,
    **plan_overrides,
) -> str:
    """Run HT prequentially once over the stream; cache the 0/1 errors, the
    prequential accuracy curve and the ground truth. Returns the stream id.
    Skips work if the cache already exists (resumable)."""
    sid = stream_id(generator, mode, seed)
    if is_cached(sid) and not force:
        return sid
    if min_training_size is None:
        min_training_size = config.MIN_TRAINING_SIZE

    from capymoa.classifier import HoeffdingTree

    t0 = time.perf_counter()
    stream, ground_truth, plan = make_stream(
        generator, mode, seed, **{**config.STREAM_SIZES, **plan_overrides})
    n = plan["length"]
    ht = HoeffdingTree(schema=stream.get_schema())
    errors = np.full(n, -1, dtype=np.int8)
    accuracy = np.full(n, np.nan, dtype=np.float32)
    seen = 0
    correct = 0
    for t in range(n):
        inst = stream.next_instance()
        if t >= min_training_size:
            pred = ht.predict(inst)
            err = int(pred is None or int(pred) != int(inst.y_index))
            errors[t] = err
            seen += 1
            correct += 1 - err
            accuracy[t] = correct / seen
        ht.train(inst)

    out = stream_cache_dir(sid)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "errors.npy", errors)
    np.save(out / "ht_accuracy.npy", accuracy)
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=1))
    meta = {
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "length": n,
        "min_training_size": min_training_size,
        "plan_overrides": plan_overrides,
        "final_accuracy": float(accuracy[n - 1]),
        "runtime_s": time.perf_counter() - t0,
    }
    # meta.json is written last: its presence marks the cache as complete.
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return sid


def cache_stream_spec(spec: tuple[str, str, int]) -> str:
    """Pool-friendly wrapper: cache one (generator, mode, seed) spec."""
    return cache_stream(*spec)


def load_cache(sid: str) -> tuple[np.ndarray, list[dict], dict]:
    """(errors, ground_truth, meta) for a cached stream."""
    d = stream_cache_dir(sid)
    if not is_cached(sid):
        raise FileNotFoundError(
            f"stream {sid!r} is not cached yet — run "
            f"`python main.py cache` (tuning) or `test` first"
        )
    errors = np.load(d / "errors.npy")
    ground_truth = json.loads((d / "ground_truth.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return errors, ground_truth, meta


# ===========================================================================
# Running a method
#
# Prequential test-run workflow.
#
# Every method runs on every test stream:
#
#   single_STEPD / single_SEED / single_HDDMWeighted / single_RDDM
#       one detector with its cluster's selected params; every raw fire is an
#       alarm (replayed from the cached HT errors)
#   ens_raw_resetall   OR of all 4; every alarm resets ALL detectors
#                  ("drift detected, okay, reset")
#   ens_cluster_vote   two votes from the same cluster within VOTE["WINDOW"]
#                  instances (or one vote from a full-weight detector) signal
#                  drift and reset ALL detectors; speed-based dynamic weights
#   ens_cluster_sens   same pair rule as ens_cluster_vote, no weights; the
#                  later voter of a pair becomes more sensitive (thresholds
#                  are changed mid-stream, per cluster — see config.SENS)
#
# Every detector runs with its cluster's params. The singles and
# ens_raw_resetall use tuning phase 1 (selected_params.json, no shared
# minimum); the two-vote methods (config.PHASE2_METHODS) use phase 2
# (cluster_params.json), where each detector also sits behind its cluster's
# shared minimum (GatedDetector: no fire is accepted until M instances after
# a reset or an accepted fire). method_params picks the file per method.
#
# The HT error sequence is deterministic given the stream, so methods read it
# from the per-stream cache (tests/test_runs.py proves equality with a
# live HT run).
# ===========================================================================
CLUSTER_OF = {
    name: cluster
    for cluster, members in config.CLUSTER_MEMBERS.items()
    for name in members
}


def method_params(method: str, phase1: dict, phase2: dict) -> dict:
    """The tuning a method runs with: phase 2 (cluster_params.json) for
    the two-vote methods, phase 1 (selected_params.json) for the rest."""
    return phase2 if method in config.PHASE2_METHODS else phase1


def params_for(name: str, selected_params: dict) -> dict:
    """A detector always runs with the params tuned for its own cluster."""
    return selected_params[CLUSTER_OF[name]][name]


def min_instances_for(name: str, selected_params: dict) -> int:
    """The shared minimum of the detector's cluster (tuning phase 2,
    cluster_params.json). 0 = no gate (a phase-1 selected_params dict)."""
    return int(selected_params.get("min_instances", {})
               .get(CLUSTER_OF[name], 0))


def build_detector(name: str, selected_params: dict):
    """One detector with its cluster's params, gated by its cluster's
    shared minimum when there is one."""
    det = make_detector(name, params_for(name, selected_params))
    m = min_instances_for(name, selected_params)
    return GatedDetector(det, m) if m > 0 else det


def build_detectors(selected_params: dict) -> dict:
    return {name: build_detector(name, selected_params)
            for name in config.DETECTOR_ORDER}


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
            name, params_for(name, selected_params), errors,
            min_instances_for(name, selected_params))
        result = {"alarms": alarms}
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


# ===========================================================================
# Run folders
#
# Saving/loading of test runs.
#
# One run = one method on one test stream, stored under
# results/results_drift_ensemble/test/<generator>,<mode>,<seed>,<method>/ with
# alarms.json, weights.csv, sensitivity.csv, raw_fires.csv and meta.json
# (written last: its presence marks the run as complete).
# ===========================================================================
WEIGHT_FIELDS = ["t", "detector", "w"]
SENS_FIELDS = ["t", "detector", "level", "threshold", "reason"]


def run_dir(sid: str, method: str):
    return config.TEST_DIR / f"{sid},{method}"


def is_run_done(sid: str, method: str) -> bool:
    return (run_dir(sid, method) / "meta.json").exists()


def _write_csv(path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_run(generator: str, mode: str, seed: int, method: str,
             result: dict, selected_params: dict) -> None:
    sid = stream_id(generator, mode, seed)
    out = run_dir(sid, method)
    out.mkdir(parents=True, exist_ok=True)

    (out / "alarms.json").write_text(
        json.dumps({"alarms": [int(t) for t in result["alarms"]]}))

    _write_csv(out / "weights.csv", WEIGHT_FIELDS,
               result.get("weights_log", []))
    _write_csv(out / "sensitivity.csv", SENS_FIELDS,
               result.get("sens_log", []))

    raw = result.get("raw_fires", [])
    raw_fields = ["t", "detector"]
    _write_csv(out / "raw_fires.csv", raw_fields,
               [{f: r.get(f, "") for f in raw_fields} for r in raw])

    meta = {
        "method": method,
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "sid": sid,
        "length": result["length"],
        "runtime_s": result["runtime_s"],
        "n_alarms": len(result["alarms"]),
        "n_raw_fires": len(raw),
        "final_weights": result.get("final_weights"),
        "final_levels": result.get("final_levels"),
        "ground_truth": result["ground_truth"],
        "selected_params": selected_params,
        "config": config.snapshot(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_selected_params() -> dict:
    """Phase-1 params (both modes, every detector; used by screening)."""
    path = config.TUNING_DIR / "selected_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run tuning phase 1 first "
            f"(`python main.py tune`, then `select`)")
    return json.loads(path.read_text())


def load_cluster_params(path=None) -> dict:
    """Phase-2 params: each cluster's shared minimum and its members'
    params (used by config.PHASE2_METHODS, the two-vote ensembles)."""
    path = config.CLUSTER_PARAMS_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run tuning phase 2 first "
            f"(`python main.py phase2`)")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Pool worker (module-level so it pickles under spawn)
# ---------------------------------------------------------------------------
_PHASES: tuple[dict, dict] | None = None  # (phase 1, phase 2)


def run_and_save(task: tuple[str, str, int, str]) -> tuple[str, str, float]:
    """Run one (generator, mode, seed, method) unit and persist it.
    Returns (sid, method, runtime_s). Safe to call in pool workers."""
    global _PHASES
    if _PHASES is None:
        _PHASES = (load_selected_params(), load_cluster_params())
    generator, mode, seed, method = task
    params = method_params(method, *_PHASES)
    result = run_method(method, generator, mode, seed, params)
    save_run(generator, mode, seed, method, result, params)
    return stream_id(generator, mode, seed), method, result["runtime_s"]


def load_runs(test_dir=None) -> list[dict]:
    """Every finished run of a roster method under ``test_dir`` (default
    config.TEST_DIR): dicts with dir, meta and alarms."""
    test_dir = config.TEST_DIR if test_dir is None else test_dir
    runs = []
    if not test_dir.exists():
        return runs
    for meta_path in sorted(test_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["method"] not in config.METHODS:
            continue  # stale folder of a method dropped from the roster
        folder = meta_path.parent
        alarms = json.loads((folder / "alarms.json").read_text())["alarms"]
        runs.append({"dir": folder, "meta": meta, "alarms": alarms})
    return runs
