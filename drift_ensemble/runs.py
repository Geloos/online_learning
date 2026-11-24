"""Running the methods on the test streams.

  * cache        one Hoeffding-tree pass per stream -> cached 0/1 errors +
                 ground truth; every detector config replays them
  * workflow     building a method's detectors with their tuned params,
                 one method on one cached stream
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
# The members are the cluster members picked by the `select` step
# (selected_params.json "clusters", config.CLUSTER_SIZE per mode by TF2),
# named <detector>_<cluster>, e.g. HDDMWeighted_ABRUPT: the detector with
# the params tuned for that cluster's mode. A detector in both clusters is
# two members. Every method runs on every test stream:
#
#   single_<member>    one member (4 singles); every raw fire is an alarm
#                  (replayed from the cached HT errors)
#   ens_raw_resetall   OR of the 4 members; every alarm resets ALL of them
#                  ("drift detected, okay, reset")
#   ens_cluster_vote   two votes from the same cluster within VOTE["WINDOW"]
#                  instances (or one vote from a full-weight detector) signal
#                  drift and reset ALL detectors; speed-based dynamic weights
#   ens_cluster_sens   same pair rule as ens_cluster_vote, no weights; the
#                  later voter of a pair becomes more sensitive (thresholds
#                  are changed mid-stream, per cluster — see config.SENS)
#
#
# The HT error sequence is deterministic given the stream, so methods read it
# from the per-stream cache (tests/test_runs.py proves equality with a
# live HT run).
# ===========================================================================
def members(selected_params: dict) -> dict[str, tuple[str, str]]:
    """member -> (detector, cluster), clusters in PURE_MODES order and
    detectors in DETECTOR_ORDER within a cluster."""
    clusters = selected_params["clusters"]
    return {f"{det}_{cluster}": (det, cluster)
            for cluster in config.PURE_MODES
            for det in config.DETECTOR_ORDER if det in clusters[cluster]}


def methods(selected_params: dict) -> list[str]:
    """The 7-method roster: one single per member, then the ensembles."""
    return ([f"single_{m}" for m in members(selected_params)]
            + config.ENSEMBLE_METHODS)


def split_member(member: str) -> tuple[str, str]:
    """'HDDMWeighted_ABRUPT' -> ('HDDMWeighted', 'ABRUPT')."""
    det, cluster = member.rsplit("_", 1)
    return det, cluster


def params_for(member: str, selected_params: dict) -> dict:
    """A member runs with the params tuned for its cluster's mode."""
    det, cluster = split_member(member)
    return selected_params[cluster][det]


def build_detector(member: str, selected_params: dict):
    return make_detector(split_member(member)[0],
                         params_for(member, selected_params))


def build_detectors(selected_params: dict) -> dict:
    return {m: build_detector(m, selected_params)
            for m in members(selected_params)}


def build_method_ensemble(method: str, selected_params: dict):
    """Fresh error-driven detector ensemble for an ensemble method of the
    roster (ens_raw_resetall, ens_cluster_vote, ens_cluster_sens), wired
    with its config. Each object exposes update(t, err) and .alarms."""
    cluster_of = {m: c for m, (_, c) in members(selected_params).items()}
    detector_of = {m: d for m, (d, _) in members(selected_params).items()}
    order = list(cluster_of)
    if method == "ens_raw_resetall":
        return RawResetAllEnsemble(
            detectors=build_detectors(selected_params),
            order=order,
        )
    if method == "ens_cluster_vote":
        return ClusterVoteEnsemble(
            detectors=build_detectors(selected_params),
            clusters=cluster_of,
            order=order,
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
            clusters=cluster_of,
            order=order,
            base_thresholds={
                m: float(params_for(m, selected_params)[
                    sens["SCALE"][detector_of[m]][0]])
                for m in order
            },
            window=sens["WINDOW"],
            late_step=sens["LATE_STEP"],
            l_max=sens["L_MAX"],
            threshold_fn=lambda m, base, level: threshold_for_level(
                detector_of[m], base, level),
            apply_fn=lambda m, det, value: set_threshold(
                det, detector_of[m], value),
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
        member = method[len("single_"):]
        alarms = run_detector_on_errors(
            split_member(member)[0], params_for(member, selected_params),
            errors)
        result = {"alarms": alarms}
    elif method in config.ENSEMBLE_METHODS:
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
    """Tuned params: {mode: {detector: params}} for both modes, plus
    "clusters": {mode: [the CLUSTER_SIZE detectors picked by TF2]}."""
    path = config.TUNING_DIR / "selected_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run tuning first "
            f"(`python main.py tune`, then `select`)")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Pool worker (module-level so it pickles under spawn)
# ---------------------------------------------------------------------------
_PARAMS: dict | None = None  # selected_params.json, loaded once per worker


def run_and_save(task: tuple[str, str, int, str]) -> tuple[str, str, float]:
    """Run one (generator, mode, seed, method) unit and persist it.
    Returns (sid, method, runtime_s). Safe to call in pool workers."""
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_selected_params()
    generator, mode, seed, method = task
    result = run_method(method, generator, mode, seed, _PARAMS)
    save_run(generator, mode, seed, method, result, _PARAMS)
    return stream_id(generator, mode, seed), method, result["runtime_s"]


def load_runs(test_dir=None, selected_params: dict | None = None
              ) -> list[dict]:
    """Every finished run under ``test_dir`` (default config.TEST_DIR):
    dicts with dir, meta and alarms. With ``selected_params``, only the
    runs made with exactly those params (older runs are stale)."""
    test_dir = config.TEST_DIR if test_dir is None else test_dir
    runs = []
    if not test_dir.exists():
        return runs
    for meta_path in sorted(test_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if (selected_params is not None
                and meta.get("selected_params") != selected_params):
            continue  # stale: run with an older tuning
        folder = meta_path.parent
        alarms = json.loads((folder / "alarms.json").read_text())["alarms"]
        runs.append({"dir": folder, "meta": meta, "alarms": alarms})
    return runs
