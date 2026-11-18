"""Grid/random search of detector parameters on cached HT errors (PHASE 5).

One tuning *task* = one (detector, config) pair evaluated on every tuning
stream. Tasks are independent, so they run in a multiprocessing pool; each
spawned worker imports capymoa itself (one JVM per worker, never a forked
one) and loads the cached error sequences once.
"""

from __future__ import annotations

import json

from drift_ensemble import config
from drift_ensemble.cache import load_cache
from drift_ensemble.detectors import iter_configs, run_detector_on_errors
from drift_ensemble.evaluation import compute_metrics, delay_curve
from drift_ensemble.streams import stream_id, tuning_stream_specs

TUNING_RUNS_CSV = config.TUNING_DIR / "tuning_runs.csv"
ROW_FIELDS = [
    "detector", "params", "generator", "mode", "seed",
    "tp", "fp", "fn", "precision", "recall", "f1", "f2",
    "mean_delay", "far", "delay_auc",
]


def params_key(params: dict) -> str:
    """Canonical JSON for a param dict (stable key across runs)."""
    return json.dumps(params, sort_keys=True)


def all_tasks() -> list[tuple[str, str]]:
    """Every (detector, params_json) task, in deterministic order."""
    return [
        (name, params_key(params))
        for name in config.TUNING_GRIDS
        for params in iter_configs(name)
    ]


def evaluate_config_on_stream(
    detector: str,
    params: dict,
    sid: str,
    errors,
    ground_truth: list[dict],
    length: int,
) -> dict:
    alarms = run_detector_on_errors(detector, params, errors)
    metrics = compute_metrics(
        alarms, ground_truth, config.MAX_DELAY, length,
        config.MIN_TRAINING_SIZE, config.DUPLICATES_AS_FP,
    )
    dc = delay_curve(alarms, ground_truth, config.DELAY_GRID,
                     config.DUPLICATES_AS_FP)
    generator, mode, seed = sid.split(",")
    return {
        "detector": detector,
        "params": params_key(params),
        "generator": generator,
        "mode": mode,
        "seed": int(seed),
        **metrics,
        "delay_auc": dc["delay_auc"],
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level so it pickles under spawn)
# ---------------------------------------------------------------------------
_WORKER_STREAMS: list[tuple[str, object, list[dict], int]] = []


def worker_init(stream_ids: list[str]) -> None:
    """Pool initializer: starts this worker's JVM (via the capymoa import in
    drift_ensemble.detectors) and loads every cached error sequence once."""
    global _WORKER_STREAMS
    _WORKER_STREAMS = []
    for sid in stream_ids:
        errors, ground_truth, meta = load_cache(sid)
        _WORKER_STREAMS.append((sid, errors, ground_truth, meta["length"]))


def run_task(task: tuple[str, str]) -> list[dict]:
    """Evaluate one (detector, params_json) on every preloaded stream."""
    detector, pjson = task
    params = json.loads(pjson)
    return [
        evaluate_config_on_stream(detector, params, sid, errors,
                                  ground_truth, length)
        for sid, errors, ground_truth, length in _WORKER_STREAMS
    ]


def tuning_stream_ids() -> list[str]:
    return [stream_id(*spec) for spec in tuning_stream_specs()]
