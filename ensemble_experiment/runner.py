"""One run = one detection method driving ExtendedDDD on one stream.

Output folder results/results_ensemble/runs/<generator>,<mode>,<seed>,<method>/:
  steps.csv     every LOG_EVERY-th step (state, accuracies, weights, caps)
  correct.npy   int8 system correctness for every step
  episodes.csv  one row per detection (see logger.EPISODE_FIELDS)
  alarms.json   signal alarms, raw detector fires, weight/sensitivity logs
  meta.json     summary + config snapshot; written LAST — its presence
                marks the run as complete (re-runs skip it).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from ensemble_experiment import config
from ensemble_experiment.bagging import EnsembleFactory
from ensemble_experiment.ddd import ExtendedDDD
from ensemble_experiment.detection import load_selected_params, make_signal
from ensemble_experiment.logger import EPISODE_FIELDS, STEP_FIELDS, RunLogger
from streams import concept_ids, mark_recurring, plan_stream, stream_from_plan

_SELECTED_PARAMS: dict | None = None


def _selected_params() -> dict:
    global _SELECTED_PARAMS
    if _SELECTED_PARAMS is None:
        _SELECTED_PARAMS = load_selected_params()
    return _SELECTED_PARAMS


def run_specs(generators=None, modes=None, seeds=None, methods=None):
    """All (generator, mode, seed, method) runs, in a fixed order."""
    generators = config.GENERATORS if generators is None else generators
    modes = config.MODES if modes is None else modes
    seeds = config.SEEDS if seeds is None else seeds
    methods = config.METHODS if methods is None else methods
    return [(g, m, s, meth) for g in generators for m in modes
            for s in seeds for meth in methods]


def run_name(generator: str, mode: str, seed: int, method: str) -> str:
    return f"{generator},{mode},{seed},{method}"


def run_dir(generator, mode, seed, method, root=None) -> Path:
    root = config.RUNS_DIR if root is None else Path(root)
    return root / run_name(generator, mode, seed, method)


def is_done(generator, mode, seed, method, root=None) -> bool:
    return (run_dir(generator, mode, seed, method, root)
            / "meta.json").exists()


def _clean(v):
    """JSON/CSV-friendly scalar."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: _clean(r.get(f)) for f in fields})


def _json(obj) -> str:
    return json.dumps(obj, indent=1, default=_clean)


def run_one(generator: str, mode: str, seed: int, method: str, *,
            selected_params: dict | None = None,
            overrides: dict | None = None,
            out_root=None,
            cfg: dict | None = None,
            min_training_size: int | None = None,
            log_every: int | None = None,
            force: bool = False) -> Path:
    out = run_dir(generator, mode, seed, method, out_root)
    if (out / "meta.json").exists() and not force:
        return out
    cfg = config.DDD if cfg is None else cfg
    selected_params = (_selected_params() if selected_params is None
                       else selected_params)
    min_training_size = (config.MIN_TRAINING_SIZE
                         if min_training_size is None else min_training_size)
    log_every = config.LOG_EVERY if log_every is None else log_every

    t0 = time.perf_counter()
    plan = plan_stream(generator, mode, seed,
                       **{**config.STREAM_SIZES, **(overrides or {})})
    stream = stream_from_plan(plan)
    ground_truth = mark_recurring(plan)
    concepts = concept_ids(plan)
    signal = make_signal(method, selected_params)
    factory = EnsembleFactory(stream.get_schema(), cfg, seed)
    logger = RunLogger(concepts, min_training_size=min_training_size,
                       log_every=log_every)
    ddd = ExtendedDDD(cfg, factory, signal,
                      min_training_size=min_training_size,
                      listener=logger.on_event)
    logger.attach(ddd)

    length = plan["length"]
    for t in range(length):
        inst = stream.next_instance()
        logger.record(t, ddd.step(t, inst))
    ddd.finish(length - 1)

    alarms = list(signal.alarms)
    episodes = logger.episode_rows(ground_truth, config.MAX_DELAY,
                                   config.DUPLICATES_AS_FP)
    summary = logger.summary(episodes, alarms, ground_truth, length,
                             config.MAX_DELAY, config.DUPLICATES_AS_FP)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "steps.csv", STEP_FIELDS, logger.rows)
    _write_csv(out / "episodes.csv", EPISODE_FIELDS, episodes)
    np.save(out / "correct.npy", logger.correct)
    ens = getattr(signal, "ensemble", None)
    (out / "alarms.json").write_text(_json({
        "alarms": alarms,
        "raw_fires": signal.raw_fires,
        **signal.logs,
        "final_weights": dict(getattr(ens, "weights", {}) or {}),
        "final_levels": dict(getattr(ens, "levels", {}) or {}),
    }))
    meta = {
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "method": method,
        "length": length,
        "n_drifts": len(ground_truth),
        "ground_truth": ground_truth,
        "functions": plan["functions"],
        "summary": {k: _clean(v) for k, v in summary.items()},
        "config": config.snapshot() if cfg is config.DDD else {
            **config.snapshot(), "DDD": cfg},
        "overrides": overrides,
        "min_training_size": min_training_size,
        "runtime_s": time.perf_counter() - t0,
    }
    (out / "meta.json").write_text(_json(meta))  # last: marks completion
    return out


def run_spec(spec: tuple[str, str, int, str]) -> str:
    """Pool-friendly wrapper: run one (generator, mode, seed, method)."""
    run_one(*spec)
    return run_name(*spec)
