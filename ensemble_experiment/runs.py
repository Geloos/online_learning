"""One run of extended DDD -> one result folder.

  * logging    RunLogger: per-step rows, episodes, and DDD's behaviour
               matched against the ground truth
  * synthetic  one (generator, mode, seed, method) run
               -> results/results_ensemble/runs/<...>/
  * real data  one (dataset, method) run on Electricity, Airlines, Covtype
               or Poker -> results/results_ensemble/real/<...>/
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import urllib.request
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

from drift_ensemble.stats import compute_metrics, match_alarms
from ensemble_experiment import config
from ensemble_experiment.ddd import ExtendedDDD, VOTERS
from ensemble_experiment.models import (
    EnsembleFactory,
    load_method_params,
    make_signal,
)
from streams import concept_ids, mark_recurring, plan_stream, stream_from_plan


# ===========================================================================
# Logging
#
# Per-step, per-episode and per-run logging of an ExtendedDDD run.
#
# The logger is the only place where DDD's behaviour meets the ground truth:
# it tags pool entries with the true concept they represent, checks recalls
# against the true current concept and labels detections TP / duplicate / FP
# with the detector study's matching rule (drift_ensemble.stats.match_alarms).
#
# Pool-entry tag = the majority true concept over the period the stored
# model was main (from the previous exit, or the stream start, to the
# detection). A recalled entry that wins its episode is re-tagged with the
# majority concept of that episode.
# ===========================================================================
STEP_FIELDS = (["t", "mode", "n_active", "correct", "acc_global",
                "acc_main"]
               + [f"acc_{v}" for v in VOTERS]
               + [f"w_{v}" for v in VOTERS]
               + ["kind", "cap_fast", "cap_slow", "pool_size", "concept"])

EPISODE_FIELDS = [
    "t_detect", "mode_at_detection", "label", "drift_idx", "delay",
    "true_type", "true_recurring", "kind", "kind_correct", "drop", "acc_ref",
    "steps_after", "t_exit", "exit_type", "winner", "cap_used",
    "cap_fast_after", "cap_slow_after", "pool_add_id", "pool_add_tag",
    "pool_evicted_id", "recalled_pool_id", "recalled_just_added",
    "recall_score", "recalled_tag", "true_concept_at_recall", "recall_hit",
]

EXIT_TYPES = ["natural_new_low", "natural_old_high", "natural_recalled",
              "forced", "interrupted", "unfinished"]


def _majority(ids: np.ndarray) -> int | None:
    if len(ids) == 0:
        return None
    return int(Counter(ids.tolist()).most_common(1)[0][0])


def label_alarms(alarms, ground_truth, max_delay, duplicates_as_fp=True):
    """alarm t -> (label, drift index | None, delay | None) under the
    detector study's matching rule."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    m = match_alarms(alarms, truth, max_delay, duplicates_as_fp)
    first = {a: k for k, a in enumerate(m.matched) if a is not None}
    starts = [d["start"] for d in truth]
    out = {}
    for a in alarms:
        if a in first:
            k = first[a]
            out[a] = ("TP", k, a - starts[k])
            continue
        k = bisect_right(starts, a) - 1
        if k >= 0 and a <= truth[k]["end"] + max_delay:
            out[a] = ("duplicate", k, a - starts[k])
        else:
            out[a] = ("FP", None, None)
    return out


class RunLogger:
    def __init__(self, concepts: np.ndarray, *, min_training_size: int,
                 log_every: int):
        self.concepts = concepts
        self.min_training_size = min_training_size
        self.log_every = log_every
        n = len(concepts)
        self.correct = np.zeros(n, dtype=np.int8)
        self.n_active = np.zeros(n, dtype=np.int8)
        self.rows: list[dict] = []
        self.n_seen = 0
        self.n_correct = 0
        self.ddd = None

        self.pool_tags: dict[int, int | None] = {}
        self.pool_add_tag: dict[int, int | None] = {}  # t_detect -> tag
        self.recall_info: dict[int, dict] = {}  # t_detect -> recall record
        self.main_since = 0
        self.t_detect = None

    def attach(self, ddd) -> None:
        self.ddd = ddd

    # ------------------------------------------------------------------
    def on_event(self, ev: dict) -> None:
        t = ev["t"]
        kind = ev["type"]
        if kind == "pool_add":
            tag = _majority(self.concepts[self.main_since:t + 1])
            self.pool_tags[ev["id"]] = tag
            self.pool_add_tag[t] = tag
        elif kind == "detection":
            self.t_detect = t
        elif kind == "recall":
            tag = self.pool_tags.get(ev["id"])
            truth = int(self.concepts[t])
            self.recall_info[self.t_detect] = {
                "recalled_tag": tag,
                "true_concept_at_recall": truth,
                "recall_hit": tag is not None and tag == truth,
            }
        elif kind == "pool_replace":
            self.pool_tags[ev["id"]] = _majority(
                self.concepts[self.t_detect:t + 1])
        elif kind == "exit":
            self.main_since = t + 1

    # ------------------------------------------------------------------
    def record(self, t: int, info: dict) -> None:
        ok = bool(info["correct"])
        self.correct[t] = ok
        d = self.ddd
        self.n_active[t] = d.n_active_ensembles()
        if t >= self.min_training_size:
            self.n_seen += 1
            self.n_correct += ok
        if t % self.log_every:
            return
        after = d.mode == "after"
        active = set(d.active_voters()) if after else set()
        row = {
            "t": t,
            "mode": d.mode,
            "n_active": int(self.n_active[t]),
            "correct": int(ok),
            "acc_global": (self.n_correct / self.n_seen
                           if self.n_seen else None),
            "acc_main": d.main_window.acc if not after else None,
            "kind": d.kind,
            "cap_fast": d.cap.get("fast"),
            "cap_slow": d.cap.get("slow"),
            "pool_size": len(d.pool),
            "concept": int(self.concepts[t]),
        }
        for v in VOTERS:
            row[f"acc_{v}"] = d.trackers[v].acc if v in active else None
            row[f"w_{v}"] = d.last_weights.get(v) if v in active else None
        self.rows.append(row)

    # ------------------------------------------------------------------
    def episode_rows(self, ground_truth, max_delay, duplicates_as_fp):
        episodes = self.ddd.episodes
        labels = label_alarms([e["t_detect"] for e in episodes],
                              ground_truth, max_delay, duplicates_as_fp)
        truth = sorted(ground_truth, key=lambda d: d["start"])
        rows = []
        for e in episodes:
            label, k, delay = labels[e["t_detect"]]
            true_type = truth[k]["type"] if k is not None else None
            kind_correct = None
            if label == "TP" and e["kind"] is not None:
                kind_correct = ((e["kind"] == "fast")
                                == (true_type == "ABRUPT"))
            rec = self.recall_info.get(e["t_detect"], {})
            rid = e.get("recalled_pool_id")
            rows.append({
                **{f: e.get(f) for f in EPISODE_FIELDS if f in e},
                "label": label,
                "drift_idx": k,
                "delay": delay,
                "true_type": true_type,
                "true_recurring": (truth[k].get("recurring")
                                   if k is not None else None),
                "kind_correct": kind_correct,
                "pool_add_tag": self.pool_add_tag.get(e["t_detect"]),
                # The entry frozen at THIS detection is a copy of old-low:
                # recalling it is not a recall of an earlier concept.
                "recalled_just_added": (None if rid is None
                                        else rid == e.get("pool_add_id")),
                "recalled_tag": rec.get("recalled_tag"),
                "true_concept_at_recall": rec.get("true_concept_at_recall"),
                "recall_hit": rec.get("recall_hit"),
            })
        return rows

    def summary(self, episode_rows, alarms, ground_truth, length,
                max_delay, duplicates_as_fp) -> dict:
        n = len(self.correct)
        warm = self.correct[self.min_training_size:]
        out = {
            "acc": float(warm.mean()) if len(warm) else float("nan"),
            "acc_all": float(self.correct.mean()) if n else float("nan"),
            "n_detections": len(alarms),
        }
        out.update(compute_metrics(alarms, ground_truth, max_delay, length,
                                   self.min_training_size, duplicates_as_fp))
        out["frac_after"] = float((self.n_active >= 4).mean())
        out["frac_4"] = float((self.n_active == 4).mean())
        out["frac_5"] = float((self.n_active == 5).mean())
        for et in EXIT_TYPES:
            out[f"exit_{et}"] = sum(r["exit_type"] == et
                                    for r in episode_rows)
        graded = [r for r in episode_rows if r["kind_correct"] is not None]
        out["n_kind_graded"] = len(graded)
        out["kind_acc"] = (sum(r["kind_correct"] for r in graded)
                           / len(graded) if graded else float("nan"))
        recalls = [r for r in episode_rows if r["recall_hit"] is not None]
        out["n_recalls"] = len(recalls)
        out["recall_hit_rate"] = (sum(r["recall_hit"] for r in recalls)
                                  / len(recalls) if recalls
                                  else float("nan"))
        older = [r for r in recalls if not r["recalled_just_added"]]
        out["n_recalls_older"] = len(older)
        out["recall_hit_rate_older"] = (sum(r["recall_hit"] for r in older)
                                        / len(older) if older
                                        else float("nan"))
        steps = [r["steps_after"] for r in episode_rows
                 if r["exit_type"] not in ("interrupted", "unfinished")]
        out["mean_steps_after"] = (float(np.mean(steps)) if steps
                                   else float("nan"))
        cap = self.ddd.cap
        out["final_cap_fast"] = cap.get("fast")
        out["final_cap_slow"] = cap.get("slow")
        return out


# ===========================================================================
# Synthetic streams
#
# One run = one detection method driving ExtendedDDD on one stream.
#
# Output folder results/results_ensemble/runs/<generator>,<mode>,<seed>,<method>/:
#   steps.csv     every LOG_EVERY-th step (state, accuracies, weights, caps)
#   correct.npy   int8 system correctness for every step
#   episodes.csv  one row per detection (see logger.EPISODE_FIELDS)
#   alarms.json   signal alarms, raw detector fires, weight/sensitivity logs
#   meta.json     summary + config snapshot; written LAST — its presence
#                 marks the run as complete (re-runs skip it).
# ===========================================================================
_METHOD_PARAMS: dict[str, dict] = {}


def _method_params(method: str) -> dict:
    """Tuned params for ``method`` (phase 1 or 2), loaded once."""
    if method not in _METHOD_PARAMS:
        _METHOD_PARAMS[method] = load_method_params(method)
    return _METHOD_PARAMS[method]


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
    selected_params = (_method_params(method) if selected_params is None
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
        "detector_params": selected_params,  # incl. cluster minimums
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


# ===========================================================================
# Real datasets
#
# Extended DDD on real datasets (no known drift positions).
#
# Real data has no ground truth, so only what needs none is reported:
# prequential accuracy (test-then-train, after the MIN_TRAINING_SIZE
# warm-up), the number of detections and the accuracy over time. Every
# detection-quality metric (TP/FP, delay, recovery, recall hits) is left out.
#
# Runs: config.REAL_METHODS (ens_raw_resetall), ONE run per dataset with
# seed config.REAL_SEED (the seed drives the online-bagging Poisson draws),
# plus one baseline:
#   no_change     predicts the previous instance's label (no learner, no
#                 seed); Electricity is known to be strongly autocorrelated,
#                 so this is the bar a learner has to clear
#
# Run folders: results/results_ensemble/real/<dataset>,<seed>,<method>/
# (correct.npy, alarms.json, episodes.csv, steps.csv, meta.json last).
# Statistics (computed from those folders only):
# statistics/ensemble_experiment/real/<dataset>/.
# ===========================================================================

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dataset_path(name: str) -> Path:
    """Local ARFF of a real dataset; downloaded once and checked against
    its md5 (a wrong file would silently change every result)."""
    spec = config.REAL_DATASETS[name]
    path = config.DATA_DIR / spec["file"]
    if not path.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.replace(path)
    if _md5(path) != spec["md5"]:
        raise ValueError(f"{path}: md5 {_md5(path)} != {spec['md5']} — "
                         f"delete it and download again")
    return path


def load_stream(name: str):
    from capymoa.stream import ARFFStream

    return ARFFStream(str(dataset_path(name)))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def real_run_specs(datasets=None, methods=None):
    """One run per dataset x method, all with seed config.REAL_SEED."""
    datasets = list(config.REAL_DATASETS) if datasets is None else datasets
    methods = config.REAL_RUN_METHODS if methods is None else methods
    return [(d, config.REAL_SEED, m) for d in datasets for m in methods]


def real_run_name(dataset: str, seed: int, method: str) -> str:
    return f"{dataset},{seed},{method}"


def real_run_dir(dataset, seed, method, root=None) -> Path:
    root = config.REAL_RUNS_DIR if root is None else Path(root)
    return root / real_run_name(dataset, seed, method)


def real_is_done(dataset, seed, method, root=None) -> bool:
    return (real_run_dir(dataset, seed, method, root) / "meta.json").exists()


def _meta(dataset, seed, method, length, summary, t0,
          detector_params=None) -> dict:
    return {"dataset": dataset, "seed": seed, "method": method,
            "length": length,
            "summary": {k: _clean(v) for k, v in summary.items()},
            "config": config.snapshot(),
            "detector_params": detector_params,  # incl. cluster minimums
            "runtime_s": time.perf_counter() - t0}


def _run_no_change(dataset, seed, out: Path, t0) -> Path:
    stream = load_stream(dataset)
    labels = []
    while stream.has_more_instances():
        labels.append(int(stream.next_instance().y_index))
    y = np.asarray(labels)
    correct = np.zeros(len(y), dtype=np.int8)
    correct[1:] = y[1:] == y[:-1]
    warm = correct[config.MIN_TRAINING_SIZE:]
    summary = {"acc": float(warm.mean()), "acc_all": float(correct.mean()),
               "n_detections": 0}
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "correct.npy", correct)
    (out / "meta.json").write_text(_json(
        _meta(dataset, seed, "no_change", len(y), summary, t0)))
    return out


def real_run_one(dataset: str, seed: int, method: str, *, out_root=None,
                 force: bool = False) -> Path:
    out = real_run_dir(dataset, seed, method, out_root)
    if (out / "meta.json").exists() and not force:
        return out
    t0 = time.perf_counter()
    if method == "no_change":
        return _run_no_change(dataset, seed, out, t0)

    length = config.REAL_DATASETS[dataset]["length"]
    stream = load_stream(dataset)
    params = _method_params(method)  # ens_raw_resetall -> phase 1
    signal = make_signal(method, params)
    factory = EnsembleFactory(stream.get_schema(), config.DDD, seed)
    # No ground truth: one dummy concept for the logger.
    logger = RunLogger(np.zeros(length, dtype=np.int64),
                       min_training_size=config.MIN_TRAINING_SIZE,
                       log_every=config.LOG_EVERY)
    ddd = ExtendedDDD(config.DDD, factory, signal,
                      min_training_size=config.MIN_TRAINING_SIZE,
                      listener=logger.on_event)
    logger.attach(ddd)
    t = 0
    while stream.has_more_instances():
        logger.record(t, ddd.step(t, stream.next_instance()))
        t += 1
    assert t == length, f"{dataset}: {t} instances, expected {length}"
    ddd.finish(length - 1)

    # Keep only the fields that need no ground truth.
    episodes = logger.episode_rows([], config.MAX_DELAY,
                                   config.DUPLICATES_AS_FP)
    gt_free = [f for f in EPISODE_FIELDS
               if f not in ("label", "drift_idx", "delay", "true_type",
                            "true_recurring", "kind_correct",
                            "pool_add_tag", "recalled_tag",
                            "true_concept_at_recall", "recall_hit")]
    full = logger.summary(episodes, list(signal.alarms), [], length,
                          config.MAX_DELAY, config.DUPLICATES_AS_FP)
    summary = {k: full[k] for k in full
               if k in ("acc", "acc_all", "n_detections", "frac_after",
                        "mean_steps_after", "final_cap_fast",
                        "final_cap_slow", "n_recalls")
               or k.startswith("exit_")}

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "steps.csv",
               [f for f in STEP_FIELDS if f != "concept"], logger.rows)
    _write_csv(out / "episodes.csv", gt_free, episodes)
    np.save(out / "correct.npy", logger.correct)
    (out / "alarms.json").write_text(_json({
        "alarms": list(signal.alarms), "raw_fires": signal.raw_fires,
        **signal.logs}))
    (out / "meta.json").write_text(_json(  # last: marks completion
        _meta(dataset, seed, method, length, summary, t0, params)))
    return out


def real_run_spec(spec) -> str:
    """Pool-friendly wrapper."""
    real_run_one(*spec)
    return real_run_name(*spec)
