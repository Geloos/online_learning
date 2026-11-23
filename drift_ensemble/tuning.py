"""Detector tuning on the cached HT errors.

  * phase 1    every grid config of every detector on the 18 tuning
               streams -> tuning_runs.csv; the best config per detector x
               mode -> selected_params.json (+ a leave-one-generator-out
               check). Used by the singles and ens_raw_resetall.
  * phase 2    one shared minimum per cluster; the cluster's detectors
               re-tuned under it -> cluster_params.json. Used by the
               two-vote ensembles (config.PHASE2_METHODS).
  * screening  extra capymoa detectors (DDM, HDDMAverage, ADWIN) as singles
               against the study's 4 (`python main.py screen`; not part of
               the full run)

The steps themselves (timing gate, worker pool, CSV appends) are run by
main.py; everything here is per task and resumable through the CSVs.
"""

from __future__ import annotations

import csv
import itertools
import json

import pandas as pd

from drift_ensemble import config
from drift_ensemble.config import PURE_MODES, tuning_stream_specs
from drift_ensemble.detectors import (
    gate_alarms,
    iter_configs,
    own_minimum,
    run_detector_on_errors,
)
from drift_ensemble.runs import load_cache
from drift_ensemble.stats import compute_metrics, delay_curve
from streams import stream_id


# ===========================================================================
# Phase 1: evaluation
#
# Grid/random search of detector parameters on cached HT errors.
#
# One tuning *task* = one (detector, config) pair evaluated on every tuning
# stream. Tasks are independent, so they run in a multiprocessing pool; each
# spawned worker imports capymoa itself (one JVM per worker, never a forked
# one) and loads the cached error sequences once.
# ===========================================================================
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


def score_alarms(alarms, sid: str, ground_truth: list[dict],
                 length: int) -> dict:
    """Stream id fields + every tuning metric of one alarm list."""
    metrics = compute_metrics(
        alarms, ground_truth, config.MAX_DELAY, length,
        config.MIN_TRAINING_SIZE, config.DUPLICATES_AS_FP,
    )
    dc = delay_curve(alarms, ground_truth, config.DELAY_GRID,
                     config.DUPLICATES_AS_FP)
    generator, mode, seed = sid.split(",")
    return {
        "generator": generator,
        "mode": mode,
        "seed": int(seed),
        **metrics,
        "delay_auc": dc["delay_auc"],
    }


def evaluate_config_on_stream(
    detector: str,
    params: dict,
    sid: str,
    errors,
    ground_truth: list[dict],
    length: int,
) -> dict:
    alarms = run_detector_on_errors(detector, params, errors)
    return {
        "detector": detector,
        "params": params_key(params),
        **score_alarms(alarms, sid, ground_truth, length),
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


# Resumable CSVs (phase 1, phase 2 and screening): a task is done once it
# has written all its rows.
def done_counts(path, key=("detector", "params")) -> dict:
    """Rows already written, per task key."""
    done: dict[tuple, int] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                k = tuple(row[f] for f in key)
                done[k] = done.get(k, 0) + 1
    return done


def append_rows(path, fields, rows) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Phase 1: selection
#
# Parameter selection from tuning runs.
#
# For each detector and each mode (ABRUPT, GRADUAL) separately: the config
# with the highest mean TUNE_OBJECTIVE over all tuning streams of that mode
# (all generators, all tuning seeds). Ties go to the higher mean precision.
# ===========================================================================
METRIC_COLS = ["f1", "f2", "precision", "recall", "mean_delay", "far",
               "delay_auc"]


def load_tuning_runs(path=None) -> pd.DataFrame:
    path = path or (config.TUNING_DIR / "tuning_runs.csv")
    df = pd.read_csv(path)
    # A task interrupted mid-write can leave a partial row set that was later
    # rerun; keep the latest copy of each (config, stream) evaluation.
    df = df.drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last",
    )
    return df


def _rank(group_means: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Sort configs best-first: objective desc, then precision desc, then
    params asc (full determinism)."""
    return group_means.sort_values(
        by=[objective, "precision", "params"],
        ascending=[False, False, True],
    )


def select_params(df: pd.DataFrame, objective: str | None = None,
                  detectors: list[str] | None = None
                  ) -> tuple[dict, pd.DataFrame]:
    """Returns ({'ABRUPT': {detector: params}, 'GRADUAL': ...}, summary df).

    The summary has one row per detector x mode: the winning config with its
    mean metrics over the tuning streams of that mode. ``detectors``
    defaults to config.DETECTOR_ORDER."""
    objective = objective or config.TUNE_OBJECTIVE
    detectors = config.DETECTOR_ORDER if detectors is None else detectors
    selected: dict[str, dict] = {m: {} for m in PURE_MODES}
    rows = []
    for mode in PURE_MODES:
        for detector in detectors:
            sub = df[(df["detector"] == detector) & (df["mode"] == mode)]
            if sub.empty:
                raise ValueError(
                    f"no tuning rows for {detector} on {mode} streams")
            means = (sub.groupby("params")[METRIC_COLS]
                     .mean().reset_index())
            counts = sub.groupby("params").size()
            expected = sub[["generator", "seed"]].drop_duplicates().shape[0]
            incomplete = counts[counts < expected]
            if not incomplete.empty:
                print(f"WARNING: {detector}/{mode}: "
                      f"{len(incomplete)} configs evaluated on fewer than "
                      f"{expected} streams")
            best = _rank(means, objective).iloc[0]
            selected[mode][detector] = json.loads(best["params"])
            rows.append({
                "detector": detector,
                "mode": mode,
                "params": best["params"],
                "n_streams": int(counts[best["params"]]),
                "objective": objective,
                **{f"mean_{c}": best[c] for c in METRIC_COLS},
            })
    summary = pd.DataFrame(rows)
    return selected, summary


def logo_check(df: pd.DataFrame, objective: str | None = None
               ) -> pd.DataFrame:
    """Leave-one-generator-out robustness check: select on 2 generators,
    report the chosen config's score on the held-out generator, next to the
    best score achievable there."""
    objective = objective or config.TUNE_OBJECTIVE
    rows = []
    generators = sorted(df["generator"].unique())
    for mode in PURE_MODES:
        for detector in config.DETECTOR_ORDER:
            sub = df[(df["detector"] == detector) & (df["mode"] == mode)]
            for holdout in generators:
                train = sub[sub["generator"] != holdout]
                test = sub[sub["generator"] == holdout]
                if train.empty or test.empty:
                    continue
                train_means = (train.groupby("params")[METRIC_COLS]
                               .mean().reset_index())
                chosen = _rank(train_means, objective).iloc[0]["params"]
                test_means = test.groupby("params")[objective].mean()
                achieved = test_means.get(chosen)
                rows.append({
                    "detector": detector,
                    "mode": mode,
                    "holdout_generator": holdout,
                    "chosen_params": chosen,
                    f"{objective}_on_holdout": achieved,
                    f"best_{objective}_on_holdout": test_means.max(),
                    "gap": (test_means.max() - achieved
                            if achieved is not None else None),
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Phase 2: one shared minimum per cluster
#
# Tuning phase 2: one shared minimum M per cluster
# (config.MIN_INSTANCES_GRID; the rule is explained above it in config.py).
#
# Runs on the cached tuning streams of the cluster's own mode (the ABRUPT
# cluster on the ABRUPT streams, GRADUAL on GRADUAL), with the phase-1 grids:
#
#   RDDM          min_n_instances is replaced by M (one config per
#                 drift_level x M)
#   STEPD, SEED   every config, scored at each M >= its own minimum
#                 (detectors.own_minimum)
#   HDDMWeighted  every config, scored at every M
#
# One task = one (cluster, detector, config) on its mode's streams. A single
# detector is never reset explicitly, so its gated alarms at every M come from
# ONE ungated replay (detectors.gate_alarms, exact in that case).
#
# Selection: for each cluster and M, each member's best config (highest mean
# TUNE_OBJECTIVE; ties -> precision, then params, as in phase 1). The
# cluster's score at M is the mean of its two members' best. The M with the
# highest score wins (ties -> higher mean precision, then the smaller M).
# ===========================================================================
PHASE2_RUNS_CSV = config.TUNING_DIR / "phase2_runs.csv"
PHASE2_SUMMARY_CSV = config.TUNING_DIR / "phase2_summary.csv"
PHASE2_ROW_FIELDS = ["cluster", "min_instances", *ROW_FIELDS]
PHASE2_TASK_KEY = ["cluster", "detector", "params"]


def fitting_minimums(name: str, params: dict, grid=None) -> list[int]:
    """The grid values M at which this config may run in its cluster."""
    grid = config.MIN_INSTANCES_GRID if grid is None else grid
    if name == "RDDM":  # its own minimum IS the cluster's minimum
        return [m for m in grid if m == int(params["min_n_instances"])]
    return [m for m in grid if own_minimum(name, params) <= m]


def phase2_tasks(grid=None) -> list[tuple[str, str, str]]:
    """Every (cluster, detector, params_json), in deterministic order.
    Configs that fit no M in the grid are left out."""
    grid = config.MIN_INSTANCES_GRID if grid is None else grid
    tasks = []
    for cluster, members in config.CLUSTER_MEMBERS.items():
        assert cluster in config.PURE_MODES, cluster
        for name in members:
            configs = iter_configs(name)
            if name == "RDDM":
                configs = [{**p, "min_n_instances": m}
                           for m in grid for p in configs]
            seen = set()
            for p in configs:
                key = params_key(p)
                if key not in seen and fitting_minimums(name, p, grid):
                    seen.add(key)
                    tasks.append((cluster, name, key))
    return tasks


def phase2_n_rows(task, n_streams_per_mode: int) -> int:
    """Rows a finished task has written."""
    _, name, pjson = task
    return n_streams_per_mode * len(fitting_minimums(name, json.loads(pjson)))


def phase2_run_task(task: tuple[str, str, str]) -> list[dict]:
    """One config on every preloaded stream of its cluster's mode, scored
    at every M it fits. Streams come from worker_init."""
    cluster, name, pjson = task
    params = json.loads(pjson)
    minimums = fitting_minimums(name, params)
    rows = []
    for sid, errors, ground_truth, length in _WORKER_STREAMS:
        if sid.split(",")[1] != cluster:
            continue
        raw = run_detector_on_errors(name, params, errors)
        for m in minimums:
            rows.append({
                "cluster": cluster,
                "min_instances": m,
                "detector": name,
                "params": pjson,
                **score_alarms(gate_alarms(raw, errors, m), sid,
                               ground_truth, length),
            })
    return rows


# ---------------------------------------------------------------------------
# Resumable CSV
# ---------------------------------------------------------------------------


def load_phase2_runs(path=None) -> pd.DataFrame:
    df = pd.read_csv(PHASE2_RUNS_CSV if path is None else path)
    # A task interrupted mid-write may have been rerun: keep the last copy.
    return df.drop_duplicates(
        subset=["cluster", "min_instances", "detector", "params",
                "generator", "mode", "seed"], keep="last")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_min_instances(df: pd.DataFrame, objective: str | None = None,
                         phase1_summary: pd.DataFrame | None = None
                         ) -> tuple[dict, pd.DataFrame]:
    """Returns (cluster_params, summary).

    cluster_params = {"min_instances": {cluster: M}, cluster: {detector:
    params}, ...}: the format runs.params_for / min_instances_for read.
    summary has one row per cluster x M x member: the member's best config
    at that M, the cluster's score, and whether that M was chosen.
    ``phase1_summary`` (tuning_summary.csv) adds the member's phase-1 score
    on the same streams, without a shared minimum, for comparison."""
    objective = objective or config.TUNE_OBJECTIVE
    phase1 = {}
    if phase1_summary is not None:
        phase1 = {(r["detector"], r["mode"]): r[f"mean_{objective}"]
                  for _, r in phase1_summary.iterrows()}
    rows = []
    for cluster, members in config.CLUSTER_MEMBERS.items():
        sub_c = df[df["cluster"] == cluster]
        expected = sub_c[["generator", "seed"]].drop_duplicates().shape[0]
        for m in sorted(sub_c["min_instances"].unique()):
            best, n_streams = {}, {}
            for name in members:
                sub = sub_c[(sub_c["min_instances"] == m)
                            & (sub_c["detector"] == name)]
                if sub.empty:
                    break
                counts = sub.groupby("params").size()
                if (counts < expected).any():
                    print(f"WARNING: {cluster} M={m} {name}: "
                          f"{int((counts < expected).sum())} configs on "
                          f"fewer than {expected} streams")
                means = (sub.groupby("params")[METRIC_COLS]
                         .mean().reset_index())
                best[name] = _rank(means, objective).iloc[0]
                n_streams[name] = int(counts[best[name]["params"]])
            if len(best) < len(members):
                continue  # a member has no config that fits this M
            score = sum(best[n][objective] for n in members) / len(members)
            precision = (sum(best[n]["precision"] for n in members)
                         / len(members))
            for name in members:
                rows.append({
                    "cluster": cluster,
                    "min_instances": int(m),
                    "objective": objective,
                    "cluster_score": score,
                    "cluster_precision": precision,
                    "detector": name,
                    "params": best[name]["params"],
                    "n_streams": n_streams[name],
                    **{f"mean_{c}": best[name][c] for c in METRIC_COLS},
                    f"phase1_mean_{objective}": phase1.get((name, cluster)),
                })
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("no phase-2 rows to select from")
    summary["chosen"] = False
    chosen: dict = {"min_instances": {}}
    for cluster in config.CLUSTER_MEMBERS:
        s = summary[summary["cluster"] == cluster]
        if s.empty:
            raise ValueError(f"no M in the grid fits every member of the "
                             f"{cluster} cluster")
        per_m = s.drop_duplicates("min_instances")
        win = int(per_m.sort_values(
            ["cluster_score", "cluster_precision", "min_instances"],
            ascending=[False, False, True]).iloc[0]["min_instances"])
        chosen["min_instances"][cluster] = win
        chosen[cluster] = {
            r["detector"]: json.loads(r["params"])
            for _, r in s[s["min_instances"] == win].iterrows()
        }
        summary.loc[(summary["cluster"] == cluster)
                    & (summary["min_instances"] == win), "chosen"] = True
    return chosen, summary


# ===========================================================================
# Screening
#
# Screening of extra capymoa detectors (config.SCREENING_GRIDS: DDM,
# HDDMAverage, ADWIN) against the study's 4, as single detectors on the cached
# HT errors only. No new streams, no change to the roster or the ensembles.
#
#   1. tune     every screening-grid config on the 18 tuning streams, plus the
#               study's 4 with their selected configs, so every number comes
#               from the same scoring code
#               -> results/results_drift_ensemble/screening/tuning_runs.csv
#   2. select   per new detector x mode (ABRUPT, GRADUAL): the config with the
#               highest mean F1 over that mode's 9 tuning streams (the
#               select_params rule); the study's 4 keep
#               tuning/selected_params.json
#   3. test     every detector with its mode's config on the cached test
#               streams of that mode (15 each)
#               -> results/results_drift_ensemble/screening/test_runs.csv
#   4. summary  two rows per detector x mode: the tuned config and capymoa's
#               default (config.DEFAULT_PARAMS)
#               -> statistics/drift_ensemble/screening.csv
#
# On the test streams the study's 4 run with the config tuned for the
# stream's mode, unlike single_* in the roster, which always use their
# cluster's config.
# ===========================================================================
SCREENING_DETECTORS = list(config.SCREENING_GRIDS)
SCREENING_TUNING_CSV = config.SCREENING_DIR / "tuning_runs.csv"
SCREENING_TEST_CSV = config.SCREENING_DIR / "test_runs.csv"
SCREENING_SUMMARY_CSV = config.STATISTICS_DIR / "screening.csv"


# ---------------------------------------------------------------------------
# Tasks and selection
# ---------------------------------------------------------------------------
def _unique_tasks(tasks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(tasks))


def _screening_defaults() -> list[tuple[str, str]]:
    return [(d, params_key(config.DEFAULT_PARAMS[d]))
            for d in config.DETECTOR_ORDER + SCREENING_DETECTORS]


def screening_tuning_tasks(study_selected: dict) -> list[tuple[str, str]]:
    """(detector, params_json) to evaluate on the tuning streams: every
    screening-grid config (defaults included), the study's 4 selected
    configs and their defaults."""
    tasks = [(name, params_key(p)) for name in SCREENING_DETECTORS
             for p in iter_configs(name, config.SCREENING_GRIDS[name])]
    tasks += [(name, params_key(study_selected[mode][name]))
              for mode in PURE_MODES for name in config.DETECTOR_ORDER]
    return _unique_tasks(tasks + _screening_defaults())


def screening_selected(tuning_df: pd.DataFrame, study_selected: dict) -> dict:
    """{mode: {detector: params}} for the study's 4 (from
    selected_params.json) and the new detectors (best mean F1 here)."""
    new, _ = select_params(tuning_df, "f1", SCREENING_DETECTORS)
    return {mode: {**{d: study_selected[mode][d]
                      for d in config.DETECTOR_ORDER},
                   **new[mode]}
            for mode in PURE_MODES}


def screening_test_tasks(selected: dict) -> list[tuple[str, str]]:
    """Every selected config and every default, for the test streams."""
    return _unique_tasks([(d, params_key(p)) for mode in PURE_MODES
                          for d, p in selected[mode].items()]
                         + _screening_defaults())


def screening_summary(tuning_df: pd.DataFrame, test_df: pd.DataFrame,
                      selected: dict) -> pd.DataFrame:
    """Two rows per detector x mode, setting = "tuned" (its selected
    config) and "default" (capymoa's): mean F1 over the mode's tuning
    streams, and medians over the mode's test streams (delay = median of
    the per-stream mean delays). Sorted by setting, then best-first by test
    F1 within each mode."""
    settings = {"tuned": lambda mode, det: selected[mode][det],
                "default": lambda mode, det: config.DEFAULT_PARAMS[det]}

    def pick(df, det, pj, mode):
        return df[(df["detector"] == det) & (df["params"] == pj)
                  & (df["mode"] == mode)]

    rows = []
    for mode, (setting, params_of), det in itertools.product(
            PURE_MODES, settings.items(),
            config.DETECTOR_ORDER + SCREENING_DETECTORS):
        pj = params_key(params_of(mode, det))
        tr = pick(tuning_df, det, pj, mode)
        te = pick(test_df, det, pj, mode)
        rows.append({
            "mode": mode, "setting": setting, "detector": det,
            "in_study": det in config.DETECTOR_ORDER, "params": pj,
            "tuning_f1": tr["f1"].mean(), "n_test": len(te),
            "test_f1": te["f1"].median(),
            "test_f1_mean": te["f1"].mean(),
            "test_precision": te["precision"].median(),
            "test_recall": te["recall"].median(),
            "test_delay": te["mean_delay"].median(),
        })
    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], categories=list(PURE_MODES),
                                ordered=True)
    df["setting"] = pd.Categorical(df["setting"], categories=list(settings),
                                   ordered=True)
    df = df.sort_values(["mode", "setting", "test_f1"],
                        ascending=[True, True, False],
                        kind="stable").reset_index(drop=True)
    df["setting"] = df["setting"].astype(str)
    df["mode"] = df["mode"].astype(str)
    return df


def load_screening_rows(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ROW_FIELDS)
    df = pd.read_csv(path)
    return df.drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last")


def screening_selected_json(selected: dict) -> str:
    return json.dumps(selected, indent=2, sort_keys=True)
