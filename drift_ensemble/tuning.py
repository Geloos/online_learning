"""Detector tuning on the cached HT errors.

  * tuning     every grid config of every detector on the 102 tuning
               streams -> tuning_runs.csv; the best config per detector x
               mode by TF2, and the clusters (the CLUSTER_SIZE best
               detectors per mode by TF2) -> selected_params.json (+ a
               leave-one-generator-out check). Every method of both
               experiments runs with it.
  * screening  extra detectors (DDM, HDDMAverage, ADWIN, KSWIN) as singles
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
from drift_ensemble.detectors import iter_configs, run_detector_on_errors
from drift_ensemble.runs import load_cache
from drift_ensemble.stats import compute_metrics
from streams import stream_id


# ===========================================================================
# Tuning: evaluation
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
    "tp", "fp", "fn", "precision", "recall", "ndt", "tf2",
    "mean_delay", "far",
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
    generator, mode, seed = sid.split(",")
    return {
        "generator": generator,
        "mode": mode,
        "seed": int(seed),
        **metrics,
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


# Resumable CSVs (tuning and screening): a task is done once it
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
# Tuning: selection
#
# Parameter selection from tuning runs.
#
# For each detector and each mode (ABRUPT, GRADUAL) separately: the config
# with the highest mean TUNE_OBJECTIVE (TF2) over all tuning streams of that
# mode (all generators, all tuning seeds). Ties go to the higher mean
# precision. Then the clusters: per mode, the CLUSTER_SIZE detectors whose
# selected config has the highest mean TF2 (select_clusters).
# ===========================================================================
METRIC_COLS = ["tf2", "precision", "recall", "ndt", "mean_delay", "far"]


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


def select_clusters(summary: pd.DataFrame, size: int | None = None
                    ) -> tuple[dict, pd.DataFrame]:
    """({mode: [detectors]}, summary + rank / in_cluster columns): per mode,
    the ``size`` (default config.CLUSTER_SIZE) detectors with the highest
    mean objective of their selected config. Ties: higher mean precision,
    then DETECTOR_ORDER. A detector may be picked for both modes."""
    size = config.CLUSTER_SIZE if size is None else size
    objective = summary["objective"].iloc[0]
    out = summary.copy()
    out["_order"] = out["detector"].map(config.DETECTOR_ORDER.index)
    out = out.sort_values(
        ["mode", f"mean_{objective}", "mean_precision", "_order"],
        ascending=[True, False, False, True], kind="stable")
    out["rank"] = out.groupby("mode").cumcount() + 1
    out["in_cluster"] = out["rank"] <= size
    clusters = {mode: [d for d in config.DETECTOR_ORDER
                       if d in set(out[(out["mode"] == mode)
                                       & out["in_cluster"]]["detector"])]
                for mode in PURE_MODES}
    out = out.drop(columns="_order").sort_values(["mode", "rank"])
    return clusters, out.reset_index(drop=True)


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
# Screening
#
# Screening of extra detectors (config.SCREENING_GRIDS: DDM, HDDMAverage,
# ADWIN, KSWIN) against the study's 4, as single detectors on the cached
# HT errors only. No new streams, no change to the roster or the ensembles.
#
#   1. tune     every screening-grid config on the tuning streams, plus the
#               study's 4 with their selected configs, so every number comes
#               from the same scoring code
#               -> results/results_drift_ensemble/screening/tuning_runs.csv
#   2. select   per new detector x mode (ABRUPT, GRADUAL): the config with the
#               highest mean TF2 over that mode's tuning streams (the
#               select_params rule), among the configs of the CURRENT grid
#               (rows of configs dropped from a grid are ignored); the
#               study's 4 keep tuning/selected_params.json
#   3. test     every detector with its mode's config on the cached test
#               streams of that mode
#               -> results/results_drift_ensemble/screening/test_runs.csv
#   4. summary  two rows per detector x mode: the tuned config and capymoa's
#               default (config.DEFAULT_PARAMS); mean test TF2 / precision /
#               recall / NDT, median delay
#               -> statistics/drift_ensemble/screening.csv
#
# On the test streams every detector runs with the config tuned for the
# stream's mode (like a cluster member, but for all 4 of the study's
# detectors and both modes).
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
    selected_params.json) and the new detectors (best mean TF2 here)."""
    new, _ = select_params(tuning_df, "tf2", SCREENING_DETECTORS)
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
    config) and "default" (capymoa's): mean TF2 over the mode's tuning
    streams, and means of TF2 / precision / recall / NDT over the mode's
    test streams (delay = median of the per-stream mean delays, as delays
    are skewed). Sorted by setting, then best-first by mean test TF2
    within each mode."""
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
            "tuning_tf2": tr["tf2"].mean(), "n_test": len(te),
            "test_tf2": te["tf2"].mean(),
            "test_tf2_std": te["tf2"].std(),
            "test_precision": te["precision"].mean(),
            "test_recall": te["recall"].mean(),
            "test_ndt": te["ndt"].mean(),
            "test_delay": te["mean_delay"].median(),
        })
    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], categories=list(PURE_MODES),
                                ordered=True)
    df["setting"] = pd.Categorical(df["setting"], categories=list(settings),
                                   ordered=True)
    df = df.sort_values(["mode", "setting", "test_tf2"],
                        ascending=[True, True, False],
                        kind="stable").reset_index(drop=True)
    df["setting"] = df["setting"].astype(str)
    df["mode"] = df["mode"].astype(str)
    return df


def only_tasks(df: pd.DataFrame, tasks) -> pd.DataFrame:
    """The rows of the given (detector, params_json) tasks. Rows of
    configs since dropped from a grid stay in the CSV but are not used."""
    keep = set(tasks)
    return df[[k in keep for k in zip(df["detector"], df["params"])]]


def load_screening_rows(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ROW_FIELDS)
    df = pd.read_csv(path)
    return df.drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last")


def screening_selected_json(selected: dict) -> str:
    return json.dumps(selected, indent=2, sort_keys=True)
