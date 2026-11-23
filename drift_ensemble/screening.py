"""Screening of extra capymoa detectors (config.SCREENING_GRIDS: DDM,
HDDMAverage, ADWIN) against the study's 4, as single detectors on the cached
HT errors only. No new streams, no change to the roster or the ensembles.

  1. tune     every screening-grid config on the 18 tuning streams, plus the
              study's 4 with their selected configs, so every number comes
              from the same scoring code
              -> results/results_drift_ensemble/screening/tuning_runs.csv
  2. select   per new detector x mode (ABRUPT, GRADUAL): the config with the
              highest mean F1 over that mode's 9 tuning streams (the
              selection.select_params rule); the study's 4 keep
              tuning/selected_params.json
  3. test     every detector with its mode's config on the cached test
              streams of that mode (15 each)
              -> results/results_drift_ensemble/screening/test_runs.csv
  4. summary  two rows per detector x mode: the tuned config and capymoa's
              default (config.DEFAULT_PARAMS)
              -> statistics/drift_ensemble/screening.csv

On the test streams the study's 4 run with the config tuned for the
stream's mode, unlike single_* in the roster, which always use their
cluster's config.
"""

from __future__ import annotations

import csv
import itertools
import json
import time

import pandas as pd

from drift_ensemble import cli, config
from drift_ensemble.config import PURE_MODES
from drift_ensemble.detectors import iter_configs
from drift_ensemble.selection import select_params
from drift_ensemble.tuning import (
    ROW_FIELDS,
    params_key,
    run_task,
    worker_init,
)

NEW_DETECTORS = list(config.SCREENING_GRIDS)
TUNING_CSV = config.SCREENING_DIR / "tuning_runs.csv"
TEST_CSV = config.SCREENING_DIR / "test_runs.csv"
SUMMARY_CSV = config.STATISTICS_DIR / "screening.csv"


# ---------------------------------------------------------------------------
# Tasks and selection
# ---------------------------------------------------------------------------
def _unique(tasks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(tasks))


def _defaults() -> list[tuple[str, str]]:
    return [(d, params_key(config.DEFAULT_PARAMS[d]))
            for d in config.DETECTOR_ORDER + NEW_DETECTORS]


def tuning_tasks(study_selected: dict) -> list[tuple[str, str]]:
    """(detector, params_json) to evaluate on the tuning streams: every
    screening-grid config (defaults included), the study's 4 selected
    configs and their defaults."""
    tasks = [(name, params_key(p)) for name in NEW_DETECTORS
             for p in iter_configs(name, config.SCREENING_GRIDS[name])]
    tasks += [(name, params_key(study_selected[mode][name]))
              for mode in PURE_MODES for name in config.DETECTOR_ORDER]
    return _unique(tasks + _defaults())


def all_selected(tuning_df: pd.DataFrame, study_selected: dict) -> dict:
    """{mode: {detector: params}} for the study's 4 (from
    selected_params.json) and the new detectors (best mean F1 here)."""
    new, _ = select_params(tuning_df, "f1", NEW_DETECTORS)
    return {mode: {**{d: study_selected[mode][d]
                      for d in config.DETECTOR_ORDER},
                   **new[mode]}
            for mode in PURE_MODES}


def test_tasks(selected: dict) -> list[tuple[str, str]]:
    """Every selected config and every default, for the test streams."""
    return _unique([(d, params_key(p)) for mode in PURE_MODES
                    for d, p in selected[mode].items()] + _defaults())


def summarize(tuning_df: pd.DataFrame, test_df: pd.DataFrame,
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
            config.DETECTOR_ORDER + NEW_DETECTORS):
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


# ---------------------------------------------------------------------------
# Resumable evaluation of (detector, config) tasks on cached streams
# ---------------------------------------------------------------------------
def load_rows(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ROW_FIELDS)
    df = pd.read_csv(path)
    return df.drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last")


def _done_counts(path) -> dict:
    done: dict[tuple[str, str], int] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["detector"], row["params"])
                done[key] = done.get(key, 0) + 1
    return done


def _append(path, rows) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def run_stage(tasks, stream_ids: list[str], path, workers: int, yes: bool,
              label: str) -> None:
    """Evaluate every pending task on every stream (rows appended to
    ``path``); timing gate as in the pipeline scripts."""
    done = _done_counts(path)
    pending = [t for t in tasks if done.get(t, 0) < len(stream_ids)]
    print(f"{label}: {len(tasks)} configs x {len(stream_ids)} streams; "
          f"{len(pending)} still to evaluate.")
    if not pending:
        return
    worker_init(stream_ids)
    t0 = time.perf_counter()
    _append(path, run_task(pending[0]))
    unit = time.perf_counter() - t0
    remaining = pending[1:]
    if not remaining:
        return
    cli.gate(unit, len(remaining), workers, yes, "config")
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=worker_init,
                      initargs=(stream_ids,)) as pool:
            for i, rows in enumerate(
                    pool.imap_unordered(run_task, remaining), 1):
                _append(path, rows)
                if i % 10 == 0 or i == len(remaining):
                    print(f"  [{i}/{len(remaining)}] configs done")
    else:
        for i, task in enumerate(remaining, 1):
            _append(path, run_task(task))
            if i % 10 == 0 or i == len(remaining):
                print(f"  [{i}/{len(remaining)}] configs done")


def selected_json(selected: dict) -> str:
    return json.dumps(selected, indent=2, sort_keys=True)
