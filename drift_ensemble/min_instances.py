"""Tuning phase 2: one shared minimum M per cluster
(config.MIN_INSTANCES_GRID; the rule is explained above it in config.py).

Runs on the cached tuning streams of the cluster's own mode (the ABRUPT
cluster on the ABRUPT streams, GRADUAL on GRADUAL), with the phase-1 grids:

  RDDM          min_n_instances is replaced by M (one config per
                drift_level x M)
  STEPD, SEED   every config, scored at each M >= its own minimum
                (detectors.own_minimum)
  HDDMWeighted  every config, scored at every M

One task = one (cluster, detector, config) on its mode's streams. A single
detector is never reset explicitly, so its gated alarms at every M come from
ONE ungated replay (detectors.gate_alarms, exact in that case).

Selection: for each cluster and M, each member's best config (highest mean
TUNE_OBJECTIVE; ties -> precision, then params, as in phase 1). The
cluster's score at M is the mean of its two members' best. The M with the
highest score wins (ties -> higher mean precision, then the smaller M).
"""

from __future__ import annotations

import csv
import json

import pandas as pd

from drift_ensemble import config, tuning
from drift_ensemble.detectors import (
    gate_alarms,
    iter_configs,
    own_minimum,
    run_detector_on_errors,
)
from drift_ensemble.selection import METRIC_COLS, _rank
from drift_ensemble.tuning import params_key, score_alarms

RUNS_CSV = config.TUNING_DIR / "phase2_runs.csv"
SUMMARY_CSV = config.TUNING_DIR / "phase2_summary.csv"
ROW_FIELDS = ["cluster", "min_instances", *tuning.ROW_FIELDS]
TASK_KEY = ["cluster", "detector", "params"]


def fitting_minimums(name: str, params: dict, grid=None) -> list[int]:
    """The grid values M at which this config may run in its cluster."""
    grid = config.MIN_INSTANCES_GRID if grid is None else grid
    if name == "RDDM":  # its own minimum IS the cluster's minimum
        return [m for m in grid if m == int(params["min_n_instances"])]
    return [m for m in grid if own_minimum(name, params) <= m]


def all_tasks(grid=None) -> list[tuple[str, str, str]]:
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


def n_rows(task, n_streams_per_mode: int) -> int:
    """Rows a finished task has written."""
    _, name, pjson = task
    return n_streams_per_mode * len(fitting_minimums(name, json.loads(pjson)))


def run_task(task: tuple[str, str, str]) -> list[dict]:
    """One config on every preloaded stream of its cluster's mode, scored
    at every M it fits. Streams come from tuning.worker_init."""
    cluster, name, pjson = task
    params = json.loads(pjson)
    minimums = fitting_minimums(name, params)
    rows = []
    for sid, errors, ground_truth, length in tuning._WORKER_STREAMS:
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
def done_counts(path=None) -> dict[tuple[str, str, str], int]:
    path = RUNS_CSV if path is None else path
    done: dict[tuple[str, str, str], int] = {}
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = tuple(row[k] for k in TASK_KEY)
            done[key] = done.get(key, 0) + 1
    return done


def append_rows(rows, path=None) -> None:
    path = RUNS_CSV if path is None else path
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def load_runs(path=None) -> pd.DataFrame:
    df = pd.read_csv(RUNS_CSV if path is None else path)
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
    params}, ...}: the format workflow.params_for / min_instances_for read.
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
