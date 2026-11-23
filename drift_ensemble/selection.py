"""Parameter selection from tuning runs.

For each detector and each mode (ABRUPT, GRADUAL) separately: the config
with the highest mean TUNE_OBJECTIVE over all tuning streams of that mode
(all generators, all tuning seeds). Ties go to the higher mean precision.
"""

from __future__ import annotations

import json

import pandas as pd

from drift_ensemble import config
from drift_ensemble.config import PURE_MODES

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
