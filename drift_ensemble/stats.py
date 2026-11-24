"""Scoring alarms against the true drifts, and the detection statistics of
every method.

  * scoring     matching alarms to drifts (TP / FP / FN, delays), metrics
                and delay curves; used by tuning, the statistics and DDD
  * statistics  per method x mode: mean / std of precision, recall, NDT
                and TF2, median delay,
                delay curves and bar charts, plus significance tests,
                per-drift-type recall, first-to-fire, detector diversity,
                sensitivity levels, runtime and the tuning heatmaps
                -> statistics/drift_ensemble/
"""

from __future__ import annotations

import itertools
import json
import math
from bisect import bisect_right
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_ensemble import config
from drift_ensemble.runs import load_runs, load_selected_params


# ===========================================================================
# Scoring
#
# Alarm <-> drift matching and detection metrics.
#
# The same matching rule is used everywhere: tuning, single detectors and the
# ensemble.
#
# Matching rule
# -------------
# The acceptable window of drift k is [start_k, end_k + max_delay].
# For each drift, the FIRST alarm inside its window is a TP with
# delay = alarm - start_k. Further alarms inside the same window are
# duplicates: counted as FP if ``duplicates_as_fp`` else ignored. Alarms
# outside every window are FP. Drifts with no alarm in their window are FN.
# Windows must not overlap (asserted); alarms are assigned to the earliest
# unmatched drift whose window contains them.
#
# TF2 (timely F2, the study's one detection score since 2026-11-24; it
# replaced F1 everywhere)
# ----------------------------------------------------------------------
# NDT_k = (t_detect - start_k) / (end_k - start_k + max_delay) is the
# normalized detection time of a detected drift: 0 at the drift start, 1 at
# the end of its window (an abrupt drift's window is max_delay long, a
# gradual one's is its width + max_delay). NDT = mean over detected drifts.
# Timely recall R_t = (1/N) * sum over detected drifts of (1 - NDT_k)
# = recall * (1 - NDT): a late detection earns less than a fast one, a
# missed drift earns 0. TF2 = F2(precision, R_t) = 5 P R_t / (4 P + R_t):
# recall (now timely) counts twice as much as precision, and precision
# still stops a detector that fires all the time.
# ===========================================================================
@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    delays: list[int] = field(default_factory=list)  # one per TP, drift order
    matched: list[int | None] = field(default_factory=list)  # alarm per drift
    tp_alarms: list[int] = field(default_factory=list)
    fp_alarms: list[int] = field(default_factory=list)  # outside every window
    duplicate_alarms: list[int] = field(default_factory=list)


def match_alarms(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    duplicates_as_fp: bool = True,
) -> MatchResult:
    """Apply the matching rule above. ``ground_truth`` is a list of dicts with
    at least 'start' and 'end' (end == start for abrupt drifts)."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    starts = [d["start"] for d in truth]
    ends = [d["end"] + max_delay for d in truth]
    for i in range(len(truth) - 1):
        assert ends[i] < starts[i + 1], (
            f"overlapping drift windows: [{starts[i]}, {ends[i]}] and "
            f"[{starts[i + 1]}, {ends[i + 1]}] — not supported"
        )

    matched: list[int | None] = [None] * len(truth)
    tp_alarms: list[int] = []
    fp_alarms: list[int] = []
    duplicate_alarms: list[int] = []

    for a in sorted(alarms):
        k = bisect_right(starts, a) - 1  # last drift with start <= a
        if k >= 0 and a <= ends[k]:
            if matched[k] is None:
                matched[k] = a
                tp_alarms.append(a)
            else:
                duplicate_alarms.append(a)
        else:
            fp_alarms.append(a)

    tp = sum(m is not None for m in matched)
    fn = len(truth) - tp
    fp = len(fp_alarms) + (len(duplicate_alarms) if duplicates_as_fp else 0)
    delays = [m - d["start"] for m, d in zip(matched, truth) if m is not None]
    return MatchResult(
        tp=tp, fp=fp, fn=fn, delays=delays, matched=matched,
        tp_alarms=tp_alarms, fp_alarms=fp_alarms,
        duplicate_alarms=duplicate_alarms,
    )


def _fbeta(precision: float, recall: float, beta: float) -> float:
    denom = beta * beta * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denom


def ndt_values(m: MatchResult, ground_truth: list[dict],
               max_delay: int) -> list[float]:
    """NDT of every detected drift (drift order): (t_detect - start) /
    (end - start + max_delay), in [0, 1]."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    out = []
    for a, d in zip(m.matched, truth):
        if a is None:
            continue
        window = d["end"] - d["start"] + max_delay
        out.append((a - d["start"]) / window if window > 0 else 0.0)
    return out


def _scores(m: MatchResult, ground_truth: list[dict],
            max_delay: int) -> dict:
    """Precision, recall, NDT and TF2 of one match (see TF2 above)."""
    precision = m.tp / (m.tp + m.fp) if (m.tp + m.fp) else 0.0
    n_drifts = m.tp + m.fn
    recall = m.tp / n_drifts if n_drifts else 0.0
    ndt = ndt_values(m, ground_truth, max_delay)
    timely_recall = (sum(1.0 - v for v in ndt) / n_drifts
                     if n_drifts else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "ndt": float(sum(ndt) / len(ndt)) if ndt else math.nan,
        "tf2": _fbeta(precision, timely_recall, 2.0),
    }


def compute_metrics(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    n_instances: int,
    min_training_size: int,
    duplicates_as_fp: bool = True,
) -> dict:
    """TP/FP/FN, precision, recall, NDT, TF2, mean delay (TPs only) and
    false alarm rate per 1000 instances (after the warm-up period)."""
    m = match_alarms(alarms, ground_truth, max_delay, duplicates_as_fp)
    monitored_k = (n_instances - min_training_size) / 1000.0
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        **_scores(m, ground_truth, max_delay),
        "mean_delay": float(sum(m.delays) / len(m.delays)) if m.delays else math.nan,
        "far": m.fp / monitored_k if monitored_k > 0 else math.nan,
    }


def delay_curve(
    alarms: list[int],
    ground_truth: list[dict],
    delay_grid: list[int],
    duplicates_as_fp: bool = True,
) -> list[dict]:
    """Precision, recall, NDT and TF2 for each max delay in the grid (the
    alarms rescored with that max delay)."""
    return [{"max_delay": d,
             **_scores(match_alarms(alarms, ground_truth, d,
                                    duplicates_as_fp), ground_truth, d)}
            for d in delay_grid]


def per_type(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    duplicates_as_fp: bool = True,
) -> dict:
    """Recall and mean delay restricted to the drifts of each type.

    Precision is only defined overall: an FP cannot be assigned to a type.
    """
    truth = sorted(ground_truth, key=lambda d: d["start"])
    m = match_alarms(alarms, truth, max_delay, duplicates_as_fp)
    out: dict[str, dict] = {}
    for dtype in ("ABRUPT", "GRADUAL"):
        idx = [i for i, d in enumerate(truth) if d["type"] == dtype]
        if not idx:
            continue
        hits = [i for i in idx if m.matched[i] is not None]
        delays = [m.matched[i] - truth[i]["start"] for i in hits]
        out[dtype] = {
            "n_drifts": len(idx),
            "recall": len(hits) / len(idx),
            "mean_delay": float(sum(delays) / len(delays)) if delays else math.nan,
        }
    return out


# ===========================================================================
# Statistics
#
# Detection statistics of the drift-ensemble experiment for every roster
# method (`python main.py stats`).
#
# Reads ONLY the run folders under results/results_drift_ensemble/test/
# (never reruns streams) and writes to statistics/drift_ensemble/:
#
#   runs.csv                  one row per run (stream x method): TP, FP, FN,
#                             precision, recall, NDT, TF2, mean delay, false
#                             alarms per 1000 instances (far), duplicate
#                             alarms (n_dup: extra alarms on an already
#                             detected drift, counted in fp)
#   delays.csv                one row per detected real drift: its delay
#                             and NDT
#   metrics.csv               one row per method x mode: summed TP/FP/FN and
#                             n_dup, mean and std of precision / recall /
#                             TF2 / NDT / far over the mode's runs, median
#                             delay over its detected drifts
#   delay_curve.csv           mean precision / recall / TF2 over runs for
#                             every max delay in STATS["delay_grid"]
#   <MODE>/singles.png        bar chart of metrics.csv: per single member
#                             the mean precision, recall and TF2 with std
#                             error bars (top row, 0-1) and its median delay
#                             (bottom row, in instances, same delay scale
#                             for both charts)
#   <MODE>/ensembles.png      the same for the ensemble methods
#   <MODE>/delay_curve.png    the delay curves, one line per method
#   significance.csv          Wilcoxon signed-rank on TF2 per mode: each
#                             ensemble vs each single, each cluster ensemble
#                             vs every other ensemble
#   per_type_mixed.csv        recall and mean delay on the abrupt vs the
#                             gradual drifts of MIXED streams
#   first_to_fire.csv         per true drift, which single member fired
#                             first inside its window
#   diversity.csv             per member pair: Q-statistic and
#                             disagreement of their detected/missed drifts
#   sensitivity_summary.csv   ens_cluster_sens: final sensitivity level per
#                             member, and how often it was nudged
#   runtime.csv               mean and total run time per method
#   tuning_heatmaps/          mean tuning TF2 over two params per detector x
#                             mode (tuning_runs.csv)
#
# The methods are the ones found in the run folders: the singles (named
# after the cluster members of the tuning) ordered by cluster and
# DETECTOR_ORDER, then the ensembles. With a selected_params.json, runs made
# with other params are skipped as stale.
#
# Why mean for P/R/TF2 but median for delay: the 51 runs of a method x mode
# are 3 generators x 17 seeds, so a method that fails on one whole generator
# fails on a third of the runs, and a median would hide that. The scores are
# bounded in [0, 1], so a mean is safe. Delays are skewed with a long tail,
# so they use the median.
#
# Scoring is match_alarms with config.MAX_DELAY: drift k is
# detected by the first alarm in [start_k, end_k + max delay], delay = alarm -
# start_k, extra alarms are FP when config.DUPLICATES_AS_FP. The warm-up comes
# from each run's snapshot.
# ===========================================================================
KEYS = ["generator", "mode", "seed", "method"]
METRICS = ["precision", "recall", "tf2"]  # the bar charts and delay curves
# Fixed colour per method (roster order), so a method keeps its colour in
# every plot; singles are dashed, ensembles solid.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Bar charts: one colour per metric (the delay bar has its own row).
METRIC_COLORS = {"precision": "#2a78d6", "recall": "#eb6834",
                 "tf2": "#1baf7a", "delay": "#6250d6"}
METRIC_NAMES = {"precision": "precision", "recall": "recall", "tf2": "TF2"}
CLUSTER_METHODS = ["ens_cluster_vote", "ens_cluster_sens"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _member_parts(member: str) -> tuple[str, str]:
    """'HDDMWeighted_ABRUPT' -> ('HDDMWeighted', 'ABRUPT')."""
    det, _, cluster = member.rpartition("_")
    return det, cluster


def roster(methods) -> list[str]:
    """Methods in report order: the singles by cluster (PURE_MODES order)
    and DETECTOR_ORDER, then the ensembles in config.ENSEMBLE_METHODS
    order."""
    def index(seq, x):
        return list(seq).index(x) if x in seq else len(seq)

    def key(m):
        if m in config.ENSEMBLE_METHODS:
            return (1, config.ENSEMBLE_METHODS.index(m), 0, m)
        det, cluster = _member_parts(m.removeprefix("single_"))
        return (0, index(config.PURE_MODES, cluster),
                index(config.DETECTOR_ORDER, det), m)
    return sorted(set(methods), key=key)


def _groups(methods) -> dict[str, list[str]]:
    r = roster(methods)
    return {"singles": [m for m in r if m.startswith("single_")],
            "ensembles": [m for m in r if not m.startswith("single_")]}


def _keys(run: dict) -> dict:
    return {k: run["meta"][k] for k in KEYS}


def _truth(run: dict) -> list[dict]:
    return sorted(run["meta"]["ground_truth"], key=lambda d: d["start"])


def _ordered(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sort by mode (config order) and method (roster order), then the
    rest of ``by``."""
    df = df.copy()
    df["mode"] = pd.Categorical(df["mode"], categories=config.MODES,
                                ordered=True)
    df["method"] = pd.Categorical(df["method"],
                                  categories=roster(df["method"]),
                                  ordered=True)
    df = df.sort_values(by, kind="stable").reset_index(drop=True)
    df["mode"] = df["mode"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def run_table(runs) -> pd.DataFrame:
    """One row per run: TP/FP/FN, precision, recall, NDT, TF2, mean delay,
    false alarm rate and duplicate alarms."""
    rows = []
    for r in runs:
        m = compute_metrics(
            r["alarms"], r["meta"]["ground_truth"], config.MAX_DELAY,
            r["meta"]["length"], r["meta"]["config"]["MIN_TRAINING_SIZE"],
            config.DUPLICATES_AS_FP)
        # extra alarms inside an already-matched drift window
        n_dup = len(match_alarms(r["alarms"], r["meta"]["ground_truth"],
                                 config.MAX_DELAY).duplicate_alarms)
        rows.append({**_keys(r), **{k: m[k] for k in
                     ["tp", "fp", "fn", "precision", "recall", "ndt",
                      "tf2", "mean_delay", "far"]}, "n_dup": n_dup})
    return _ordered(pd.DataFrame(rows), ["mode", "method", "generator",
                                         "seed"])


def delay_table(runs) -> pd.DataFrame:
    """One row per detected real drift (its TP alarm, delay and NDT)."""
    rows = []
    for r in runs:
        truth = _truth(r)
        matched = match_alarms(r["alarms"], truth, config.MAX_DELAY,
                               config.DUPLICATES_AS_FP).matched
        for k, (d, a) in enumerate(zip(truth, matched)):
            if a is None:
                continue
            window = d["end"] - d["start"] + config.MAX_DELAY
            rows.append({**_keys(r), "drift": k, "type": d["type"],
                         "start": d["start"], "end": d["end"],
                         "t_detect": a, "delay": a - d["start"],
                         "ndt": (a - d["start"]) / window})
    cols = KEYS + ["drift", "type", "start", "end", "t_detect", "delay",
                   "ndt"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return _ordered(pd.DataFrame(rows, columns=cols),
                    ["mode", "method", "generator", "seed", "drift"])


def method_summary(run_df: pd.DataFrame, delay_df: pd.DataFrame
                   ) -> pd.DataFrame:
    """One row per method x mode: summed TP/FP/FN, mean and std of
    precision / recall / TF2 / NDT / far over the runs, median delay over
    the detected drifts (see the module docstring for why)."""
    rows = []
    for mode in config.MODES:
        for method in roster(run_df["method"]):
            x = run_df[(run_df["mode"] == mode) & (run_df["method"] == method)]
            if x.empty:
                continue
            d = delay_df[(delay_df["mode"] == mode)
                         & (delay_df["method"] == method)]
            row = {"mode": mode, "method": method, "n_runs": len(x),
                   "tp": int(x["tp"].sum()), "fp": int(x["fp"].sum()),
                   "fn": int(x["fn"].sum()), "n_dup": int(x["n_dup"].sum())}
            for k in METRICS + ["ndt", "far"]:
                row[f"{k}_mean"] = x[k].mean()
                row[f"{k}_std"] = x[k].std(ddof=1) if len(x) > 1 else 0.0
            row["n_detected"] = len(d)
            row["delay_median"] = d["delay"].median() if len(d) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def delay_curve_table(runs, grid=None) -> pd.DataFrame:
    """Mean precision / recall / TF2 over runs per method x mode x max
    delay (every run rescored with each max delay of ``grid``)."""
    grid = config.STATS["delay_grid"] if grid is None else grid
    rows = []
    for r in runs:
        for p in delay_curve(r["alarms"], r["meta"]["ground_truth"], grid,
                             config.DUPLICATES_AS_FP):
            rows.append({**_keys(r), **p})
    df = pd.DataFrame(rows)
    out = (df.groupby(["mode", "method", "max_delay"])
           .agg(n_runs=("recall", "size"),
                **{k: (k, "mean") for k in METRICS})
           .reset_index())
    return _ordered(out, ["mode", "method", "max_delay"])


# ---------------------------------------------------------------------------
# More tables: significance, drift types, detectors, sensitivity, runtime
# ---------------------------------------------------------------------------
MIXED_MODES = ("MIXED",)  # abrupt and gradual drifts together


def significance(run_df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank on TF2 over the (generator, seed) pairs, per
    mode: each ensemble vs each single member, plus each cluster ensemble
    vs every other ensemble (each pair once)."""
    from scipy.stats import wilcoxon

    groups = _groups(run_df["method"])
    singles, ensembles = groups["singles"], groups["ensembles"]
    comparisons = [(e, s) for e in ensembles for s in singles]
    for a in [m for m in CLUSTER_METHODS if m in ensembles]:
        for b in ensembles:
            if b != a and (b, a) not in comparisons:
                comparisons.append((a, b))
    rows = []
    for mode in config.MODES:
        sub = run_df[run_df["mode"] == mode]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=["generator", "seed"],
                                columns="method", values="tf2")
        for a, b in comparisons:
            if a not in pivot or b not in pivot:
                continue
            paired = pivot[[a, b]].dropna()
            x, y = paired[a], paired[b]
            try:
                stat, p = wilcoxon(x, y)
            except ValueError:  # all differences are zero
                stat, p = math.nan, math.nan
            rows.append({"mode": mode, "comparison": f"{a}_vs_{b}",
                         "n_pairs": len(paired),
                         "mean_tf2_a": float(x.mean()),
                         "mean_tf2_b": float(y.mean()),
                         "mean_diff": float((x - y).mean()),
                         "wilcoxon_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def per_type_mixed(runs) -> pd.DataFrame:
    """Recall and mean delay per drift type on the streams that mix abrupt
    and gradual drifts (MIXED)."""
    rows = []
    for r in runs:
        if r["meta"]["mode"] not in MIXED_MODES:
            continue
        pt = per_type(r["alarms"], r["meta"]["ground_truth"],
                      config.MAX_DELAY, config.DUPLICATES_AS_FP)
        for dtype, vals in pt.items():
            rows.append({**_keys(r), "drift_type": dtype, **vals})
    if not rows:
        return pd.DataFrame()
    out = (pd.DataFrame(rows)
           .groupby(["mode", "method", "drift_type"], observed=True)
           .agg(n_runs=("recall", "size"), recall=("recall", "mean"),
                mean_delay=("mean_delay", "mean")).reset_index())
    return _ordered(out, ["mode", "drift_type", "method"])


def _single_alarms(runs) -> dict[str, dict]:
    """sid -> {"meta": ..., "alarms": {member: alarms}} from the single
    member runs."""
    by_sid: dict[str, dict] = {}
    for r in runs:
        if r["meta"]["method"].startswith("single_"):
            entry = by_sid.setdefault(r["meta"]["sid"],
                                      {"meta": r["meta"], "alarms": {}})
            entry["alarms"][_label(r["meta"]["method"])] = r["alarms"]
    return by_sid


def first_to_fire(runs) -> pd.DataFrame:
    """For each true drift: which single member fired first inside its
    window (ties: roster order)."""
    members = [_label(m) for m in
               _groups(r["meta"]["method"] for r in runs)["singles"]]
    counts: dict[tuple, int] = {}
    for entry in _single_alarms(runs).values():
        meta = entry["meta"]
        gt = sorted(meta["ground_truth"], key=lambda d: d["start"])
        firsts = {name: match_alarms(alarms, gt, config.MAX_DELAY).matched
                  for name, alarms in entry["alarms"].items()}
        for k, drift in enumerate(gt):
            best_name, best_t = "none", None
            for name in members:
                t = firsts.get(name, [None] * len(gt))[k]
                if t is not None and (best_t is None or t < best_t):
                    best_name, best_t = name, t
            key = (meta["mode"], drift["type"], best_name)
            counts[key] = counts.get(key, 0) + 1
    df = pd.DataFrame([{"mode": mode, "drift_type": dtype, "member": name,
                        "cluster": (_member_parts(name)[1]
                                    if name != "none" else "—"),
                        "n_first": n}
                       for (mode, dtype, name), n in sorted(counts.items())])
    if df.empty:
        return df
    totals = df.groupby(["mode", "drift_type"])["n_first"].transform("sum")
    df["share"] = df["n_first"] / totals
    return df


def diversity(runs) -> pd.DataFrame:
    """Pairwise Q-statistic and disagreement between the members over the
    detected/missed vector of all true drifts (single runs)."""
    members = [_label(m) for m in
               _groups(r["meta"]["method"] for r in runs)["singles"]]
    by_mode: dict[str, dict[str, list[int]]] = {}
    for entry in _single_alarms(runs).values():
        meta = entry["meta"]
        gt = sorted(meta["ground_truth"], key=lambda d: d["start"])
        for name, alarms in entry["alarms"].items():
            matched = match_alarms(alarms, gt, config.MAX_DELAY).matched
            by_mode.setdefault(meta["mode"], {}).setdefault(name, []).extend(
                int(m is not None) for m in matched)
    rows = []
    for mode, dets in sorted(by_mode.items()):
        for a, b in itertools.combinations(members, 2):
            if a not in dets or b not in dets:
                continue
            va, vb = np.array(dets[a]), np.array(dets[b])
            n11 = int(((va == 1) & (vb == 1)).sum())
            n00 = int(((va == 0) & (vb == 0)).sum())
            n10 = int(((va == 1) & (vb == 0)).sum())
            n01 = int(((va == 0) & (vb == 1)).sum())
            denom = n11 * n00 + n01 * n10
            rows.append({
                "mode": mode, "member_a": a, "member_b": b,
                "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "q_statistic": ((n11 * n00 - n01 * n10) / denom
                                if denom else math.nan),
                "disagreement": ((n10 + n01) / len(va) if len(va)
                                 else math.nan)})
    return pd.DataFrame(rows)


def sensitivity_summary(runs) -> pd.DataFrame:
    """ens_cluster_sens: final sensitivity level per member, averaged per
    mode, and how often each member was nudged for confirming late."""
    rows = []
    for r in runs:
        meta = r["meta"]
        if meta["method"] != "ens_cluster_sens":
            continue
        path = r["dir"] / "sensitivity.csv"
        log = pd.read_csv(path) if path.exists() else pd.DataFrame()
        final = meta.get("final_levels") or {}
        for member in final:
            n_late = int((log["detector"] == member).sum()) if len(log) else 0
            rows.append({"method": meta["method"], "mode": meta["mode"],
                         "member": member,
                         "final_level": final[member],
                         "n_late": n_late})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).groupby(["method", "mode", "member"])
            .agg(n_runs=("final_level", "size"),
                 mean_final_level=("final_level", "mean"),
                 min_final_level=("final_level", "min"),
                 max_final_level=("final_level", "max"),
                 mean_n_late=("n_late", "mean"))
            .reset_index())


def runtime(runs) -> pd.DataFrame:
    """Mean and total run time per method."""
    df = pd.DataFrame([{"method": r["meta"]["method"],
                        "runtime_s": r["meta"]["runtime_s"]} for r in runs])
    out = (df.groupby("method")
           .agg(n_runs=("runtime_s", "size"),
                mean_runtime_s=("runtime_s", "mean"),
                total_runtime_s=("runtime_s", "sum")).reset_index())
    out["method"] = pd.Categorical(out["method"],
                                   categories=roster(out["method"]),
                                   ordered=True)
    out = out.sort_values("method").reset_index(drop=True)
    out["method"] = out["method"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_delay_curves(curve: pd.DataFrame, out_dir) -> list[str]:
    """<MODE>/delay_curve.png: recall against the allowed max delay, one
    line per method. The window ends max delay after the drift END, so a
    gradual drift can be detected at max delay 0 (inside its transition)."""
    files = []
    methods = roster(curve["method"])  # one colour per method, every mode
    for mode in config.MODES:
        sub = curve[curve["mode"] == mode]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5.4))
        for color, method in zip(COLORS, methods):
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            ax.plot(m["max_delay"], m["recall"], color=color, lw=1.8,
                    ls="--" if method.startswith("single_") else "-",
                    marker="o", ms=3, label=_label(method))
        ax.set_xlabel("max delay after the drift end (instances)",
                      fontsize=9, color=MUTED)
        ax.set_ylabel("recall (mean over runs)", fontsize=9, color=MUTED)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(0, int(sub["max_delay"].max()) + 1, 100))
        _style(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9,
                  frameon=False)
        n = int(sub["n_runs"].max())
        fig.suptitle(f"{mode}: recall against the allowed delay "
                     f"(mean over {n} runs per method)",
                     fontsize=12, color=INK)
        fig.tight_layout()
        out = out_dir / mode / "delay_curve.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        files.append(f"{mode}/{out.name}")
    return files


def _label(name: str) -> str:
    return name.removeprefix("single_")


def _bar_chart(rows: pd.DataFrame, title: str, delay_top: float,
               out) -> None:
    """Per method: precision, recall and TF2 bars on a 0-1 axis (top row)
    and the delay bar in instances (bottom row), the columns aligned.
    Delay gets its own row because it is in a different unit."""
    x = np.arange(len(rows))
    w = 0.22
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    for i, metric in enumerate(METRICS):
        v = rows[f"{metric}_mean"].to_numpy(dtype=float)
        s = np.nan_to_num(rows[f"{metric}_std"].to_numpy(dtype=float))
        # error bars clipped to [0, 1], the range of the metric
        lo, hi = np.minimum(s, v), np.minimum(s, 1.0 - v)
        pos = x + (i - 1) * w
        top.bar(pos, v, w, color=METRIC_COLORS[metric],
                edgecolor="white", lw=1.5, label=METRIC_NAMES[metric])
        top.errorbar(pos, v, yerr=[lo, hi], fmt="none", ecolor=MUTED,
                     elinewidth=1, capsize=3)
        for p, val, h in zip(pos, v, hi):
            if not np.isnan(val):
                top.text(p, val + h + 0.015, f"{val:.2f}", ha="center",
                         va="bottom", fontsize=7.5, color=MUTED)
    top.set_ylim(0, 1.12)
    top.set_yticks(np.linspace(0, 1, 6))
    top.set_ylabel("mean over runs (± std)", fontsize=9, color=MUTED)
    top.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9,
               frameon=False)
    _style(top)

    d = rows["delay_median"].to_numpy(dtype=float)
    bot.bar(x, np.nan_to_num(d), w, color=METRIC_COLORS["delay"],
            edgecolor="white", lw=1.5)
    for p, val in zip(x, d):
        txt = "no detections" if np.isnan(val) else f"{val:.0f}"
        bot.text(p, (0 if np.isnan(val) else val) + delay_top * 0.03, txt,
                 ha="center", va="bottom", fontsize=7.5, color=MUTED)
    bot.set_ylim(0, delay_top)
    bot.set_ylabel("median delay\n(instances after\nthe drift start)",
                   fontsize=9, color=MUTED)
    _style(bot)
    for ax in (top, bot):
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", length=0)
    bot.set_xticks(x)
    bot.set_xticklabels([_label(m) for m in rows["method"]], fontsize=10,
                        color=INK)

    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


HEATMAP_PARAMS = {
    "STEPD": ("window_size", "alpha_drift"),
    "SEED": ("delta", "epsilon_prime"),
    "HDDMWeighted": ("drift_confidence", "lambda_"),
    "RDDM": ("min_n_instances", "drift_level"),
}


def plot_tuning_heatmaps(out_dir, tuning_csv=None) -> list[str]:
    """<detector>_<mode>.png: mean tuning TF2 over two of the
    detector's params (averaged over the others and the streams)."""
    from plotnine import (aes, geom_text, geom_tile, ggplot, labs,
                          scale_fill_gradient, theme_bw)

    path = config.TUNING_DIR / "tuning_runs.csv" if tuning_csv is None \
        else tuning_csv
    if not path.exists():
        return []
    df = pd.read_csv(path).drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for detector, (p1, p2) in HEATMAP_PARAMS.items():
        sub = df[df["detector"] == detector]
        if sub.empty:
            continue
        params = sub["params"].map(json.loads)
        sub = sub.assign(_p1=[p[p1] for p in params],
                         _p2=[p[p2] for p in params])
        for mode in sorted(sub["mode"].unique()):
            m = (sub[sub["mode"] == mode]
                 .groupby(["_p1", "_p2"])["tf2"].mean().reset_index())
            m["label"] = m["tf2"].map(lambda v: f"{v:.2f}")
            for col in ("_p1", "_p2"):
                m[col] = pd.Categorical(
                    m[col].astype(str),
                    categories=[str(v) for v in sorted(m[col].unique())])
            p = (ggplot(m, aes("_p1", "_p2", fill="tf2"))
                 + geom_tile() + geom_text(aes(label="label"), size=7)
                 + scale_fill_gradient(low="#f7fbff", high="#08519c",
                                       limits=(0, 1))
                 + labs(title=f"{detector} mean TF2 — {mode}", x=p1, y=p2)
                 + theme_bw())
            out = out_dir / f"{detector}_{mode}.png"
            p.save(out, width=7, height=5, dpi=150, verbose=False)
            files.append(out.name)
    return files


def plot_bars(summary: pd.DataFrame, out_dir) -> list[str]:
    """<MODE>/singles.png and <MODE>/ensembles.png from metrics.csv. Both
    charts of a mode share the delay scale."""
    files = []
    for mode in config.MODES:
        sub = summary[summary["mode"] == mode]
        if sub.empty:
            continue
        top = sub["delay_median"].max()
        top = 100.0 if np.isnan(top) else max(100.0, np.ceil(top / 100) * 100)
        delay_top = top * 1.2
        (out_dir / mode).mkdir(parents=True, exist_ok=True)
        for group, methods in _groups(summary["method"]).items():
            rows = sub[sub["method"].isin(methods)]
            if rows.empty:
                continue
            n = int(rows["n_runs"].max())
            kind = ("single members" if group == "singles"
                    else "ensembles")
            out = out_dir / mode / f"{group}.png"
            _bar_chart(rows, f"{mode}: {kind} ({n} runs per method, "
                             f"max delay {config.MAX_DELAY})",
                       delay_top, out)
            files.append(f"{mode}/{out.name}")
    return files


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_statistics(test_dir=None, out_dir=None) -> pd.DataFrame:
    """Write every table and plot listed in the module docstring; returns
    the metrics table (empty if there are no finished runs)."""
    out_dir = config.STATISTICS_DIR if out_dir is None else out_dir
    selected = (load_selected_params()
                if (config.TUNING_DIR / "selected_params.json").exists()
                else None)
    runs = load_runs(test_dir, selected)
    if not runs:
        print("No finished test runs found.")
        return pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df = run_table(runs)
    delay_df = delay_table(runs)
    summary = method_summary(run_df, delay_df)
    curve = delay_curve_table(runs)
    for name, df in [("runs", run_df), ("delays", delay_df),
                     ("metrics", summary), ("delay_curve", curve),
                     ("significance", significance(run_df)),
                     ("per_type_mixed", per_type_mixed(runs)),
                     ("first_to_fire", first_to_fire(runs)),
                     ("diversity", diversity(runs)),
                     ("sensitivity_summary", sensitivity_summary(runs)),
                     ("runtime", runtime(runs))]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
    plot_delay_curves(curve, out_dir)
    plot_bars(summary, out_dir)
    plot_tuning_heatmaps(out_dir / "tuning_heatmaps")
    return summary
