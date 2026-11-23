"""Detection statistics of the drift-ensemble experiment for every roster
method (drift_ensemble/scripts/05_analyze.py).

Reads ONLY the run folders under results/results_drift_ensemble/test/
(never reruns streams) and writes to statistics/drift_ensemble/:

  runs.csv                  one row per run (stream x method): TP, FP, FN,
                            precision, recall, F1, mean delay
  delays.csv                one row per detected real drift: its delay
  metrics.csv               one row per method x mode: summed TP/FP/FN,
                            mean and std of precision / recall / F1 over
                            the mode's runs, median delay over its
                            detected drifts
  delay_curve.csv           mean precision / recall / F1 over runs for
                            every max delay in STATS["delay_grid"]
  <MODE>/singles.png        bar chart of metrics.csv: per single detector
                            the mean precision, recall and F1 with std
                            error bars (top row, 0-1) and its median delay
                            (bottom row, in instances, same delay scale
                            for both charts)
  <MODE>/ensembles.png      the same for the four ensemble methods
  <MODE>/delay_curve.png    the delay curves, one line per method

Why mean for P/R/F1 but median for delay: the 15 runs of a method x mode
are 3 generators x 5 seeds, so a method that fails on one whole generator
fails on a third of the runs, and a median would hide that. P/R/F1 are
bounded in [0, 1], so a mean is safe. Delays are skewed with a long tail,
so they use the median.

Scoring is evaluation.match_alarms with config.MAX_DELAY: drift k is
detected by the first alarm in [start_k, end_k + max delay], delay = alarm -
start_k, extra alarms are FP when config.DUPLICATES_AS_FP. The warm-up comes
from each run's snapshot.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from drift_ensemble import config  # noqa: E402
from drift_ensemble.analysis import load_runs  # noqa: E402
from drift_ensemble.evaluation import (  # noqa: E402
    compute_metrics,
    delay_curve,
    match_alarms,
)

KEYS = ["generator", "mode", "seed", "method"]
METRICS = ["precision", "recall", "f1"]
# Fixed colour per method (config.METHODS order), so a method keeps its
# colour in every plot; singles are dashed, ensembles solid.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Bar charts: one colour per metric (the delay bar has its own row).
METRIC_COLORS = {"precision": "#2a78d6", "recall": "#eb6834",
                 "f1": "#1baf7a", "delay": "#6250d6"}
METRIC_NAMES = {"precision": "precision", "recall": "recall", "f1": "F1"}
GROUPS = {"singles": [m for m in config.METHODS if m.startswith("single_")],
          "ensembles": [m for m in config.METHODS
                        if not m.startswith("single_")]}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _keys(run: dict) -> dict:
    return {k: run["meta"][k] for k in KEYS}


def _truth(run: dict) -> list[dict]:
    return sorted(run["meta"]["ground_truth"], key=lambda d: d["start"])


def _ordered(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sort by mode and method in config order (then the rest of ``by``)."""
    df = df.copy()
    df["mode"] = pd.Categorical(df["mode"], categories=config.MODES,
                                ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=config.METHODS,
                                  ordered=True)
    df = df.sort_values(by, kind="stable").reset_index(drop=True)
    df["mode"] = df["mode"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def run_table(runs) -> pd.DataFrame:
    """One row per run: TP/FP/FN, precision, recall, F1, mean delay."""
    rows = []
    for r in runs:
        m = compute_metrics(
            r["alarms"], r["meta"]["ground_truth"], config.MAX_DELAY,
            r["meta"]["length"], r["meta"]["config"]["MIN_TRAINING_SIZE"],
            config.DUPLICATES_AS_FP)
        rows.append({**_keys(r), **{k: m[k] for k in
                     ["tp", "fp", "fn", "precision", "recall", "f1",
                      "mean_delay"]}})
    return _ordered(pd.DataFrame(rows), ["mode", "method", "generator",
                                         "seed"])


def delay_table(runs) -> pd.DataFrame:
    """One row per detected real drift (its TP alarm and delay)."""
    rows = []
    for r in runs:
        truth = _truth(r)
        matched = match_alarms(r["alarms"], truth, config.MAX_DELAY,
                               config.DUPLICATES_AS_FP).matched
        for k, (d, a) in enumerate(zip(truth, matched)):
            if a is None:
                continue
            rows.append({**_keys(r), "drift": k, "type": d["type"],
                         "start": d["start"], "end": d["end"],
                         "t_detect": a, "delay": a - d["start"]})
    cols = KEYS + ["drift", "type", "start", "end", "t_detect", "delay"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return _ordered(pd.DataFrame(rows, columns=cols),
                    ["mode", "method", "generator", "seed", "drift"])


def method_summary(run_df: pd.DataFrame, delay_df: pd.DataFrame
                   ) -> pd.DataFrame:
    """One row per method x mode: summed TP/FP/FN, mean and std of
    precision / recall / F1 over the runs, median delay over the detected
    drifts (see the module docstring for why)."""
    rows = []
    for mode in config.MODES:
        for method in config.METHODS:
            x = run_df[(run_df["mode"] == mode) & (run_df["method"] == method)]
            if x.empty:
                continue
            d = delay_df[(delay_df["mode"] == mode)
                         & (delay_df["method"] == method)]
            row = {"mode": mode, "method": method, "n_runs": len(x),
                   "tp": int(x["tp"].sum()), "fp": int(x["fp"].sum()),
                   "fn": int(x["fn"].sum())}
            for k in METRICS:
                row[f"{k}_mean"] = x[k].mean()
                row[f"{k}_std"] = x[k].std(ddof=1) if len(x) > 1 else 0.0
            row["n_detected"] = len(d)
            row["delay_median"] = d["delay"].median() if len(d) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def delay_curve_table(runs, grid=None) -> pd.DataFrame:
    """Mean precision / recall / F1 over runs per method x mode x max
    delay (every run rescored with each max delay of ``grid``)."""
    grid = config.STATS["delay_grid"] if grid is None else grid
    rows = []
    for r in runs:
        dc = delay_curve(r["alarms"], r["meta"]["ground_truth"], grid,
                         config.DUPLICATES_AS_FP)
        for p in dc["points"]:
            rows.append({**_keys(r), **p})
    df = pd.DataFrame(rows)
    out = (df.groupby(["mode", "method", "max_delay"])
           .agg(n_runs=("f1", "size"), **{k: (k, "mean") for k in METRICS})
           .reset_index())
    return _ordered(out, ["mode", "method", "max_delay"])


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
    for mode in config.MODES:
        sub = curve[curve["mode"] == mode]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5.4))
        for color, method in zip(COLORS, config.METHODS):
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
    """Per method: precision, recall and F1 bars on a 0-1 axis (top row)
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
        for group, methods in GROUPS.items():
            rows = sub[sub["method"].isin(methods)]
            if rows.empty:
                continue
            n = int(rows["n_runs"].max())
            kind = ("single detectors" if group == "singles"
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
    runs = load_runs(test_dir)
    if not runs:
        print("No finished test runs found.")
        return pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df = run_table(runs)
    delay_df = delay_table(runs)
    summary = method_summary(run_df, delay_df)
    curve = delay_curve_table(runs)
    for name, df in [("runs", run_df), ("delays", delay_df),
                     ("metrics", summary), ("delay_curve", curve)]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
    plot_delay_curves(curve, out_dir)
    plot_bars(summary, out_dir)
    return summary
