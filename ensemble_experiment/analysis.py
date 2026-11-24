"""Statistics of the extended-DDD experiment for one method (config.STATS,
the method with the best DDD accuracy).

Reads ONLY the run folders under results/results_ensemble/runs/ (never
re-runs streams) and writes to statistics/ensemble_experiment/:

  recovery.csv              one row per mode (ABRUPT, GRADUAL, MIXED, RECURRING)
  drifts.csv                one row per run x real drift of the method
  accuracy_200_<MODE>.png   median moving accuracy over the method's runs

Recovery uses only real drifts and the alarm that detected each one (its
TP episode); false alarms never count. Per detected drift:
  acc_before           accuracy over the stable period before the drift:
                       from the previous drift's end (the first drift: from
                       the end of the warm-up) up to the drift start
  acc_drift_to_detect  accuracy from the drift start up to its detection
  rec_instances        instances after the detection until the accuracy
                       counted from the detection (all instances since it)
                       first reaches RECOVERY["level"] x acc_before; checked
                       only once RECOVERY["min_instances"] instances have
                       passed. NaN = not before the next drift.

Plots: one dot per block of STATS["acc_block"] instances = the median over
the method's runs of that block's accuracy. When every run of a mode has its
drifts at the same positions (ABRUPT, GRADUAL) the x-axis is the whole
stream; otherwise (MIXED, RECURRING: the order of abrupt and gradual drifts
differs per seed) the blocks are aligned on each drift start, from
STATS["pre"] before to STATS["post"] after, with the median over drifts and
one line for abrupt and one for gradual drifts.

Real datasets (no drift positions, real_statistics): accuracy.csv and
accuracy_over_time.png per dataset under
statistics/ensemble_experiment/real/<dataset>/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensemble_experiment import config


KEYS = ["generator", "mode", "seed", "method"]


def load_runs(runs_dir=None, method: str | None = None
              ) -> tuple[pd.DataFrame, list]:
    """(episodes, metas) of every complete run of ``method`` (default: every
    method); metas = [(run folder, meta dict)]."""
    runs_dir = config.RUNS_DIR if runs_dir is None else runs_dir
    eps, metas = [], []
    if not runs_dir.exists():
        return pd.DataFrame(), []
    for d in sorted(runs_dir.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if method is not None and meta["method"] != method:
            continue
        ep = pd.read_csv(d / "episodes.csv")
        if len(ep):
            for k in KEYS:
                ep.insert(0, k, meta[k])
            eps.append(ep)
        metas.append((d, meta))
    episodes = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    return episodes, metas


# ---------------------------------------------------------------------------
# Recovery after real drifts
# ---------------------------------------------------------------------------
def drift_bounds(ground_truth, length: int) -> list[tuple[int, int, int]]:
    """(start, end, next drift start) of every real drift, in time order;
    the last drift's "next start" is the stream length."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    nxt = [d["start"] for d in truth[1:]] + [length]
    return [(d["start"], d["end"], n) for d, n in zip(truth, nxt)]


def recovery_instances(correct, t_detect: int, stop: int, target: float,
                       min_instances: int) -> int | None:
    """Instances after ``t_detect`` until the accuracy counted from the
    detection (instances t_detect+1 .. t, all of them) first reaches
    ``target``, checked only once ``min_instances`` instances have passed;
    None if that does not happen before ``stop`` (the next drift)."""
    c = np.asarray(correct[t_detect + 1:stop], dtype=float)
    n = np.arange(1, len(c) + 1)
    acc = np.cumsum(c) / n
    hit = np.flatnonzero((acc >= target - 1e-12) & (n >= min_instances))
    return int(n[hit[0]]) if len(hit) else None


def drift_table(metas, episodes: pd.DataFrame, rec: dict | None = None,
                warmup: int | None = None) -> pd.DataFrame:
    """One row per (run, real drift) of the given runs.

    A drift counts as detected when the run has a TP episode matched to it
    (logger.label_alarms); its t_detect is that alarm."""
    rec = config.RECOVERY if rec is None else rec
    warmup = config.MIN_TRAINING_SIZE if warmup is None else warmup
    tp = {}
    if not episodes.empty:
        e = episodes[episodes["label"] == "TP"]
        for r in e[KEYS + ["drift_idx", "t_detect"]].itertuples(index=False):
            tp[(r.generator, r.mode, r.seed, r.method,
                int(r.drift_idx))] = int(r.t_detect)
    rows = []
    for d, m in metas:
        c = np.load(d / "correct.npy").astype(float)
        key = tuple(m[k] for k in KEYS)
        truth = sorted(m["ground_truth"], key=lambda g: g["start"])
        prev_end = warmup
        for k, (g, (s, e, nxt)) in enumerate(
                zip(truth, drift_bounds(truth, len(c)))):
            td = tp.get((*key, k))
            before = float(c[prev_end:s].mean())
            n = (None if td is None else recovery_instances(
                c, td, nxt, rec["level"] * before, rec["min_instances"]))
            rows.append({
                **dict(zip(KEYS, key)), "drift": k, "type": g["type"],
                "recurring": g.get("recurring"), "start": s, "end": e,
                "next_start": nxt, "detected": td is not None,
                "t_detect": td, "delay": None if td is None else td - s,
                "acc_before": before,
                "acc_drift_to_detect": (np.nan if td is None
                                        else float(c[s:td + 1].mean())),
                "rec_instances": np.nan if n is None else n,
                "recovered": None if td is None else n is not None,
            })
            prev_end = e
    return pd.DataFrame(rows)


def mode_summary(drifts: pd.DataFrame) -> pd.DataFrame:
    """One row per mode (config.MODES order) of one method's drift rows:
    medians over its detected real drifts."""
    rows = []
    for mode in config.MODES:
        x = drifts[drifts["mode"] == mode]
        if x.empty:
            continue
        det = x[x["detected"]]
        rec = det["rec_instances"]
        rows.append({
            "mode": mode,
            "drifts": len(x),
            "detected": len(det),
            "acc_before_median": det["acc_before"].median(),
            "acc_drift_to_detect_median": det["acc_drift_to_detect"].median(),
            "rec_instances_median": rec.median(),
            "recovered_pct": (100 * rec.notna().mean() if len(det)
                              else np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Median accuracy per block of instances (one plot point per block)
# ---------------------------------------------------------------------------
def block_acc(correct, block: int) -> np.ndarray:
    """Accuracy of each consecutive, non-overlapping ``block`` of instances
    (a last partial block is dropped)."""
    c = np.asarray(correct, dtype=float)
    n = len(c) // block
    return c[:n * block].reshape(n, block).mean(axis=1)


def _nanmedian(stack: np.ndarray) -> np.ndarray:
    """Column median ignoring NaN, without all-NaN warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def median_blocks_stream(arrays, block: int) -> np.ndarray:
    """Median over runs of every block's accuracy (runs are cut to the
    shortest one). Block k covers instances [k*block, (k+1)*block)."""
    n = min(len(a) for a in arrays)
    return _nanmedian(np.stack([block_acc(a[:n], block) for a in arrays]))


def median_blocks_aligned(runs, block: int, pre: int, post: int
                          ) -> tuple[np.ndarray, int]:
    """Median over drifts of the block accuracies from ``pre`` before to
    ``post`` after each drift start (blocks start at the drift start, so
    block k covers [start - pre + k*block, ... + block)).
    ``runs`` = [(correct, [drift starts])]. Returns (curve of
    (pre + post) // block values, number of drifts used)."""
    segs = []
    for correct, starts in runs:
        for s in starts:
            if s - pre >= 0 and s + post <= len(correct):
                segs.append(block_acc(correct[s - pre:s + post], block))
    if not segs:
        return np.full((pre + post) // block, np.nan), 0
    return _nanmedian(np.stack(segs)), len(segs)


def _same_positions(metas) -> bool:
    pos = {tuple((d["start"], d["end"]) for d in m["ground_truth"])
           for _, m in metas}
    return len(pos) == 1


_DOTS = dict(marker="o", ms=3, lw=1.0)


def plot_accuracy_200(metas, arrays, mode: str, method: str, out_dir,
                      stats: dict | None = None) -> None:
    """accuracy_<block>_<mode>.png for ``method`` (``metas`` = its runs of
    ``mode``): one dot per block of STATS["acc_block"] instances."""
    stats = config.STATS if stats is None else stats
    b = stats["acc_block"]
    if not metas:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    if _same_positions(metas):
        curve = median_blocks_stream([arrays[d.name] for d, _ in metas], b)
        for g in metas[0][1]["ground_truth"]:
            if g["end"] > g["start"]:
                ax.axvspan(g["start"], g["end"], color="0.85", lw=0)
            else:
                ax.axvline(g["start"], color="0.5", lw=0.8, ls="--")
        ax.plot(np.arange(len(curve)) * b + b / 2, curve, **_DOTS)
        ax.set(xlabel="instance (dashed = abrupt drift, grey = gradual "
                      "transition)",
               title=f"{method}, {mode}: median accuracy per {b} instances "
                     f"over {len(metas)} runs")
    else:
        # Abrupt and gradual drifts recover differently: one line each.
        pre, post = stats["pre"], stats["post"]
        x = np.arange(-pre, post, b) + b / 2
        widths = {g["end"] - g["start"] for _, m in metas
                  for g in m["ground_truth"]} - {0}
        if widths:
            ax.axvspan(0, max(widths), color="0.9", lw=0,
                       label="gradual transition")
        ax.axvline(0, color="0.5", lw=0.8, ls="--", label="drift start")
        for typ in ["ABRUPT", "GRADUAL"]:
            curve, n = median_blocks_aligned(
                [(arrays[d.name], [g["start"] for g in m["ground_truth"]
                                   if g["type"] == typ])
                 for d, m in metas], b, pre, post)
            if n:
                ax.plot(x, curve, label=f"{typ.lower()} drifts (n={n})",
                        **_DOTS)
        ax.legend(fontsize=8, loc="lower right")
        ax.set(xlabel="instances since drift start",
               title=f"{method}, {mode}: median accuracy per {b} instances, "
                     f"aligned on each drift ({len(metas)} runs)")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_{b}_{mode}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def run_analysis(runs_dir=None, out_dir=None, method: str | None = None
                 ) -> pd.DataFrame:
    """Write recovery.csv, drifts.csv and the plots for ``method``
    (default config.STATS["method"]); returns the recovery table."""
    out_dir = config.ANALYSIS_DIR if out_dir is None else out_dir
    method = config.STATS["method"] if method is None else method
    episodes, metas = load_runs(runs_dir, method)
    if not metas:
        print(f"No finished runs of {method} found.")
        return pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)

    drifts = drift_table(metas, episodes)
    drifts.to_csv(out_dir / "drifts.csv", index=False)
    summary = mode_summary(drifts)
    summary.to_csv(out_dir / "recovery.csv", index=False)

    arrays = {d.name: np.load(d / "correct.npy") for d, _ in metas}
    for mode in config.MODES:
        sel = [(d, m) for d, m in metas if m["mode"] == mode]
        plot_accuracy_200(sel, arrays, mode, method, out_dir)
    return summary


# ===========================================================================
# Real datasets (reads run folders only)
# ===========================================================================
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load_real_runs(root=None) -> list[dict]:
    root = config.REAL_RUNS_DIR if root is None else Path(root)
    runs = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        if (d / "meta.json").exists():
            alarms = (json.loads((d / "alarms.json").read_text())["alarms"]
                      if (d / "alarms.json").exists() else [])
            runs.append({"meta": json.loads((d / "meta.json").read_text()),
                         "correct": np.load(d / "correct.npy"),
                         "alarms": alarms})
    return runs


def accuracy_table(runs):
    """One row per dataset x method: prequential accuracy after the
    warm-up (acc) and over every instance (acc_all), and the number of
    detections."""
    df = pd.DataFrame([{"dataset": r["meta"]["dataset"],
                        "method": r["meta"]["method"],
                        "seed": r["meta"]["seed"],
                        "acc": r["meta"]["summary"]["acc"],
                        "acc_all": r["meta"]["summary"]["acc_all"],
                        "n_detections": r["meta"]["summary"]["n_detections"]}
                       for r in runs])
    df["method"] = pd.Categorical(df["method"], config.REAL_RUN_METHODS,
                                  ordered=True)
    return df.sort_values(["dataset", "method"]).reset_index(drop=True)


def plot_accuracy(runs, dataset: str, out, block: int) -> None:
    """Accuracy per block of ``block`` instances, one line per method; the
    no-change baseline grey and dotted, DDD's detections as ticks."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, method in enumerate(config.REAL_RUN_METHODS):
        rs = [r for r in runs if r["meta"]["dataset"] == dataset
              and r["meta"]["method"] == method]
        if not rs:
            continue
        r = rs[0]
        c = r["correct"]
        n = len(c) // block * block
        acc = c[:n].reshape(-1, block).mean(axis=1)
        x = (np.arange(len(acc)) + 0.5) * block
        if method in config.REAL_BASELINES:
            style = dict(color="#898781", lw=1.6, ls=":")
        else:
            style = dict(color=COLORS[i], lw=1.8, ls="-")
        label = f"{method} (acc {r['meta']['summary']['acc']:.3f})"
        ax.plot(x, acc, label=label, **style)
        if r["alarms"]:
            ax.plot(r["alarms"], np.full(len(r["alarms"]), 0.0), "|",
                    color=style["color"], ms=12, mew=1.2,
                    transform=ax.get_xaxis_transform(),
                    label=f"{method} detections ({len(r['alarms'])})")
    ax.axvline(config.MIN_TRAINING_SIZE, color=MUTED, lw=0.8)
    ax.set_xlabel("instance", fontsize=9, color=MUTED)
    ax.set_ylabel(f"accuracy per {block} instances", fontsize=9, color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9,
              frameon=False)
    fig.suptitle(f"{dataset}: prequential accuracy over time",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def real_statistics(root=None, out_dir=None):
    """accuracy.csv and accuracy_over_time.png per dataset."""
    out_dir = config.REAL_STATS_DIR if out_dir is None else Path(out_dir)
    runs = load_real_runs(root)
    if not runs:
        print("No finished real-data runs found.")
        return None
    table = accuracy_table(runs)
    for dataset in table["dataset"].unique():
        d = out_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        table[table["dataset"] == dataset].to_csv(d / "accuracy.csv",
                                                  index=False)
        plot_accuracy(runs, dataset, d / "accuracy_over_time.png",
                      config.REAL_DATASETS[dataset]["acc_block"])
    return table
