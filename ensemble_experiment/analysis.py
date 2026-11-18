"""Tables and plots of the extended-DDD experiment.

Reads ONLY the run folders under results/results_ensemble/runs/ (never
re-runs streams) and writes to results/results_ensemble/analysis/:
  runs.csv                          one row per run (meta.json summaries)
  summary_by_mode_method.csv        means over generators x seeds
  summary_by_generator_mode_method.csv
  episodes_all.csv                  every episode of every run
  acc_curves_<MODE>.png             windowed accuracy, one stream per mode
  caps_<MODE>.png                   cap_fast / cap_slow over that stream
  exit_types.png                    exit-type shares per method and mode
  acc_vs_f1.png                     DDD accuracy vs detection F1
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ensemble_experiment import config  # noqa: E402
from ensemble_experiment.logger import EXIT_TYPES  # noqa: E402

KEYS = ["generator", "mode", "seed", "method"]
MEAN_COLS = ["acc", "f1", "precision", "recall", "mean_delay", "far",
             "n_detections", "frac_after", "frac_5", "mean_steps_after",
             "kind_acc", "recall_hit_rate", "recall_hit_rate_older"]
SUM_COLS = (["tp", "fp", "fn", "n_recalls", "n_recalls_older",
             "n_kind_graded"] + [f"exit_{e}" for e in EXIT_TYPES])
ACC_WINDOW = 500  # moving-average window for accuracy curves


def load_runs(runs_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(runs, episodes, metas) of every complete run of a roster method."""
    runs_dir = config.RUNS_DIR if runs_dir is None else runs_dir
    rows, eps, metas = [], [], []
    if not runs_dir.exists():
        return pd.DataFrame(), pd.DataFrame(), []
    for d in sorted(runs_dir.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["method"] not in config.METHODS:
            continue  # stale method
        key = {k: meta[k] for k in KEYS}
        rows.append({**key, **meta["summary"],
                     "runtime_s": meta["runtime_s"]})
        ep = pd.read_csv(d / "episodes.csv")
        if len(ep):
            for k, v in key.items():
                ep.insert(0, k, v)
            eps.append(ep)
        metas.append((d, meta))
    runs = pd.DataFrame(rows)
    episodes = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    return runs, episodes, metas


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mode"] = pd.Categorical(df["mode"], config.MODES, ordered=True)
    df["method"] = pd.Categorical(df["method"], config.METHODS, ordered=True)
    return df


def summarize(runs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    runs = _order(runs)
    g = runs.groupby(by, observed=True)
    out = g[MEAN_COLS].mean()
    out["acc_std"] = g["acc"].std()
    out["f1_std"] = g["f1"].std()
    out = out.join(g[SUM_COLS].sum())
    out["n_runs"] = g.size()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _representative(metas, mode):
    """(generator, seed) of the first stream of ``mode`` with the most
    finished methods."""
    counts: dict[tuple, int] = {}
    for _, m in metas:
        if m["mode"] == mode:
            k = (m["generator"], m["seed"])
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return min(k for k, c in counts.items() if c == best)


def _drift_marks(ax, ground_truth):
    for d in ground_truth:
        if d["end"] > d["start"]:
            ax.axvspan(d["start"], d["end"], color="0.85", lw=0)
        else:
            ax.axvline(d["start"], color="0.6", lw=0.8, ls="--")


def plot_acc_curves(metas, out_dir) -> None:
    for mode in config.MODES:
        rep = _representative(metas, mode)
        if rep is None:
            continue
        gen, seed = rep
        sel = [(d, m) for d, m in metas
               if (m["generator"], m["mode"], m["seed"]) == (gen, mode, seed)]
        fig, ax = plt.subplots(figsize=(11, 4))
        _drift_marks(ax, sel[0][1]["ground_truth"])
        kernel = np.ones(ACC_WINDOW) / ACC_WINDOW
        for d, m in sorted(sel, key=lambda x: config.METHODS.index(
                x[1]["method"])):
            c = np.load(d / "correct.npy").astype(float)
            ma = np.convolve(c, kernel, mode="valid")
            ax.plot(np.arange(ACC_WINDOW - 1, len(c)), ma, lw=0.9,
                    label=f"{m['method']} ({m['summary']['acc']:.3f})")
        ax.set(title=f"DDD accuracy, {gen} {mode} seed {seed} "
                     f"(window {ACC_WINDOW}; grey = drifts)",
               xlabel="instance", ylabel="accuracy")
        ax.legend(fontsize=7, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / f"acc_curves_{mode}.png", dpi=130)
        plt.close(fig)


def plot_caps(metas, out_dir) -> None:
    for mode in config.MODES:
        rep = _representative(metas, mode)
        if rep is None:
            continue
        gen, seed = rep
        sel = [(d, m) for d, m in metas
               if (m["generator"], m["mode"], m["seed"]) == (gen, mode, seed)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
        for d, m in sorted(sel, key=lambda x: config.METHODS.index(
                x[1]["method"])):
            st = pd.read_csv(d / "steps.csv")
            for ax, col in zip(axes, ["cap_fast", "cap_slow"]):
                ax.step(st["t"], st[col], where="post", lw=0.9,
                        label=m["method"])
        for ax, col in zip(axes, ["cap_fast", "cap_slow"]):
            _drift_marks(ax, sel[0][1]["ground_truth"])
            ax.set(title=f"{col}, {gen} {mode} seed {seed}",
                   xlabel="instance")
        axes[0].set_ylabel("cap (instances)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"caps_{mode}.png", dpi=130)
        plt.close(fig)


def plot_exit_types(summary: pd.DataFrame, out_dir) -> None:
    cols = [f"exit_{e}" for e in EXIT_TYPES]
    modes = [m for m in config.MODES if m in set(summary["mode"].astype(str))]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        s = summary[summary["mode"].astype(str) == mode].set_index("method")
        shares = s[cols].div(s[cols].sum(axis=1).replace(0, np.nan), axis=0)
        bottom = np.zeros(len(shares))
        for col in cols:
            ax.bar(range(len(shares)), shares[col].fillna(0), bottom=bottom,
                   label=col[len("exit_"):])
            bottom += shares[col].fillna(0).to_numpy()
        ax.set_xticks(range(len(shares)))
        ax.set_xticklabels([str(i) for i in shares.index], rotation=60,
                           ha="right", fontsize=7)
        ax.set_title(mode)
    axes[0][0].set_ylabel("share of episodes")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "exit_types.png", dpi=130)
    plt.close(fig)


def plot_acc_vs_f1(summary: pd.DataFrame, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mode in config.MODES:
        s = summary[summary["mode"].astype(str) == mode]
        if s.empty:
            continue
        ax.scatter(s["f1"], s["acc"], label=mode, s=25)
        for _, r in s.iterrows():
            ax.annotate(str(r["method"]).replace("single_", ""),
                        (r["f1"], r["acc"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    ax.set(xlabel="detection F1 (mean)", ylabel="DDD accuracy (mean)",
           title="Does better detection give a better DDD?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "acc_vs_f1.png", dpi=130)
    plt.close(fig)


def run_analysis(runs_dir=None, out_dir=None) -> pd.DataFrame:
    out_dir = config.ANALYSIS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, episodes, metas = load_runs(runs_dir)
    if runs.empty:
        print("No finished runs found.")
        return runs
    runs.sort_values(KEYS).to_csv(out_dir / "runs.csv", index=False)
    if not episodes.empty:
        episodes.to_csv(out_dir / "episodes_all.csv", index=False)
    by_mode = summarize(runs, ["mode", "method"])
    by_mode.to_csv(out_dir / "summary_by_mode_method.csv", index=False)
    summarize(runs, ["generator", "mode", "method"]).to_csv(
        out_dir / "summary_by_generator_mode_method.csv", index=False)
    plot_acc_curves(metas, out_dir)
    plot_caps(metas, out_dir)
    plot_exit_types(by_mode, out_dir)
    plot_acc_vs_f1(by_mode, out_dir)
    return by_mode
