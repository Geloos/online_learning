"""Analysis of finished test runs (drift_ensemble/scripts/05_analyze.py).

Reads ONLY result folders (never reruns experiments) and writes the tables
and plots under results/results_drift_ensemble/analysis/. Scoring parameters
(max delay, delay grid, duplicates-as-FP) are analysis-time choices taken
from the current config — alarms do not depend on them; only warm-up comes
from each run's snapshot, so shrunken smoke runs analyze correctly too.
"""

from __future__ import annotations

import itertools
import json
import math

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)  # headless plotting for plotnine

from drift_ensemble import config
from drift_ensemble.evaluation import (
    compute_metrics,
    delay_curve,
    match_alarms,
    per_type,
)

SINGLE_METHODS = [m for m in config.METHODS if m.startswith("single_")]
ENSEMBLE_METHODS = [m for m in config.METHODS if m.startswith("ens_")]
# Methods that log detector weights, with their starting weight.
WEIGHTED_METHODS = {
    "ens_cluster_vote": lambda cfg: cfg["VOTE"]["W_INIT"],
}
# Methods that log per-detector sensitivity levels (sensitivity.csv).
SENS_METHODS = [m for m in ENSEMBLE_METHODS if m == "ens_cluster_sens"]
# Cluster-pair ensembles, compared against every other ensemble.
CLUSTER_METHODS = [m for m in ENSEMBLE_METHODS
                   if m in ("ens_cluster_vote", "ens_cluster_sens")]


def tables_dir():
    d = config.ANALYSIS_DIR / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir():
    d = config.ANALYSIS_DIR / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs() -> list[dict]:
    runs = []
    if not config.TEST_DIR.exists():
        return runs
    for meta_path in sorted(config.TEST_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["method"] not in config.METHODS:
            continue  # stale folder of a method dropped from the roster
        folder = meta_path.parent
        alarms = json.loads((folder / "alarms.json").read_text())["alarms"]
        runs.append({"dir": folder, "meta": meta, "alarms": alarms})
    return runs


def _csv(run: dict, name: str) -> pd.DataFrame:
    path = run["dir"] / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _eval_cfg(run: dict) -> dict:
    # Matching/scoring parameters (max_delay, delay_grid, duplicates_as_fp)
    # are analysis-time choices: alarms do not depend on them, so runs are
    # re-scored with the CURRENT config. Only min_training_size shaped the
    # run itself and therefore comes from its snapshot.
    c = run["meta"]["config"]
    return {
        "max_delay": config.MAX_DELAY,
        "min_training_size": c["MIN_TRAINING_SIZE"],
        "duplicates_as_fp": config.DUPLICATES_AS_FP,
        "delay_grid": config.DELAY_GRID,
    }


def _method_cat(s: pd.Series) -> pd.Categorical:
    return pd.Categorical(s, categories=config.METHODS, ordered=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def run_index(runs) -> pd.DataFrame:
    rows = [{
        "generator": r["meta"]["generator"],
        "mode": r["meta"]["mode"],
        "seed": r["meta"]["seed"],
        "method": r["meta"]["method"],
        "dir": r["dir"].name,
        "length": r["meta"]["length"],
        "n_alarms": r["meta"]["n_alarms"],
        "runtime_s": r["meta"]["runtime_s"],
    } for r in runs]
    return pd.DataFrame(rows)


def per_run_metrics(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        e = _eval_cfg(r)
        m = compute_metrics(
            r["alarms"], r["meta"]["ground_truth"], e["max_delay"],
            r["meta"]["length"], e["min_training_size"],
            e["duplicates_as_fp"])
        # Extra alarms inside an already-matched drift window (a subset of
        # fp when DUPLICATES_AS_FP): the same drift reported twice.
        n_dup = len(match_alarms(r["alarms"], r["meta"]["ground_truth"],
                                 e["max_delay"]).duplicate_alarms)
        rows.append({
            "generator": r["meta"]["generator"],
            "mode": r["meta"]["mode"],
            "seed": r["meta"]["seed"],
            "method": r["meta"]["method"],
            **m,
            "n_dup": n_dup,
        })
    return pd.DataFrame(rows)


MODE_ORDER = ["ABRUPT", "GRADUAL", "MIXED", "ALL"]


def detection_metrics(metrics: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summed TP/FP/FN plus mean ± std and median of the ratio metrics per
    method x mode over (generator, seed), with an extra ALL row per method
    aggregating every run regardless of mode. Returns the numeric wide
    table and a formatted 'mean ± std' table for LaTeX."""
    cols = ["precision", "recall", "f1", "mean_delay", "far"]
    all_rows = metrics.copy()
    all_rows["mode"] = "ALL"
    m2 = pd.concat([metrics, all_rows], ignore_index=True)
    g = m2.groupby(["method", "mode"], observed=True)
    agg = g[cols].agg(["mean", "std", "median"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = g[["tp", "fp", "fn", "n_dup"]].sum().join(agg)
    agg["n_runs"] = g.size()
    agg = agg.reset_index()
    agg["method"] = _method_cat(agg["method"])
    agg["mode"] = pd.Categorical(agg["mode"], categories=MODE_ORDER,
                                 ordered=True)
    agg = agg.sort_values(["mode", "method"])

    pretty = agg[["method", "mode", "n_runs", "tp", "fp", "fn",
                  "n_dup"]].copy()
    for c in cols:
        pretty[c] = [
            f"{m:.3f} ± {s:.3f}" if not math.isnan(m) else "—"
            for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"].fillna(0.0))
        ]
    return agg, pretty


def detection_medians(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compact median-only view: per method x mode (plus ALL), the median
    over runs of precision, recall, F1, delay and FAR."""
    agg, _ = detection_metrics(metrics)
    out = agg[["method", "mode", "n_runs",
               "precision_median", "recall_median", "f1_median",
               "mean_delay_median", "far_median"]].copy()
    return out.rename(columns={
        "precision_median": "precision",
        "recall_median": "recall",
        "f1_median": "f1",
        "mean_delay_median": "delay",
        "far_median": "far",
    })


def delay_curve_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        e = _eval_cfg(r)
        dc = delay_curve(r["alarms"], r["meta"]["ground_truth"],
                         e["delay_grid"], e["duplicates_as_fp"])
        for p in dc["points"]:
            rows.append({
                "method": r["meta"]["method"],
                "mode": r["meta"]["mode"],
                "generator": r["meta"]["generator"],
                "seed": r["meta"]["seed"],
                **p,
            })
    df = pd.DataFrame(rows)
    out = (df.groupby(["method", "mode", "max_delay"], observed=True)
           [["recall", "f1"]].mean().reset_index())
    out["method"] = _method_cat(out["method"])
    return out.sort_values(["mode", "method", "max_delay"])


def per_type_mixed(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r["meta"]["mode"] != "MIXED":
            continue
        e = _eval_cfg(r)
        pt = per_type(r["alarms"], r["meta"]["ground_truth"],
                      e["max_delay"], e["duplicates_as_fp"])
        for dtype, vals in pt.items():
            rows.append({
                "method": r["meta"]["method"],
                "generator": r["meta"]["generator"],
                "seed": r["meta"]["seed"],
                "drift_type": dtype,
                **vals,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = (df.groupby(["method", "drift_type"], observed=True)
           .agg(n_runs=("recall", "size"), recall=("recall", "mean"),
                mean_delay=("mean_delay", "mean")).reset_index())
    out["method"] = _method_cat(out["method"])
    return out.sort_values(["drift_type", "method"])


def first_to_fire(runs) -> pd.DataFrame:
    """For each true drift: which detector fired first inside its window
    (using the single-detector runs' raw fires)."""
    cluster_of = {name: cl for cl, ms in config.CLUSTER_MEMBERS.items()
                  for name in ms}
    by_sid: dict[str, dict] = {}
    for r in runs:
        if r["meta"]["method"] in SINGLE_METHODS:
            sid = r["meta"]["sid"]
            by_sid.setdefault(sid, {"meta": r["meta"], "alarms": {}})
            name = r["meta"]["method"][len("single_"):]
            by_sid[sid]["alarms"][name] = r["alarms"]

    counts: dict[tuple, int] = {}
    for sid, entry in by_sid.items():
        meta = entry["meta"]
        gt = sorted(meta["ground_truth"], key=lambda d: d["start"])
        max_delay = config.MAX_DELAY
        firsts = {
            name: match_alarms(alarms, gt, max_delay).matched
            for name, alarms in entry["alarms"].items()
        }
        for k, drift in enumerate(gt):
            best_name, best_t = "none", None
            for name in config.DETECTOR_ORDER:
                t = firsts.get(name, [None] * len(gt))[k]
                if t is not None and (best_t is None or t < best_t):
                    best_name, best_t = name, t
            key = (meta["mode"], drift["type"], best_name)
            counts[key] = counts.get(key, 0) + 1

    rows = [{
        "mode": mode, "drift_type": dtype, "detector": name,
        "cluster": cluster_of.get(name, "—"), "n_first": n,
    } for (mode, dtype, name), n in sorted(counts.items())]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    totals = df.groupby(["mode", "drift_type"])["n_first"].transform("sum")
    df["share"] = df["n_first"] / totals
    return df


def diversity(runs) -> pd.DataFrame:
    """Pairwise Q-statistic and disagreement between the 4 detectors over
    the binary detected/missed vector of all true drifts (single runs)."""
    vectors: dict[tuple, dict[str, list[int]]] = {}
    for r in runs:
        if r["meta"]["method"] not in SINGLE_METHODS:
            continue
        meta = r["meta"]
        name = meta["method"][len("single_"):]
        gt = sorted(meta["ground_truth"], key=lambda d: d["start"])
        matched = match_alarms(r["alarms"], gt,
                               config.MAX_DELAY).matched
        key = (meta["mode"], meta["sid"])
        vectors.setdefault(key, {})[name] = [int(m is not None)
                                             for m in matched]

    by_mode: dict[str, dict[str, list[int]]] = {}
    for (mode, _sid), dets in vectors.items():
        for name, vec in dets.items():
            by_mode.setdefault(mode, {}).setdefault(name, []).extend(vec)

    rows = []
    for mode, dets in sorted(by_mode.items()):
        for a, b in itertools.combinations(config.DETECTOR_ORDER, 2):
            if a not in dets or b not in dets:
                continue
            va, vb = np.array(dets[a]), np.array(dets[b])
            n11 = int(((va == 1) & (vb == 1)).sum())
            n00 = int(((va == 0) & (vb == 0)).sum())
            n10 = int(((va == 1) & (vb == 0)).sum())
            n01 = int(((va == 0) & (vb == 1)).sum())
            denom = n11 * n00 + n01 * n10
            q = (n11 * n00 - n01 * n10) / denom if denom else math.nan
            rows.append({
                "mode": mode, "detector_a": a, "detector_b": b,
                "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "q_statistic": q,
                "disagreement": (n10 + n01) / len(va) if len(va) else
                math.nan,
            })
    return pd.DataFrame(rows)


def significance(metrics: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank on F1 over (generator x seed) pairs, per mode:
    each ensemble vs each single detector, plus each cluster ensemble
    (CLUSTER_METHODS) vs every other ensemble (each pair once)."""
    from scipy.stats import wilcoxon

    comparisons = [(ens, base) for ens in ENSEMBLE_METHODS
                   for base in SINGLE_METHODS]
    for a in CLUSTER_METHODS:
        for b in ENSEMBLE_METHODS:
            if b != a and (b, a) not in comparisons:
                comparisons.append((a, b))
    rows = []
    for mode, sub in metrics.groupby("mode"):
        pivot = sub.pivot_table(index=["generator", "seed"],
                                columns="method", values="f1")
        for a, b in comparisons:
            if a not in pivot or b not in pivot:
                continue
            paired = pivot[[a, b]].dropna()
            x, y = paired[a], paired[b]
            try:
                stat, p = wilcoxon(x, y)
            except ValueError:  # all differences are zero
                stat, p = math.nan, math.nan
            rows.append({
                "mode": mode,
                "comparison": f"{a}_vs_{b}",
                "n_pairs": len(paired),
                "mean_f1_a": float(x.mean()),
                "mean_f1_b": float(y.mean()),
                "mean_diff": float((x - y).mean()),
                "wilcoxon_stat": stat,
                "p_value": p,
            })
    return pd.DataFrame(rows)


def runtime(runs) -> pd.DataFrame:
    df = run_index(runs)
    out = (df.groupby("method", observed=True)
           .agg(n_runs=("runtime_s", "size"),
                mean_runtime_s=("runtime_s", "mean"),
                total_runtime_s=("runtime_s", "sum")).reset_index())
    out["method"] = _method_cat(out["method"])
    return out.sort_values("method")


# ---------------------------------------------------------------------------
# Plots (plotnine)
# ---------------------------------------------------------------------------
def plot_delay_curves(dc: pd.DataFrame) -> list[str]:
    from plotnine import (aes, geom_line, geom_point, ggplot, labs,
                          scale_x_continuous, theme_bw, ylim)

    files = []
    for mode in sorted(dc["mode"].unique()):
        sub = dc[dc["mode"] == mode]
        p = (ggplot(sub, aes("max_delay", "recall", color="method"))
             + geom_line() + geom_point(size=1)
             + scale_x_continuous(breaks=sorted(sub["max_delay"].unique()))
             + ylim(0, 1)
             + labs(title=f"Recall vs allowed delay — {mode}",
                    x="max delay (instances)", y="recall")
             + theme_bw())
        out = plots_dir() / f"delay_curve_{mode}.png"
        p.save(out, width=8, height=4.5, dpi=150, verbose=False)
        files.append(out.name)
    return files


def plot_weights(runs) -> list[str]:
    from plotnine import (aes, geom_step, geom_vline, ggplot, labs,
                          theme_bw, ylim)

    files = []
    for r in runs:
        method = r["meta"]["method"]
        if method not in WEIGHTED_METHODS:
            continue
        meta = r["meta"]
        w_init = WEIGHTED_METHODS[method](meta["config"])
        w = _csv(r, "weights.csv")
        frames = []
        for det in config.DETECTOR_ORDER:
            sub = w[w["detector"] == det] if not w.empty else pd.DataFrame()
            pts = [{"t": 0, "w": w_init, "detector": det}]
            if not sub.empty:
                pts += sub[["t", "w", "detector"]].to_dict("records")
            last = pts[-1]["w"]
            pts.append({"t": meta["length"], "w": last, "detector": det})
            frames.append(pd.DataFrame(pts))
        dfw = pd.concat(frames, ignore_index=True)
        drift_starts = [d["start"] for d in meta["ground_truth"]]
        p = (ggplot(dfw, aes("t", "w", color="detector"))
             + geom_vline(xintercept=drift_starts, linetype="dashed",
                          alpha=0.3)
             + geom_step()
             + ylim(0, 1.05)
             + labs(title=f"Detector weights — {meta['sid']} ({method})",
                    x="instance", y="weight")
             + theme_bw())
        out = plots_dir() / (f"weights_{meta['generator']}_{meta['mode']}_"
                             f"{meta['seed']}_{method}.png")
        p.save(out, width=9, height=4, dpi=150, verbose=False)
        files.append(out.name)
    return files


def sensitivity_summary(runs) -> pd.DataFrame:
    """Final sensitivity level per detector, averaged per mode, plus how
    often each detector was nudged for confirming late."""
    rows = []
    for r in runs:
        meta = r["meta"]
        if meta["method"] not in SENS_METHODS:
            continue
        log = _csv(r, "sensitivity.csv")
        final = meta.get("final_levels") or {}
        for det in config.DETECTOR_ORDER:
            sub = log[log["detector"] == det] if not log.empty else log
            rows.append({
                "method": meta["method"],
                "mode": meta["mode"],
                "detector": det,
                "final_level": final.get(det, 0.0),
                "n_late": len(sub),
            })
    if not rows:
        return pd.DataFrame(columns=["method", "mode", "detector", "n_runs",
                                     "mean_final_level", "min_final_level",
                                     "max_final_level", "mean_n_late"])
    df = pd.DataFrame(rows)
    return (df.groupby(["method", "mode", "detector"])
            .agg(n_runs=("final_level", "size"),
                 mean_final_level=("final_level", "mean"),
                 min_final_level=("final_level", "min"),
                 max_final_level=("final_level", "max"),
                 mean_n_late=("n_late", "mean"))
            .reset_index())


def plot_sensitivity(runs) -> list[str]:
    from plotnine import (aes, geom_step, geom_vline, ggplot, labs,
                          theme_bw, ylim)

    files = []
    for r in runs:
        meta = r["meta"]
        method = meta["method"]
        if method not in SENS_METHODS:
            continue
        sens_cfg = meta["config"]["SENS"]
        log = _csv(r, "sensitivity.csv")
        frames = []
        for det in config.DETECTOR_ORDER:
            sub = (log[log["detector"] == det] if not log.empty
                   else pd.DataFrame())
            pts = [{"t": 0, "level": 0.0, "detector": det}]
            if not sub.empty:
                pts += sub[["t", "level", "detector"]].to_dict("records")
            pts.append({"t": meta["length"], "level": pts[-1]["level"],
                        "detector": det})
            frames.append(pd.DataFrame(pts))
        dfl = pd.concat(frames, ignore_index=True)
        drift_starts = [d["start"] for d in meta["ground_truth"]]
        p = (ggplot(dfl, aes("t", "level", color="detector"))
             + geom_vline(xintercept=drift_starts, linetype="dashed",
                          alpha=0.3)
             + geom_step()
             + ylim(0, sens_cfg["L_MAX"])
             + labs(title=f"Detector sensitivity — {meta['sid']} ({method})",
                    x="instance",
                    y="sensitivity level (+ = more sensitive)")
             + theme_bw())
        out = plots_dir() / (f"sensitivity_{meta['generator']}_"
                             f"{meta['mode']}_{meta['seed']}_{method}.png")
        p.save(out, width=9, height=4, dpi=150, verbose=False)
        files.append(out.name)
    return files


HEATMAP_PARAMS = {
    "STEPD": ("window_size", "alpha_drift"),
    "SEED": ("delta", "epsilon_prime"),
    "HDDMWeighted": ("drift_confidence", "lambda_"),
    "RDDM": ("min_n_instances", "drift_level"),
}


def plot_tuning_heatmaps() -> list[str]:
    from plotnine import (aes, geom_text, geom_tile, ggplot, labs,
                          scale_fill_gradient, theme_bw)

    path = config.TUNING_DIR / "tuning_runs.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).drop_duplicates(
        subset=["detector", "params", "generator", "mode", "seed"],
        keep="last")
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
                 .groupby(["_p1", "_p2"])["f1"].mean().reset_index())
            m["label"] = m["f1"].map(lambda v: f"{v:.2f}")
            m["_p1"] = pd.Categorical(
                m["_p1"].astype(str),
                categories=[str(v) for v in sorted(m["_p1"].unique())])
            m["_p2"] = pd.Categorical(
                m["_p2"].astype(str),
                categories=[str(v) for v in sorted(m["_p2"].unique())])
            p = (ggplot(m, aes("_p1", "_p2", fill="f1"))
                 + geom_tile() + geom_text(aes(label="label"), size=7)
                 + scale_fill_gradient(low="#f7fbff", high="#08519c",
                                       limits=(0, 1))
                 + labs(title=f"{detector} mean F1 — {mode}", x=p1, y=p2)
                 + theme_bw())
            out = plots_dir() / f"tuning_heatmap_{detector}_{mode}.png"
            p.save(out, width=7, height=5, dpi=150, verbose=False)
            files.append(out.name)
    return files


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def analyze(make_plots: bool = True) -> dict:
    """Build every table (and optionally every plot); returns the tables."""
    import traceback

    runs = load_runs()
    if not runs:
        raise FileNotFoundError(
            f"no finished runs under {config.TEST_DIR} — run "
            f"drift_ensemble/scripts/04_run_test.py first")
    td = tables_dir()
    out: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    def section(name, fn, *args, tex=False):
        try:
            df = fn(*args)
            out[name] = df
            df.to_csv(td / f"{name}.csv", index=False)
            print(f"  tables/{name}.csv  ({len(df)} rows)")
        except Exception:
            failures.append(name)
            print(f"  FAILED: {name}\n{traceback.format_exc()}")

    metrics = per_run_metrics(runs)
    out["per_run_metrics"] = metrics
    metrics.to_csv(td / "per_run_metrics.csv", index=False)

    def detection(_):
        agg, pretty = detection_metrics(metrics)
        (td / "detection_metrics.tex").write_text(
            pretty.to_latex(index=False), encoding="utf-8")
        return agg

    idx = run_index(runs)
    out["run_index"] = idx
    idx.to_csv(config.ANALYSIS_DIR / "run_index.csv", index=False)
    print(f"  run_index.csv  ({len(idx)} rows)")

    section("detection_metrics", detection, None)
    section("detection_medians", detection_medians, metrics)
    section("delay_curve", delay_curve_table, runs)
    section("per_type_mixed", per_type_mixed, runs)
    section("first_to_fire", first_to_fire, runs)
    section("diversity", diversity, runs)
    section("significance", significance, metrics)
    section("sensitivity_summary", sensitivity_summary, runs)
    section("runtime", runtime, runs)

    if make_plots:
        for name, fn, args in [
            ("delay_curve plots", plot_delay_curves, (out["delay_curve"],)),
            ("weight plots", plot_weights, (runs,)),
            ("sensitivity plots", plot_sensitivity, (runs,)),
            ("tuning heatmaps", plot_tuning_heatmaps, ()),
        ]:
            try:
                files = fn(*args)
                print(f"  plots: {name}: {len(files)} file(s)")
            except Exception:
                failures.append(name)
                print(f"  FAILED: {name}\n{traceback.format_exc()}")

    if failures:
        raise RuntimeError(f"analysis sections failed: {failures}")
    return out
