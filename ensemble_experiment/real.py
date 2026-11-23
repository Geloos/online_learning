"""Extended DDD on real datasets (no known drift positions).

Real data has no ground truth, so only what needs none is reported:
prequential accuracy (test-then-train, after the MIN_TRAINING_SIZE
warm-up), the number of detections and the accuracy over time. Every
detection-quality metric (TP/FP, delay, recovery, recall hits) is left out.

Runs: config.REAL_METHODS (ens_raw_resetall), ONE run per dataset with
seed config.REAL_SEED (the seed drives the online-bagging Poisson draws),
plus one baseline:
  no_change     predicts the previous instance's label (no learner, no
                seed); Electricity is known to be strongly autocorrelated,
                so this is the bar a learner has to clear

Run folders: results/results_ensemble/real/<dataset>,<seed>,<method>/
(correct.npy, alarms.json, episodes.csv, steps.csv, meta.json last).
Statistics (computed from those folders only):
statistics/ensemble_experiment/real/<dataset>/.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from ensemble_experiment import config
from ensemble_experiment.bagging import EnsembleFactory
from ensemble_experiment.ddd import ExtendedDDD
from ensemble_experiment.detection import make_signal
from ensemble_experiment.logger import EPISODE_FIELDS, STEP_FIELDS, RunLogger
from ensemble_experiment.runner import _clean, _json, _method_params, \
    _write_csv

BASELINES = ["no_change"]
METHODS = list(config.REAL_METHODS) + BASELINES


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dataset_path(name: str) -> Path:
    """Local ARFF of a real dataset; downloaded once and checked against
    its md5 (a wrong file would silently change every result)."""
    spec = config.REAL_DATASETS[name]
    path = config.DATA_DIR / spec["file"]
    if not path.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.replace(path)
    if _md5(path) != spec["md5"]:
        raise ValueError(f"{path}: md5 {_md5(path)} != {spec['md5']} — "
                         f"delete it and download again")
    return path


def load_stream(name: str):
    from capymoa.stream import ARFFStream

    return ARFFStream(str(dataset_path(name)))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_specs(datasets=None, methods=None):
    """One run per dataset x method, all with seed config.REAL_SEED."""
    datasets = list(config.REAL_DATASETS) if datasets is None else datasets
    methods = METHODS if methods is None else methods
    return [(d, config.REAL_SEED, m) for d in datasets for m in methods]


def run_name(dataset: str, seed: int, method: str) -> str:
    return f"{dataset},{seed},{method}"


def run_dir(dataset, seed, method, root=None) -> Path:
    root = config.REAL_RUNS_DIR if root is None else Path(root)
    return root / run_name(dataset, seed, method)


def is_done(dataset, seed, method, root=None) -> bool:
    return (run_dir(dataset, seed, method, root) / "meta.json").exists()


def _meta(dataset, seed, method, length, summary, t0,
          detector_params=None) -> dict:
    return {"dataset": dataset, "seed": seed, "method": method,
            "length": length,
            "summary": {k: _clean(v) for k, v in summary.items()},
            "config": config.snapshot(),
            "detector_params": detector_params,  # incl. cluster minimums
            "runtime_s": time.perf_counter() - t0}


def _run_no_change(dataset, seed, out: Path, t0) -> Path:
    stream = load_stream(dataset)
    labels = []
    while stream.has_more_instances():
        labels.append(int(stream.next_instance().y_index))
    y = np.asarray(labels)
    correct = np.zeros(len(y), dtype=np.int8)
    correct[1:] = y[1:] == y[:-1]
    warm = correct[config.MIN_TRAINING_SIZE:]
    summary = {"acc": float(warm.mean()), "acc_all": float(correct.mean()),
               "n_detections": 0}
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "correct.npy", correct)
    (out / "meta.json").write_text(_json(
        _meta(dataset, seed, "no_change", len(y), summary, t0)))
    return out


def run_one(dataset: str, seed: int, method: str, *, out_root=None,
            force: bool = False) -> Path:
    out = run_dir(dataset, seed, method, out_root)
    if (out / "meta.json").exists() and not force:
        return out
    t0 = time.perf_counter()
    if method == "no_change":
        return _run_no_change(dataset, seed, out, t0)

    length = config.REAL_DATASETS[dataset]["length"]
    stream = load_stream(dataset)
    params = _method_params(method)  # ens_raw_resetall -> phase 1
    signal = make_signal(method, params)
    factory = EnsembleFactory(stream.get_schema(), config.DDD, seed)
    # No ground truth: one dummy concept for the logger.
    logger = RunLogger(np.zeros(length, dtype=np.int64),
                       min_training_size=config.MIN_TRAINING_SIZE,
                       log_every=config.LOG_EVERY)
    ddd = ExtendedDDD(config.DDD, factory, signal,
                      min_training_size=config.MIN_TRAINING_SIZE,
                      listener=logger.on_event)
    logger.attach(ddd)
    t = 0
    while stream.has_more_instances():
        logger.record(t, ddd.step(t, stream.next_instance()))
        t += 1
    assert t == length, f"{dataset}: {t} instances, expected {length}"
    ddd.finish(length - 1)

    # Keep only the fields that need no ground truth.
    episodes = logger.episode_rows([], config.MAX_DELAY,
                                   config.DUPLICATES_AS_FP)
    gt_free = [f for f in EPISODE_FIELDS
               if f not in ("label", "drift_idx", "delay", "true_type",
                            "true_recurring", "kind_correct",
                            "pool_add_tag", "recalled_tag",
                            "true_concept_at_recall", "recall_hit")]
    full = logger.summary(episodes, list(signal.alarms), [], length,
                          config.MAX_DELAY, config.DUPLICATES_AS_FP)
    summary = {k: full[k] for k in full
               if k in ("acc", "acc_all", "n_detections", "frac_after",
                        "mean_steps_after", "final_cap_fast",
                        "final_cap_slow", "n_recalls")
               or k.startswith("exit_")}

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "steps.csv",
               [f for f in STEP_FIELDS if f != "concept"], logger.rows)
    _write_csv(out / "episodes.csv", gt_free, episodes)
    np.save(out / "correct.npy", logger.correct)
    (out / "alarms.json").write_text(_json({
        "alarms": list(signal.alarms), "raw_fires": signal.raw_fires,
        **signal.logs}))
    (out / "meta.json").write_text(_json(  # last: marks completion
        _meta(dataset, seed, method, length, summary, t0, params)))
    return out


def run_spec(spec) -> str:
    """Pool-friendly wrapper."""
    run_one(*spec)
    return run_name(*spec)


# ---------------------------------------------------------------------------
# Statistics (reads run folders only)
# ---------------------------------------------------------------------------
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
    import pandas as pd

    df = pd.DataFrame([{"dataset": r["meta"]["dataset"],
                        "method": r["meta"]["method"],
                        "seed": r["meta"]["seed"],
                        "acc": r["meta"]["summary"]["acc"],
                        "acc_all": r["meta"]["summary"]["acc_all"],
                        "n_detections": r["meta"]["summary"]["n_detections"]}
                       for r in runs])
    df["method"] = pd.Categorical(df["method"], METHODS, ordered=True)
    return df.sort_values(["dataset", "method"]).reset_index(drop=True)


def plot_accuracy(runs, dataset: str, out, block: int) -> None:
    """Accuracy per block of ``block`` instances, one line per method; the
    no-change baseline grey and dotted, DDD's detections as ticks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, method in enumerate(METHODS):
        rs = [r for r in runs if r["meta"]["dataset"] == dataset
              and r["meta"]["method"] == method]
        if not rs:
            continue
        r = rs[0]
        c = r["correct"]
        n = len(c) // block * block
        acc = c[:n].reshape(-1, block).mean(axis=1)
        x = (np.arange(len(acc)) + 0.5) * block
        if method in BASELINES:
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
