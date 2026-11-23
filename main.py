"""Run both experiments end to end, or one step of them.

    python main.py [--workers N] [--yes] [--only EXPERIMENT] [--dry-run]
    python main.py STEP [--workers N] [--yes] [step options]

Steps, in the order of a full run:

  drift_ensemble (experiment 1)
    cache      cache the HT errors of the 18 tuning streams
    tune       tuning phase 1: every grid config on the tuning streams
    select     tuning phase 1: the best config per detector x mode
    phase2     tuning phase 2: each cluster's shared minimum (used by the
               two-vote methods)
    test       cache the 60 test streams and run the 7 methods on them
    stats      statistics -> statistics/drift_ensemble/
  ensemble_experiment (experiment 2)
    ddd        extended DDD on the synthetic streams
    ddd-stats  statistics -> statistics/ensemble_experiment/
    real       DDD on Electricity, Airlines, Covtype and Poker, and their
               statistics (always with REAL_JVM_ARGS, at most
               REAL_MAX_WORKERS workers)

  screen       (not part of the full run) DDM, HDDMAverage and ADWIN
               screened against the study's 4

A full run starts each step as its own process and stops at the first
failure. Every step is resumable: finished work is skipped. Long steps time
one unit of work, print an estimate and ask for confirmation; --yes skips
the question.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("drift_ensemble", "ensemble_experiment")


# ===========================================================================
# Shared helpers: time estimate, confirmation, worker pool
# ===========================================================================
def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, s = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m:02d}m"


def gate(unit_seconds: float, n_remaining: int, workers: int, yes: bool,
         what: str) -> None:
    """Print the estimated total time and stop for confirmation (unless
    --yes)."""
    total = unit_seconds * n_remaining / max(1, workers)
    print(f"One {what} took {fmt_duration(unit_seconds)}; "
          f"{n_remaining} more with {workers} worker(s) -> "
          f"estimated {fmt_duration(total)}.")
    if yes:
        print("--yes given, proceeding.")
        return
    try:
        ans = input("Proceed? [y/N] ")
    except EOFError:
        ans = ""
    if ans.strip().lower() not in ("y", "yes"):
        sys.exit("Aborted. Re-run with --yes to skip this prompt.")


def run_pool(fn, units, *, workers: int, yes: bool, what: str, done=None,
             label=None, init=None, initargs=(), every: int = 1) -> None:
    """Run fn on every unit: the first one here (timed), then, after the
    time estimate is confirmed, the rest in a pool of spawned workers (one
    JVM each). done(result) runs in this process for every result, in
    completion order; label(result) names it in the progress lines."""
    if not units:
        return
    if init is not None:
        init(*initargs)
    t0 = time.perf_counter()
    result = fn(units[0])
    unit = time.perf_counter() - t0
    if done is not None:
        done(result)
    if label is not None:
        print(f"  done {label(result)}", flush=True)
    rest = units[1:]
    if not rest:
        return
    workers = max(1, workers)
    gate(unit, len(rest), workers, yes, what)
    start = time.perf_counter()

    def progress(i, result):
        if done is not None:
            done(result)
        if i % every == 0 or i == len(rest):
            eta = (time.perf_counter() - start) / i * (len(rest) - i)
            name = label(result) if label is not None else f"{what}s done"
            print(f"  [{i}/{len(rest)}] {name}  (ETA {fmt_duration(eta)})",
                  flush=True)

    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=init, initargs=initargs) as pool:
            for i, result in enumerate(pool.imap_unordered(fn, rest), 1):
                progress(i, result)
    else:
        for i, u in enumerate(rest, 1):
            progress(i, fn(u))


def _cache_streams(specs, args, label: str) -> None:
    from drift_ensemble.runs import cache_stream_spec, is_cached
    from streams import stream_id

    pending = [s for s in specs if not is_cached(stream_id(*s))]
    print(f"{label}: {len(specs)} streams, {len(pending)} to cache.")
    run_pool(cache_stream_spec, pending, workers=args.workers, yes=args.yes,
             what="stream", label=str)


def _tuning_streams_cached() -> list[str]:
    from drift_ensemble.runs import is_cached
    from drift_ensemble.tuning import tuning_stream_ids

    ids = tuning_stream_ids()
    missing = [s for s in ids if not is_cached(s)]
    if missing:
        raise SystemExit(f"{len(missing)} tuning streams are not cached yet "
                         f"(e.g. {missing[0]}) — run `python main.py cache` "
                         f"first")
    return ids


def _show(df, title: str = "") -> None:
    import pandas as pd

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_colwidth", 70,
                           "display.float_format", "{:.3f}".format):
        if title:
            print(title)
        print(df.to_string(index=False))


# ===========================================================================
# Experiment 1: drift_ensemble
# ===========================================================================
def step_cache(args) -> None:
    from drift_ensemble import config

    print(f"Cache dir: {config.CACHE_DIR}")
    _cache_streams(config.tuning_stream_specs(), args, "Tuning streams")


def step_tune(args) -> None:
    from drift_ensemble import tuning

    ids = _tuning_streams_cached()
    tasks = tuning.all_tasks()[:args.limit]
    done = tuning.done_counts(tuning.TUNING_RUNS_CSV)
    pending = [t for t in tasks if done.get(t, 0) < len(ids)]
    print(f"{len(tasks)} configs x {len(ids)} streams; {len(pending)} still "
          f"to evaluate.")
    run_pool(tuning.run_task, pending, workers=args.workers, yes=args.yes,
             what="config", init=tuning.worker_init, initargs=(ids,),
             every=10,
             done=lambda rows: tuning.append_rows(
                 tuning.TUNING_RUNS_CSV, tuning.ROW_FIELDS, rows))
    print(f"Done — {tuning.TUNING_RUNS_CSV}")


def step_select(args) -> None:
    import json

    from drift_ensemble import config, tuning

    df = tuning.load_tuning_runs()
    print(f"{len(df)} tuning rows ({df['detector'].nunique()} detectors, "
          f"{df.groupby('detector')['params'].nunique().to_dict()} configs)")
    selected, summary = tuning.select_params(df, args.objective)
    out_json = config.TUNING_DIR / "selected_params.json"
    out_json.write_text(json.dumps(selected, indent=2))
    summary.to_csv(config.TUNING_DIR / "tuning_summary.csv", index=False)
    tuning.logo_check(df, args.objective).to_csv(
        config.TUNING_DIR / "logo_check.csv", index=False)

    cluster_of = {name: cl for cl, ms in config.CLUSTER_MEMBERS.items()
                  for name in ms}
    summary.insert(1, "cluster", summary["detector"].map(cluster_of))
    _show(summary[["detector", "cluster", "mode", "mean_f1", "mean_recall",
                   "mean_precision", "mean_mean_delay", "mean_far"]])
    print(f"\nWrote {out_json} and tuning_summary.csv")
    print(f"\nCluster assignment in drift_ensemble/config.py: "
          f"{config.CLUSTER_MEMBERS}. Check that tuning_summary.csv still "
          f"justifies it: a detector belongs to the cluster whose params "
          f"hurt it least on the other stream type (see config.py).")


def step_phase2(args) -> None:
    import json

    import pandas as pd

    from drift_ensemble import config, tuning

    ids = _tuning_streams_cached()
    per_mode = {m: sum(1 for s in ids if s.split(",")[1] == m)
                for m in config.PURE_MODES}
    tasks = tuning.phase2_tasks()
    done = tuning.done_counts(tuning.PHASE2_RUNS_CSV, tuning.PHASE2_TASK_KEY)
    pending = [t for t in tasks
               if done.get(t, 0) < tuning.phase2_n_rows(t, per_mode[t[0]])]
    print(f"Phase 2: M grid {config.MIN_INSTANCES_GRID}, clusters "
          f"{config.CLUSTER_MEMBERS}")
    print(f"{len(tasks)} configs (each on its cluster's {per_mode} tuning "
          f"streams); {len(pending)} still to evaluate.")
    run_pool(tuning.phase2_run_task, pending, workers=args.workers,
             yes=args.yes, what="config", init=tuning.worker_init,
             initargs=(ids,), every=10,
             done=lambda rows: tuning.append_rows(
                 tuning.PHASE2_RUNS_CSV, tuning.PHASE2_ROW_FIELDS, rows))

    # Selection, always redone from the full CSV.
    phase1_path = config.TUNING_DIR / "tuning_summary.csv"
    phase1 = pd.read_csv(phase1_path) if phase1_path.exists() else None
    chosen, summary = tuning.select_min_instances(tuning.load_phase2_runs(),
                                                  phase1_summary=phase1)
    summary.to_csv(tuning.PHASE2_SUMMARY_CSV, index=False)
    config.CLUSTER_PARAMS_PATH.write_text(json.dumps(chosen, indent=2))

    obj = config.TUNE_OBJECTIVE
    _show(summary[["cluster", "min_instances", "cluster_score", "detector",
                   f"mean_{obj}", "mean_precision", "mean_recall",
                   "mean_mean_delay", f"phase1_mean_{obj}", "chosen"]],
          f"\nBest config per member at each M (cluster_score = mean of the "
          f"two members' {obj}; phase1 = no shared minimum):")
    print("\nChosen minimum per cluster: "
          + ", ".join(f"{c} M = {m}"
                      for c, m in chosen["min_instances"].items()))
    for cluster in config.CLUSTER_MEMBERS:
        for name, params in chosen[cluster].items():
            print(f"  {cluster:8s} {name:13s} {params}")
    print(f"\nWrote {tuning.PHASE2_SUMMARY_CSV}\n"
          f"Wrote {config.CLUSTER_PARAMS_PATH}")


def step_test(args) -> None:
    from drift_ensemble import config
    from drift_ensemble.runs import (
        is_run_done,
        load_cluster_params,
        load_selected_params,
        run_and_save,
    )
    from streams import stream_id

    load_selected_params()  # fail fast if phase 1 has not run
    params = load_cluster_params()  # ... or phase 2
    print(f"Phase-1 params (singles, ens_raw_resetall): "
          f"{config.TUNING_DIR / 'selected_params.json'}")
    print(f"Phase-2 params ({', '.join(config.PHASE2_METHODS)}): "
          f"{config.CLUSTER_PARAMS_PATH}, min_instances = "
          f"{params['min_instances']}")

    specs = config.test_stream_specs()
    print(f"Cache dir: {config.CACHE_DIR}")
    _cache_streams(specs, args, "Test streams")

    methods = args.methods or config.METHODS
    units = [(g, m, s, meth) for (g, m, s) in specs for meth in methods
             if not is_run_done(stream_id(g, m, s), meth)]
    print(f"\nTest runs: {len(specs)} streams x {len(methods)} methods; "
          f"{len(units)} still to run.")
    config.TEST_DIR.mkdir(parents=True, exist_ok=True)
    run_pool(run_and_save, units, workers=args.workers, yes=args.yes,
             what="run",
             label=lambda r: f"{r[0]},{r[1]} ({fmt_duration(r[2])})")
    print(f"Done — results under {config.TEST_DIR}")


def step_stats(args) -> None:
    from drift_ensemble import config
    from drift_ensemble.stats import run_statistics

    print(f"Statistics of the runs under {config.TEST_DIR} ...")
    summary = run_statistics()
    if summary.empty:
        return
    _show(summary, f"\nPer method x mode (max delay {config.MAX_DELAY}): "
                   f"mean and std of P/R/F1 over runs; delay = median over "
                   f"detected drifts")
    print(f"\nStatistics written to {config.STATISTICS_DIR}")


def step_screen(args) -> None:
    from drift_ensemble import config, tuning
    from drift_ensemble.runs import is_cached, load_selected_params
    from streams import stream_id

    study = load_selected_params()
    tune_ids = tuning.tuning_stream_ids()
    test_ids = [stream_id(*s) for s in config.test_stream_specs()
                if s[1] in config.PURE_MODES]
    missing = [s for s in tune_ids + test_ids if not is_cached(s)]
    if missing:
        raise SystemExit(f"{len(missing)} streams are not cached (e.g. "
                         f"{missing[0]}); run `python main.py cache` and "
                         f"`test` first")

    def stage(tasks, ids, path, label):
        done = tuning.done_counts(path)
        pending = [t for t in tasks if done.get(t, 0) < len(ids)]
        print(f"{label}: {len(tasks)} configs x {len(ids)} streams; "
              f"{len(pending)} still to evaluate.")
        run_pool(tuning.run_task, pending, workers=args.workers,
                 yes=args.yes, what="config", init=tuning.worker_init,
                 initargs=(ids,), every=10,
                 done=lambda rows: tuning.append_rows(
                     path, tuning.ROW_FIELDS, rows))

    stage(tuning.screening_tuning_tasks(study), tune_ids,
          tuning.SCREENING_TUNING_CSV, "Tuning streams")
    tuning_df = tuning.load_screening_rows(tuning.SCREENING_TUNING_CSV)
    selected = tuning.screening_selected(tuning_df, study)
    (config.SCREENING_DIR / "selected_params.json").write_text(
        tuning.screening_selected_json(selected))
    stage(tuning.screening_test_tasks(selected), test_ids,
          tuning.SCREENING_TEST_CSV, "Test streams (ABRUPT, GRADUAL)")
    summary = tuning.screening_summary(
        tuning_df, tuning.load_screening_rows(tuning.SCREENING_TEST_CSV),
        selected)
    config.STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tuning.SCREENING_SUMMARY_CSV, index=False)
    _show(summary.drop(columns=["test_f1_mean"]),
          "\nEach detector with the config tuned for the mode (tuning_f1 = "
          "mean over 9 tuning streams; test_* = medians over 15 test "
          "streams)")
    print(f"\nWritten to {tuning.SCREENING_SUMMARY_CSV}")


# ===========================================================================
# Experiment 2: ensemble_experiment
# ===========================================================================
def step_ddd(args) -> None:
    from ensemble_experiment.runs import is_done, run_spec, run_specs

    specs = run_specs(args.generators, args.modes, args.seeds, args.methods)
    pending = [s for s in specs if not is_done(*s)]
    print(f"{len(specs)} runs, {len(pending)} to do.")
    run_pool(run_spec, pending, workers=args.workers, yes=args.yes,
             what="run", label=str)


def step_ddd_stats(args) -> None:
    from ensemble_experiment import config
    from ensemble_experiment.analysis import run_analysis

    recovery = run_analysis()
    if recovery.empty:
        return
    _show(recovery, f"{config.STATS['method']}: recovery after detected "
                    f"real drifts (medians)")
    print(f"\nStatistics written to {config.ANALYSIS_DIR}")


def step_real(args) -> None:
    from ensemble_experiment import config

    # The large datasets need more heap; the JVM starts after this line.
    os.environ["CAPYMOA_JVM_ARGS"] = config.REAL_JVM_ARGS
    workers = min(args.workers, config.REAL_MAX_WORKERS)
    print(f"CAPYMOA_JVM_ARGS={config.REAL_JVM_ARGS}, {workers} worker(s)")

    from ensemble_experiment.analysis import real_statistics
    from ensemble_experiment.runs import (
        dataset_path,
        real_is_done,
        real_run_spec,
        real_run_specs,
    )

    specs = real_run_specs(args.datasets, args.methods)
    for d in dict.fromkeys(s[0] for s in specs):
        dataset_path(d)  # download once, before the workers start
    pending = [s for s in specs if not real_is_done(*s)]
    print(f"{len(specs)} runs, {len(pending)} to do.")
    run_pool(real_run_spec, pending, workers=workers, yes=args.yes,
             what="run", label=str)
    table = real_statistics()
    if table is not None:
        print(table.to_string(index=False, float_format="{:.4f}".format))
        print(f"\nStatistics written to {config.REAL_STATS_DIR}")


# ===========================================================================
# Command line
# ===========================================================================
# name -> (function, experiment (None = not in the full run), description)
STEPS = {
    "cache": (step_cache, "drift_ensemble",
              "cache the HT errors of the tuning streams"),
    "tune": (step_tune, "drift_ensemble",
             "tuning phase 1: every grid config on the tuning streams"),
    "select": (step_select, "drift_ensemble",
               "tuning phase 1: best config per detector x mode"),
    "phase2": (step_phase2, "drift_ensemble",
               "tuning phase 2: each cluster's shared minimum"),
    "test": (step_test, "drift_ensemble",
             "run the 7 methods on the test streams"),
    "stats": (step_stats, "drift_ensemble", "statistics of experiment 1"),
    "ddd": (step_ddd, "ensemble_experiment",
            "run extended DDD on the synthetic streams"),
    "ddd-stats": (step_ddd_stats, "ensemble_experiment",
                  "statistics of the DDD runs"),
    "real": (step_real, "ensemble_experiment",
             "DDD on the real datasets, and their statistics"),
    "screen": (step_screen, None,
               "screen DDM / HDDMAverage / ADWIN (not in the full run)"),
}


def build_parser() -> argparse.ArgumentParser:
    from drift_ensemble import config as c1
    from ensemble_experiment import config as c2

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `python main.py STEP -h` for a step's options.")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel worker processes, one JVM each "
                         "(default 6)")
    ap.add_argument("--yes", action="store_true",
                    help="skip every time-estimate confirmation")
    ap.add_argument("--only", choices=EXPERIMENTS,
                    help="full run: only one of the two experiments")
    ap.add_argument("--dry-run", action="store_true",
                    help="full run: print the steps without running them")

    # Step parsers repeat --workers / --yes; SUPPRESS keeps the top-level
    # values when a step does not give them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=argparse.SUPPRESS,
                        help="parallel worker processes (default 6)")
    common.add_argument("--yes", action="store_true",
                        default=argparse.SUPPRESS,
                        help="skip the time-estimate confirmation")
    sub = ap.add_subparsers(dest="step", metavar="STEP")
    p = {name: sub.add_parser(name, parents=[common], help=desc,
                              description=desc)
         for name, (_, _, desc) in STEPS.items()}
    p["tune"].add_argument("--limit", type=int, default=None,
                           help="only the first N configs (smoke testing)")
    p["select"].add_argument("--objective", default=None,
                             choices=["f1", "f2", "delay_auc"],
                             help="default: config.TUNE_OBJECTIVE")
    p["test"].add_argument("--methods", nargs="+", choices=c1.METHODS)
    p["ddd"].add_argument("--generators", nargs="+", choices=c2.GENERATORS)
    p["ddd"].add_argument("--modes", nargs="+", choices=c2.MODES)
    p["ddd"].add_argument("--seeds", nargs="+", type=int)
    p["ddd"].add_argument("--methods", nargs="+", choices=c2.METHODS)
    p["real"].add_argument("--datasets", nargs="+",
                           choices=list(c2.REAL_DATASETS))
    p["real"].add_argument("--methods", nargs="+",
                           choices=c2.REAL_RUN_METHODS)
    return ap


def full_run(args) -> None:
    plan = [name for name, (_, exp, _) in STEPS.items()
            if exp is not None and (args.only is None or exp == args.only)]
    for i, name in enumerate(plan, 1):
        _, exp, desc = STEPS[name]
        extra = ["--workers", str(args.workers)] + (
            ["--yes"] if args.yes else [])
        print(f"\n=== [{i}/{len(plan)}] {exp}: {desc} "
              f"(python main.py {' '.join([name, *extra])})", flush=True)
        if args.dry_run:
            continue
        subprocess.run([sys.executable, str(ROOT / "main.py"), name, *extra],
                       check=True, cwd=ROOT)
    print("" if args.dry_run else "\nDone.")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    args = build_parser().parse_args()
    if args.step is None:
        full_run(args)
    else:
        STEPS[args.step][0](args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nStep failed (exit code {e.returncode}): "
                 f"{' '.join(map(str, e.cmd))}")
