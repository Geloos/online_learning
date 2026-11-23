"""Tuning phase 2: tune one shared minimum per cluster
(config.MIN_INSTANCES_GRID), then pick it.

Appends rows to results/results_drift_ensemble/tuning/phase2_runs.csv
(resumable: finished configs are skipped), then writes phase2_summary.csv
and cluster_params.json: the params and the minimum that the two-vote
methods (config.PHASE2_METHODS: ens_cluster_vote, ens_cluster_sens) use in
the test runs (04) and the DDD runs (ensemble_experiment). The singles and
ens_raw_resetall keep the phase-1 params.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=cli.default_workers())
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from drift_ensemble import config
    from drift_ensemble import min_instances as p2
    from drift_ensemble.cache import is_cached
    from drift_ensemble.tuning import tuning_stream_ids, worker_init

    stream_ids = tuning_stream_ids()
    missing = [sid for sid in stream_ids if not is_cached(sid)]
    if missing:
        raise SystemExit(
            f"{len(missing)} tuning streams are not cached yet (e.g. "
            f"{missing[0]}) — run 01_cache_tuning_streams.py first")
    per_mode = {m: sum(1 for s in stream_ids if s.split(",")[1] == m)
                for m in config.PURE_MODES}

    tasks = p2.all_tasks()
    done = p2.done_counts()
    pending = [t for t in tasks
               if done.get(t, 0) < p2.n_rows(t, per_mode[t[0]])]
    print(f"Phase 2: M grid {config.MIN_INSTANCES_GRID}, clusters "
          f"{config.CLUSTER_MEMBERS}")
    print(f"{len(tasks)} configs (each on its cluster's "
          f"{per_mode} tuning streams); {len(pending)} still to evaluate.")

    if pending:
        config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
        worker_init(stream_ids)
        t0 = time.perf_counter()
        p2.append_rows(p2.run_task(pending[0]))
        unit = time.perf_counter() - t0
        remaining = pending[1:]
        if remaining:
            cli.gate(unit, len(remaining), args.workers, args.yes, "config")
            start = time.perf_counter()

            def progress(i):
                if i % 10 == 0 or i == len(remaining):
                    eta = ((time.perf_counter() - start) / i
                           * (len(remaining) - i))
                    print(f"  [{i}/{len(remaining)}] configs done "
                          f"(ETA {cli.fmt_duration(eta)})", flush=True)

            if args.workers > 1:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(args.workers, initializer=worker_init,
                              initargs=(stream_ids,)) as pool:
                    for i, rows in enumerate(
                            pool.imap_unordered(p2.run_task, remaining), 1):
                        p2.append_rows(rows)
                        progress(i)
            else:
                for i, task in enumerate(remaining, 1):
                    p2.append_rows(p2.run_task(task))
                    progress(i)

    # Selection (always re-done from the full CSV).
    phase1_path = config.TUNING_DIR / "tuning_summary.csv"
    phase1 = pd.read_csv(phase1_path) if phase1_path.exists() else None
    chosen, summary = p2.select_min_instances(p2.load_runs(),
                                              phase1_summary=phase1)
    summary.to_csv(p2.SUMMARY_CSV, index=False)
    config.CLUSTER_PARAMS_PATH.write_text(json.dumps(chosen, indent=2))

    obj = config.TUNE_OBJECTIVE
    cols = ["cluster", "min_instances", "cluster_score", "detector",
            f"mean_{obj}", "mean_precision", "mean_recall",
            "mean_mean_delay", f"phase1_mean_{obj}", "chosen"]
    print("\nBest config per member at each M (cluster_score = mean of the "
          f"two members' {obj}; phase1 = no shared minimum):")
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))
    print("\nChosen minimum per cluster: "
          + ", ".join(f"{c} M = {m}"
                      for c, m in chosen["min_instances"].items()))
    for cluster in config.CLUSTER_MEMBERS:
        for name, params in chosen[cluster].items():
            print(f"  {cluster:8s} {name:13s} {params}")
    print(f"\nWrote {p2.SUMMARY_CSV}\nWrote {config.CLUSTER_PARAMS_PATH}")


if __name__ == "__main__":
    main()
