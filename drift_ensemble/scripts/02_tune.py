"""Evaluate every detector config on every cached tuning stream.

Appends one row per (detector, config, stream) to
results/results_drift_ensemble/tuning/tuning_runs.csv. Resumable: finished
configs are skipped.
"""

import argparse
import csv
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def _load_done_counts(path, n_fields) -> dict:
    done: dict[tuple[str, str], int] = {}
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["detector"], row["params"])
            done[key] = done.get(key, 0) + 1
    return done


def _append_rows(path, fields, rows) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=cli.default_workers())
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N configs (smoke testing)")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.cache import is_cached
    from drift_ensemble.tuning import (
        ROW_FIELDS,
        TUNING_RUNS_CSV,
        all_tasks,
        run_task,
        tuning_stream_ids,
        worker_init,
    )

    stream_ids = tuning_stream_ids()
    missing = [sid for sid in stream_ids if not is_cached(sid)]
    if missing:
        raise SystemExit(
            f"{len(missing)} tuning streams are not cached yet "
            f"(e.g. {missing[0]}) — run drift_ensemble/scripts/01_cache_tuning_streams.py "
            f"first")

    tasks = all_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    done = _load_done_counts(TUNING_RUNS_CSV, len(ROW_FIELDS))
    pending = [t for t in tasks if done.get(t, 0) < len(stream_ids)]
    print(f"{len(tasks)} configs x {len(stream_ids)} streams; "
          f"{len(pending)} configs still to evaluate.")
    if not pending:
        print(f"Nothing to do — see {TUNING_RUNS_CSV}")
        return
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)

    # Time one unit of work (one config on all streams), then confirm.
    worker_init(stream_ids)
    t0 = time.perf_counter()
    rows = run_task(pending[0])
    unit = time.perf_counter() - t0
    _append_rows(TUNING_RUNS_CSV, ROW_FIELDS, rows)
    remaining = pending[1:]
    if not remaining:
        print(f"Done — {TUNING_RUNS_CSV}")
        return
    cli.gate(unit, len(remaining), args.workers, args.yes, "config")

    start = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=worker_init,
                      initargs=(stream_ids,)) as pool:
            for i, rows in enumerate(
                    pool.imap_unordered(run_task, remaining), 1):
                _append_rows(TUNING_RUNS_CSV, ROW_FIELDS, rows)
                if i % 10 == 0 or i == len(remaining):
                    elapsed = time.perf_counter() - start
                    eta = elapsed / i * (len(remaining) - i)
                    print(f"  [{i}/{len(remaining)}] configs done "
                          f"(ETA {cli.fmt_duration(eta)})")
    else:
        for i, task in enumerate(remaining, 1):
            _append_rows(TUNING_RUNS_CSV, ROW_FIELDS, run_task(task))
            if i % 10 == 0 or i == len(remaining):
                elapsed = time.perf_counter() - start
                eta = elapsed / i * (len(remaining) - i)
                print(f"  [{i}/{len(remaining)}] configs done "
                      f"(ETA {cli.fmt_duration(eta)})")
    print(f"Done — {TUNING_RUNS_CSV}")


if __name__ == "__main__":
    main()
