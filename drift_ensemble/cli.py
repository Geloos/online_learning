"""Helpers shared by the command-line scripts of both experiments: the
time-one-unit -> estimate -> confirm gate, duration formatting, the default
worker count and parallel stream caching. Importing it also puts the repo
root (home of the shared streams.py) on sys.path."""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) // 2)


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
    """The spec's rule: time one unit of work, print the estimated total,
    stop for confirmation (unless --yes)."""
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


def cache_streams(specs, workers: int, yes: bool, label: str) -> None:
    """Resumable, optionally parallel HT-error caching for a stream set,
    with the timing gate. One JVM per spawned worker."""
    from drift_ensemble.cache import cache_stream_spec, is_cached
    from streams import stream_id

    pending = [s for s in specs if not is_cached(stream_id(*s))]
    print(f"{label}: {len(specs)} streams, {len(pending)} to cache.")
    if not pending:
        return

    t0 = time.perf_counter()
    cache_stream_spec(pending[0])
    unit = time.perf_counter() - t0
    print(f"  cached {stream_id(*pending[0])}")
    remaining = pending[1:]
    if not remaining:
        return
    gate(unit, len(remaining), workers, yes, "stream")

    start = time.perf_counter()
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, sid in enumerate(
                    pool.imap_unordered(cache_stream_spec, remaining), 1):
                elapsed = time.perf_counter() - start
                eta = elapsed / i * (len(remaining) - i)
                print(f"  [{i}/{len(remaining)}] {sid}  "
                      f"(ETA {fmt_duration(eta)})")
    else:
        for i, spec in enumerate(remaining, 1):
            sid = cache_stream_spec(spec)
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(remaining) - i)
            print(f"  [{i}/{len(remaining)}] {sid}  "
                  f"(ETA {fmt_duration(eta)})")
