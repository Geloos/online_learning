"""PHASE 9 step 1: cache HT errors for the 45 test streams, then run every
method on every test stream (resumable; one JVM per worker)."""

import argparse
import time

import _common

VERIFIED = ("ens_verified", "ens_verified_resetall")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=_common.default_workers())
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset of methods to run")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.runio import is_run_done, load_selected_params
    from drift_ensemble.runio import run_and_save
    from drift_ensemble.streams import stream_id, test_stream_specs

    load_selected_params()  # fail fast with a clear message if 02/03 not run

    specs = test_stream_specs()
    print(f"Cache dir: {config.CACHE_DIR}")
    _common.cache_streams(specs, args.workers, args.yes, "test streams")

    methods = (args.methods.split(",") if args.methods else config.METHODS)
    unknown = set(methods) - set(config.METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")

    units = [(g, m, s, meth)
             for (g, m, s) in specs
             for meth in methods
             if not is_run_done(stream_id(g, m, s), meth)]
    print(f"\nTest runs: {len(specs)} streams x {len(methods)} methods; "
          f"{len(units)} still to run.")
    if not units:
        print("Nothing to do.")
        return
    config.TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Time one unit of each kind (verified runs iterate the full stream and
    # are far slower than cached replays), then confirm the estimate.
    slow = [u for u in units if u[3] in VERIFIED]
    fast = [u for u in units if u[3] not in VERIFIED]
    unit_fast = unit_slow = 0.0
    if fast:
        t0 = time.perf_counter()
        run_and_save(fast[0])
        unit_fast = time.perf_counter() - t0
        print(f"  timed {fast[0][3]} on {stream_id(*fast[0][:3])}: "
              f"{_common.fmt_duration(unit_fast)}")
        fast = fast[1:]
    if slow:
        t0 = time.perf_counter()
        run_and_save(slow[0])
        unit_slow = time.perf_counter() - t0
        print(f"  timed {slow[0][3]} on {stream_id(*slow[0][:3])}: "
              f"{_common.fmt_duration(unit_slow)}")
        slow = slow[1:]

    remaining = slow + fast  # long runs first keeps the pool busy
    if not remaining:
        print("Done.")
        return
    workers = max(1, args.workers)
    est = (unit_fast * len(fast) + unit_slow * len(slow)) / workers
    print(f"Estimated remaining time: {_common.fmt_duration(est)} "
          f"({len(slow)} verified + {len(fast)} replay runs, "
          f"{workers} worker(s)).")
    _common.gate(max(unit_fast, unit_slow), len(remaining), workers,
                 args.yes, "run (upper bound per unit)")

    start = time.perf_counter()
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, (sid, method, rt) in enumerate(
                    pool.imap_unordered(run_and_save, remaining), 1):
                elapsed = time.perf_counter() - start
                eta = elapsed / i * (len(remaining) - i)
                print(f"  [{i}/{len(remaining)}] {sid},{method} "
                      f"({_common.fmt_duration(rt)}; "
                      f"ETA {_common.fmt_duration(eta)})")
    else:
        for i, unit in enumerate(remaining, 1):
            sid, method, rt = run_and_save(unit)
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(remaining) - i)
            print(f"  [{i}/{len(remaining)}] {sid},{method} "
                  f"({_common.fmt_duration(rt)}; "
                  f"ETA {_common.fmt_duration(eta)})")
    print(f"Done — results under {config.TEST_DIR}")


if __name__ == "__main__":
    main()
