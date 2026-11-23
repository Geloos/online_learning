"""Cache HT errors for the 60 test streams, then run every method on every
test stream (resumable; one JVM per worker)."""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=cli.default_workers())
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset of methods to run")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.runio import (
        is_run_done,
        load_cluster_params,
        load_selected_params,
    )
    from drift_ensemble.runio import run_and_save
    from drift_ensemble.config import test_stream_specs
    from streams import stream_id

    load_selected_params()  # fail fast if phase 1 (02/03) not run
    params = load_cluster_params()  # ... or phase 2 (03b)
    print(f"Phase-1 params (singles, ens_raw_resetall): "
          f"{config.TUNING_DIR / 'selected_params.json'}")
    print(f"Phase-2 params ({', '.join(config.PHASE2_METHODS)}): "
          f"{config.CLUSTER_PARAMS_PATH}, min_instances = "
          f"{params['min_instances']}")

    specs = test_stream_specs()
    print(f"Cache dir: {config.CACHE_DIR}")
    cli.cache_streams(specs, args.workers, args.yes, "test streams")

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

    # Time one run, then confirm the estimate for the rest.
    t0 = time.perf_counter()
    run_and_save(units[0])
    unit = time.perf_counter() - t0
    print(f"  timed {units[0][3]} on {stream_id(*units[0][:3])}: "
          f"{cli.fmt_duration(unit)}")
    remaining = units[1:]
    if not remaining:
        print("Done.")
        return
    workers = max(1, args.workers)
    cli.gate(unit, len(remaining), workers, args.yes, "run")

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
                      f"({cli.fmt_duration(rt)}; "
                      f"ETA {cli.fmt_duration(eta)})")
    else:
        for i, unit in enumerate(remaining, 1):
            sid, method, rt = run_and_save(unit)
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(remaining) - i)
            print(f"  [{i}/{len(remaining)}] {sid},{method} "
                  f"({cli.fmt_duration(rt)}; "
                  f"ETA {cli.fmt_duration(eta)})")
    print(f"Done — results under {config.TEST_DIR}")


if __name__ == "__main__":
    main()
