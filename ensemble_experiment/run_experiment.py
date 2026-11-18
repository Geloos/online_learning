"""Run the extended-DDD experiment: every (generator, mode, seed, method).

Resumable: runs whose meta.json exists are skipped. Times one run, prints
an estimate and asks for confirmation (--yes skips the prompt). Each
worker is a spawned process with its own JVM.

    .venv\\Scripts\\python.exe ensemble_experiment/run_experiment.py --workers 6
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift_ensemble.cli import default_workers, fmt_duration, gate  # noqa: E402
from ensemble_experiment import config  # noqa: E402
from ensemble_experiment.runner import (  # noqa: E402
    is_done,
    run_name,
    run_spec,
    run_specs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generators", nargs="+", choices=config.GENERATORS)
    ap.add_argument("--modes", nargs="+", choices=config.MODES)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--methods", nargs="+", choices=config.METHODS)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    specs = run_specs(args.generators, args.modes, args.seeds, args.methods)
    pending = [s for s in specs if not is_done(*s)]
    print(f"{len(specs)} runs, {len(pending)} to do.")
    if not pending:
        return

    t0 = time.perf_counter()
    run_spec(pending[0])
    unit = time.perf_counter() - t0
    print(f"  done {run_name(*pending[0])}")
    remaining = pending[1:]
    if not remaining:
        return
    gate(unit, len(remaining), args.workers, args.yes, "run")

    start = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for i, name in enumerate(
                    pool.imap_unordered(run_spec, remaining), 1):
                elapsed = time.perf_counter() - start
                eta = elapsed / i * (len(remaining) - i)
                print(f"  [{i}/{len(remaining)}] {name}  "
                      f"(ETA {fmt_duration(eta)})", flush=True)
    else:
        for i, spec in enumerate(remaining, 1):
            name = run_spec(spec)
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(remaining) - i)
            print(f"  [{i}/{len(remaining)}] {name}  "
                  f"(ETA {fmt_duration(eta)})", flush=True)


if __name__ == "__main__":
    main()
