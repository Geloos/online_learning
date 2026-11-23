"""Run extended DDD on real datasets (config.REAL_DATASETS), then write
their statistics (real.py). Resumable: runs whose meta.json exists are
skipped. Times one run, prints an estimate and asks for confirmation
(--yes skips the prompt).

    .venv\\Scripts\\python.exe ensemble_experiment/run_real.py --workers 4
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
from ensemble_experiment import real  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+",
                    choices=list(config.REAL_DATASETS))
    ap.add_argument("--methods", nargs="+", choices=real.METHODS)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    specs = real.run_specs(args.datasets, args.methods)
    for d in {s[0] for s in specs}:
        real.dataset_path(d)  # download once, before the workers start
    pending = [s for s in specs if not real.is_done(*s)]
    print(f"{len(specs)} runs, {len(pending)} to do.")

    if pending:
        t0 = time.perf_counter()
        real.run_spec(pending[0])
        unit = time.perf_counter() - t0
        print(f"  done {real.run_name(*pending[0])}")
        remaining = pending[1:]
        if remaining:
            gate(unit, len(remaining), args.workers, args.yes, "run")
            start = time.perf_counter()
            if args.workers > 1:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(args.workers) as pool:
                    it = pool.imap_unordered(real.run_spec, remaining)
                    for i, name in enumerate(it, 1):
                        eta = ((time.perf_counter() - start) / i
                               * (len(remaining) - i))
                        print(f"  [{i}/{len(remaining)}] {name}  "
                              f"(ETA {fmt_duration(eta)})", flush=True)
            else:
                for i, spec in enumerate(remaining, 1):
                    name = real.run_spec(spec)
                    eta = ((time.perf_counter() - start) / i
                           * (len(remaining) - i))
                    print(f"  [{i}/{len(remaining)}] {name}  "
                          f"(ETA {fmt_duration(eta)})", flush=True)

    table = real.real_statistics()
    if table is not None:
        print(table.to_string(index=False, float_format="{:.4f}".format))
        print(f"\nStatistics written to {config.REAL_STATS_DIR}")


if __name__ == "__main__":
    main()
