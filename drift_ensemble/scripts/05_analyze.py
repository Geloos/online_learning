"""Build every analysis table and plot from the finished run folders
(never reruns experiments)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.analysis import analyze

    print(f"Analyzing runs under {config.TEST_DIR} ...")
    analyze(make_plots=not args.no_plots)
    print(f"\nTables: {config.ANALYSIS_DIR / 'tables'}")
    if not args.no_plots:
        print(f"Plots:  {config.ANALYSIS_DIR / 'plots'}")


if __name__ == "__main__":
    main()
