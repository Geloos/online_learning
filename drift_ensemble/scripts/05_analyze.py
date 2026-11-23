"""Compute every statistic from the finished run folders (never reruns
experiments) into statistics/drift_ensemble/: the analysis tables and plots
(analysis/), then the detection statistics of every method (precision,
recall, F1, delay: medians, delay curves, bar charts per mode)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from drift_ensemble import config
    from drift_ensemble.analysis import analyze
    from drift_ensemble.stats import run_statistics

    print(f"Analyzing runs under {config.TEST_DIR} ...")
    analyze(make_plots=not args.no_plots)
    print(f"\nTables: {config.ANALYSIS_DIR / 'tables'}")
    if not args.no_plots:
        print(f"Plots:  {config.ANALYSIS_DIR / 'plots'}")

    summary = run_statistics()
    if summary.empty:
        return
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(f"\nPer method x mode (max delay {config.MAX_DELAY}): mean and "
              f"std of P/R/F1 over runs; delay = median over detected "
              f"drifts")
        print(summary.to_string(index=False))
    print(f"\nStatistics written to {config.STATISTICS_DIR}")


if __name__ == "__main__":
    main()
