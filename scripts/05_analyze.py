"""PHASE 9 step 2: build every analysis table and plot from the finished
run folders (never reruns experiments)."""

import argparse

import _common  # noqa: F401  (sys.path bootstrap)


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
