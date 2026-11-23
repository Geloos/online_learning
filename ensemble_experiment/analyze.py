"""Build the extended-DDD statistics (best method) from the finished runs.

    .venv\\Scripts\\python.exe ensemble_experiment/analyze.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from ensemble_experiment import config  # noqa: E402
from ensemble_experiment.analysis import run_analysis  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    recovery = run_analysis()
    if recovery.empty:
        return
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(f"{config.STATS['method']}: recovery after detected real "
              f"drifts (medians)")
        print(recovery.to_string(index=False))
    print(f"\nStatistics written to {config.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
