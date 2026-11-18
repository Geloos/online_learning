"""Build the extended-DDD tables and plots from the finished run folders.

    .venv\\Scripts\\python.exe ensemble_experiment/analyze.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from ensemble_experiment import config  # noqa: E402
from ensemble_experiment.analysis import run_analysis  # noqa: E402


def main() -> None:
    summary = run_analysis()
    if summary.empty:
        return
    cols = ["mode", "method", "n_runs", "acc", "f1", "recall", "precision",
            "mean_delay", "frac_after", "kind_acc", "recall_hit_rate",
            "recall_hit_rate_older"]
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    print(f"\nTables and plots written to {config.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
