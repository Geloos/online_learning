"""Screen extra capymoa detectors (DDM, HDDMAverage, ADWIN) against the
study's 4 on ABRUPT and GRADUAL F1, as single detectors on the cached HT
errors (no new streams; roster and ensembles untouched). Resumable.

    .venv\\Scripts\\python.exe drift_ensemble/scripts/screen_detectors.py --workers 6 --yes
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=cli.default_workers())
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from drift_ensemble import config, screening
    from drift_ensemble.cache import is_cached
    from drift_ensemble.runio import load_selected_params
    from drift_ensemble.tuning import tuning_stream_ids
    from streams import stream_id

    study = load_selected_params()
    tune_ids = tuning_stream_ids()
    test_ids = [stream_id(*s) for s in config.test_stream_specs()
                if s[1] in config.PURE_MODES]
    missing = [s for s in tune_ids + test_ids if not is_cached(s)]
    if missing:
        raise SystemExit(f"{len(missing)} streams are not cached (e.g. "
                         f"{missing[0]}); run scripts 01 and 04 first")

    screening.run_stage(screening.tuning_tasks(study), tune_ids,
                        screening.TUNING_CSV, args.workers, args.yes,
                        "Tuning streams")
    tuning_df = screening.load_rows(screening.TUNING_CSV)
    selected = screening.all_selected(tuning_df, study)
    (config.SCREENING_DIR / "selected_params.json").write_text(
        screening.selected_json(selected))

    screening.run_stage(screening.test_tasks(selected), test_ids,
                        screening.TEST_CSV, args.workers, args.yes,
                        "Test streams (ABRUPT, GRADUAL)")
    summary = screening.summarize(tuning_df,
                                  screening.load_rows(screening.TEST_CSV),
                                  selected)
    config.STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(screening.SUMMARY_CSV, index=False)

    with pd.option_context("display.width", 220,
                           "display.max_columns", 20,
                           "display.max_colwidth", 70,
                           "display.float_format", "{:.3f}".format):
        print("\nEach detector with the config tuned for the mode "
              "(tuning_f1 = mean over 9 tuning streams; test_* = medians "
              "over 15 test streams)")
        print(summary.drop(columns=["test_f1_mean"]).to_string(index=False))
    print(f"\nWritten to {screening.SUMMARY_CSV}")


if __name__ == "__main__":
    main()
