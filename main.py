"""Run both experiments end to end and produce every result.

  1-5  drift_ensemble: cache tuning streams -> tune -> select params ->
       test runs -> analysis        (results/results_drift_ensemble/)
  6-7  ensemble_experiment: DDD runs -> analysis
                                    (results/results_ensemble/)

Every step is resumable: finished work is skipped, so re-running main.py
only does what is missing. Without --yes, each long step prints a time
estimate and asks for confirmation.

    python main.py [--workers N] [--yes] [--only EXPERIMENT] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIFT = ROOT / "drift_ensemble" / "scripts"
DDD = ROOT / "ensemble_experiment"
EXPERIMENTS = ("drift_ensemble", "ensemble_experiment")


def steps(workers: int, yes: bool, only: str | None) -> list[tuple]:
    """(experiment, label, script, args) in execution order."""
    par = ["--workers", str(workers)] + (["--yes"] if yes else [])
    all_steps = [
        ("drift_ensemble", "cache the tuning streams",
         DRIFT / "01_cache_tuning_streams.py", par),
        ("drift_ensemble", "tune the detectors",
         DRIFT / "02_tune.py", par),
        ("drift_ensemble", "select detector params",
         DRIFT / "03_select_params.py", []),
        ("drift_ensemble", "run the test streams",
         DRIFT / "04_run_test.py", par),
        ("drift_ensemble", "analyze",
         DRIFT / "05_analyze.py", []),
        ("ensemble_experiment", "run extended DDD",
         DDD / "run_experiment.py", par),
        ("ensemble_experiment", "analyze",
         DDD / "analyze.py", []),
    ]
    return [s for s in all_steps if only is None or s[0] == only]


def _cluster_reminder() -> None:
    sys.path.insert(0, str(ROOT))
    from drift_ensemble import config

    print(f"\nCluster assignment in drift_ensemble/config.py: "
          f"{config.CLUSTER_MEMBERS}")
    print(f"Check that {config.TUNING_DIR / 'tuning_summary.csv'} still "
          f"justifies it (each detector joins the mode where its tuned F1 "
          f"is higher).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the drift-ensemble experiment, then the extended-"
                    "DDD experiment, and produce all tables and plots.")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel worker processes, one JVM each "
                         "(default 6)")
    ap.add_argument("--yes", action="store_true",
                    help="skip every time-estimate confirmation")
    ap.add_argument("--only", choices=EXPERIMENTS,
                    help="run only one of the two experiments")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    plan = steps(args.workers, args.yes, args.only)
    for i, (exp, label, script, extra) in enumerate(plan, 1):
        cmd = [sys.executable, str(script), *extra]
        rel = script.relative_to(ROOT).as_posix()
        print(f"\n=== [{i}/{len(plan)}] {exp}: {label} "
              f"({' '.join([rel, *extra])})", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT)
        if script.name == "03_select_params.py":
            _cluster_reminder()
    print("\nDone." if not args.dry_run else "")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nStep failed (exit code {e.returncode}): "
                 f"{' '.join(map(str, e.cmd))}")
