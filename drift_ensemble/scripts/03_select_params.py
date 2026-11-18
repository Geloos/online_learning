"""Pick the best config per detector x mode from tuning_runs.csv.

Writes results/results_drift_ensemble/tuning/selected_params.json,
tuning_summary.csv (the table that justifies the cluster assignment) and
logo_check.csv (leave-one-generator-out robustness check).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from drift_ensemble import cli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default=None,
                    help="override config.TUNE_OBJECTIVE (f1, f2, delay_auc)")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.selection import (
        load_tuning_runs,
        logo_check,
        select_params,
    )

    df = load_tuning_runs()
    print(f"{len(df)} tuning rows "
          f"({df['detector'].nunique()} detectors, "
          f"{df.groupby('detector')['params'].nunique().to_dict()} configs)")

    selected, summary = select_params(df, args.objective)
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.TUNING_DIR / "selected_params.json"
    out_json.write_text(json.dumps(selected, indent=2))
    out_csv = config.TUNING_DIR / "tuning_summary.csv"
    summary.to_csv(out_csv, index=False)

    logo = logo_check(df, args.objective)
    logo.to_csv(config.TUNING_DIR / "logo_check.csv", index=False)

    print(f"\nWrote {out_json}\nWrote {out_csv}\n")
    with_cluster = summary.copy()
    cluster_of = {name: cl for cl, ms in config.CLUSTER_MEMBERS.items()
                  for name in ms}
    with_cluster.insert(1, "cluster", with_cluster["detector"].map(
        cluster_of))
    cols = ["detector", "cluster", "mode", "mean_f1", "mean_recall",
            "mean_precision", "mean_mean_delay", "mean_far"]
    print(with_cluster[cols].to_string(index=False,
                                       float_format=lambda v: f"{v:.3f}"))
    print("\nCheck: each detector should score best on its own cluster's "
          "mode (CLUSTER_MEMBERS in config.py).")


if __name__ == "__main__":
    main()
