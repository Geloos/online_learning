"""PHASE 3: cache HT 0/1 errors for the 18 tuning streams (resumable)."""

import argparse

import _common


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=_common.default_workers())
    ap.add_argument("--yes", action="store_true",
                    help="skip the time-estimate confirmation")
    args = ap.parse_args()

    from drift_ensemble import config
    from drift_ensemble.streams import tuning_stream_specs

    print(f"Cache dir: {config.CACHE_DIR}")
    _common.cache_streams(tuning_stream_specs(), args.workers, args.yes,
                          "tuning streams")
    print("Done.")


if __name__ == "__main__":
    main()
