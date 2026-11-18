"""Saving/loading of test runs.

One run = one method on one test stream, stored under
results/results_drift_ensemble/test/<generator>,<mode>,<seed>,<method>/ with
alarms.json, weights.csv, sensitivity.csv, raw_fires.csv and meta.json
(written last: its presence marks the run as complete).
"""

from __future__ import annotations

import csv
import json

from drift_ensemble import config
from streams import stream_id
from drift_ensemble.workflow import run_method

WEIGHT_FIELDS = ["t", "detector", "w"]
SENS_FIELDS = ["t", "detector", "level", "threshold", "reason"]


def run_dir(sid: str, method: str):
    return config.TEST_DIR / f"{sid},{method}"


def is_run_done(sid: str, method: str) -> bool:
    return (run_dir(sid, method) / "meta.json").exists()


def _write_csv(path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_run(generator: str, mode: str, seed: int, method: str,
             result: dict, selected_params: dict) -> None:
    sid = stream_id(generator, mode, seed)
    out = run_dir(sid, method)
    out.mkdir(parents=True, exist_ok=True)

    (out / "alarms.json").write_text(
        json.dumps({"alarms": [int(t) for t in result["alarms"]]}))

    _write_csv(out / "weights.csv", WEIGHT_FIELDS,
               result.get("weights_log", []))
    _write_csv(out / "sensitivity.csv", SENS_FIELDS,
               result.get("sens_log", []))

    raw = result.get("raw_fires", [])
    raw_fields = ["t", "detector"]
    _write_csv(out / "raw_fires.csv", raw_fields,
               [{f: r.get(f, "") for f in raw_fields} for r in raw])

    meta = {
        "method": method,
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "sid": sid,
        "length": result["length"],
        "runtime_s": result["runtime_s"],
        "n_alarms": len(result["alarms"]),
        "n_raw_fires": len(raw),
        "final_weights": result.get("final_weights"),
        "final_levels": result.get("final_levels"),
        "ground_truth": result["ground_truth"],
        "selected_params": selected_params,
        "config": config.snapshot(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_selected_params() -> dict:
    path = config.TUNING_DIR / "selected_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run drift_ensemble/scripts/02_tune.py "
            f"and 03_select_params.py first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Pool worker (module-level so it pickles under spawn)
# ---------------------------------------------------------------------------
_SELECTED: dict | None = None


def run_and_save(task: tuple[str, str, int, str]) -> tuple[str, str, float]:
    """Run one (generator, mode, seed, method) unit and persist it.
    Returns (sid, method, runtime_s). Safe to call in pool workers."""
    global _SELECTED
    if _SELECTED is None:
        _SELECTED = load_selected_params()
    generator, mode, seed, method = task
    result = run_method(method, generator, mode, seed, _SELECTED)
    save_run(generator, mode, seed, method, result, _SELECTED)
    return stream_id(generator, mode, seed), method, result["runtime_s"]
