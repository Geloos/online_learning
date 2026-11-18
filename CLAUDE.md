# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis ("ptyxiakh") repo with two experiments on synthetic CapyMOA drift streams. Both use one
shared stream generator, the root `streams.py`.

1. **`drift_ensemble/`, the detection study.** A Hoeffding Tree learns prequentially and is
   **never reset**. Four detectors (STEPD, SEED, HDDMWeighted, RDDM) watch its 0/1 error, and
   ensembles combine their alarms. Stream design and tuning grids follow Cerqueira et al.'s
   benchmark, but all code is written from scratch: their repo is unlicensed, so never copy
   from it.
2. **`ensemble_experiment/`, the user's extended DDD** (Minku & Yao 2011 plus a fast/slow drift
   classification, an adaptive per-kind cap on the after-drift mode, and a concept pool with
   recall). Here the learner IS adapted: online-bagging ensembles of Hoeffding trees. DDD's
   drift signal is one of experiment 1's detection methods.

The old NB + McNemar verification approach (`verifier.py`, `ens_verified*`, `ens_cooldown`)
was removed on 2026-11-18. It wrongly rejected about 28% of real alarms, so don't
reintroduce it.

## Commands

Always use the venv's Python from the repo root:

```
.venv\Scripts\python.exe main.py [--workers N] [--yes] [--only drift_ensemble|ensemble_experiment] [--dry-run]
.venv\Scripts\python.exe -m pytest                  # both suites (~15 s)
.venv\Scripts\python.exe -m pytest -m "not jvm"     # unit tests only, no JVM startup
.venv\Scripts\python.exe -m pytest drift_ensemble/tests/test_cluster_vote.py::test_name
```

`main.py` runs `drift_ensemble/scripts/01..05`, then `ensemble_experiment/run_experiment.py`
and `analyze.py`. It runs each step as a subprocess and stops at the first failure.
- Every step is resumable: finished work is skipped.
- Long steps time one unit, print an estimate and wait for confirmation. `--yes` skips the
  prompt; it is required when running non-interactively.
- Each worker spawns its own JVM. On this machine (12 logical cores, 16 GB RAM) 6 workers is
  the sweet spot.
- capymoa lets every JVM grow to 8 GB by default. A 6-worker DDD run was killed for low
  memory here, so set `CAPYMOA_JVM_ARGS="-Xmx1g -Xss10M"`.

After `03_select_params.py`, `results/results_drift_ensemble/tuning/tuning_summary.csv` must
justify `CLUSTER_MEMBERS` in `drift_ensemble/config.py`. The current evidence-based split is
SEED/RDDM = ABRUPT and STEPD/HDDMWeighted = GRADUAL. `main.py` prints a reminder after step 3
but does not stop.

## Critical: import order and sys.path

CapyMOA needs `JAVA_HOME`, which is unset on this machine. `drift_ensemble/_java.py` derives
it from `java -XshowSettings:properties` when `drift_ensemble.config` is imported. **Every
module that imports capymoa (directly or indirectly) must import `drift_ensemble.config`
first** (`ensemble_experiment.config` does so). `streams.stream_from_plan` calls
`ensure_java_home()` itself.

`drift_ensemble.config` also puts the repo root on `sys.path`, so `import streams` works.
The scripts and test conftests bootstrap the root too.

## Shared: `streams.py` (root)

- **Contents:** generator definitions (`FUNCTIONS`, `INIT_FUNCTION`) and `plan_stream`, which
  is pure Python and covers ABRUPT, GRADUAL, MIXED and RECURRING. Also `stream_from_plan` /
  `make_stream`, `concept_ids` and `mark_recurring`.
- **Sizes:** stream sizes are required keyword arguments. Each experiment passes
  `config.STREAM_SIZES`.
- **Reproducibility:** the RNG consumption order is documented in the module and must not
  change. The 2026-11-18 merge was verified to reproduce every old plan and every cached
  `ground_truth.json` exactly.

## Experiment 1: `drift_ensemble/`

`config.py` holds every hyperparameter (sizes, seeds, grids, clusters, `VOTE`, `SENS`,
method roster). Tuning and test seeds are asserted disjoint there. Results go under
`results/results_drift_ensemble/`.

- **`cache.py`:** runs one HT pass per stream and stores `errors.npy`, `ht_accuracy.npy` and
  `ground_truth.json` under `cache/`. HT's error sequence is deterministic given the stream,
  so **everything downstream replays these cached errors**. `tests/test_cache_replay.py`
  proves equivalence with a live run.
- **Tuning:** `detectors.py` (factory, grids, `set_threshold`) → `tuning.py` (grid × cached
  errors) → `selection.py` (best config per detector × mode, plus a LOGO check).
- **`ensemble.py`:** `RawResetAllEnsemble`, `ClusterVoteEnsemble` and
  `ClusterSensitivityEnsemble`. `workflow.run_method` runs the 8 roster methods
  (`config.METHODS`):
  - `single_*` ×4.
  - `ens_raw`: OR; each detector only self-resets.
  - `ens_raw_resetall`: OR; every alarm resets ALL detectors.
  - `ens_cluster_vote`: two same-cluster votes within `VOTE["WINDOW"]` instances, or one
    full-weight vote, signal drift and reset all detectors. The earlier voter gains weight
    and the later loses it; weights are clamped to [0.5, 1.0].
  - `ens_cluster_sens`: same pair rule and reset-all. The later voter's threshold becomes
    more sensitive, scaled by its lag. Levels only rise, capped at `L_MAX`, and are applied
    mid-stream via `detectors.set_threshold`.
  - `workflow.build_method_ensemble` builds the three ensemble classes. Experiment 2 reuses
    it.
- **Outputs:** `runio.py` defines the run-folder format (alarms.json, weights.csv,
  sensitivity.csv, raw_fires.csv, meta.json). `analysis.py` reads *only* result folders and
  must never recompute from streams.
- **Scripts:** `scripts/01..05` are thin CLIs; shared CLI helpers live in `cli.py`.
- **Tests:** in `tests/`, using `StubDetector` from `tests/fakes.py`. JVM tests carry the `jvm`
  marker.

## Experiment 2: `ensemble_experiment/`

All parameters are in `config.py`:
- streams: 8 drifts every 5,000; 4 modes; RECURRING cycles 3 concepts
- seeds: 201–210
- `DDD` dict
- 7 methods: `ens_raw` is omitted because DDD resets the whole signal on every alarm, which
  makes it identical to `ens_raw_resetall`.

The full grid is 840 runs; each takes about 17 s. Results go under
`results/results_ensemble/runs/` (`meta.json` written last means done).

- **`bagging.py`:** online bagging that drives the Java `moa_learner` of capymoa
  HoeffdingTrees (NB-adaptive leaves).
- **`ddd.py`:** the state machine. Ensembles and signal are injected, so the tests use fakes.
- **`detection.py`:** wraps experiment 1's methods as signals.
- **`logger.py`:** matches DDD's behaviour against the ground truth: TP/FP, pool concept tags
  and recall hits.
- **Live runs:** runs are live. No error cache is possible because DDD's errors depend on its
  own alarms.
- **Choices beyond the user's spec:**
  - The signal is reset on every detection, and on any exit whose winner isn't new-low.
  - Recall may pick the entry frozen at the same detection; this is tracked as
    `recalled_just_added` and `recall_hit_rate_older`.

## capymoa 0.9.0 quirks (verified by probes, do not "fix")

- All four detectors restart their own statistics after `detected_change()`. They fire again
  without an explicit `reset()` (`DETECTOR_SELF_RESETS` in config.py).
- `reset()` only clears the Python side. `ensemble.reset_detector` also calls MOA's
  `resetLearning()`.
- `GradualDrift(position=...)` means the drift **center**, so we pass `start`/`end` instead. Its
  constructor prints a stray "None", which is suppressed in `streams.stream_from_plan`.
- capymoa 0.9.0 imports torch unconditionally; the CPU build satisfies it.
- Changing a detector's MOA Option mid-stream does nothing, because thresholds are cached in
  private Java fields:
  - STEPD/RDDM read options only in `prepareForUse`.
  - HDDM_W/SEED re-read them on `resetLearning`.
  - `detectors.set_threshold` sets Option **and** field, which survives every reset path
    (`tests/test_set_threshold.py`).
- DDD frozen copies must use MOA `copy()`; Python `deepcopy` doesn't duplicate the Java tree.
- Poisson bagging trains k separate times. A single weight-k call is NOT equivalent for
  NB-adaptive trees (93–100% prediction agreement).
