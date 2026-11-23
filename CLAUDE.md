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
.venv\Scripts\python.exe main.py STEP [--workers N] [--yes] [step options]   # one step; STEP -h
.venv\Scripts\python.exe -m pytest                  # both suites (~20 s)
.venv\Scripts\python.exe -m pytest -m "not jvm"     # unit tests only, no JVM startup
.venv\Scripts\python.exe -m pytest drift_ensemble/tests/test_detectors.py::test_name
```

`main.py` is the only entry point. Every step is a subcommand: `cache`, `tune`, `select`,
`phase2`, `test`, `stats` (experiment 1), then `ddd`, `ddd-stats` and `real` (experiment 2).
`screen` exists too but isn't part of the full run. Without a step, main.py runs the 9 in
order, each as a subprocess, and stops at the first failure.
- **Layout:** restructured on 2026-11-23 at the user's request, because they found it
  overengineered. It went from 66 to 24 `.py` files: no `scripts/` folders, one module per
  topic, one test file per module. Don't split it back into many small files.
  `main.run_pool` is the one time-one-unit → confirm → worker-pool loop that every step uses.
- The `real` step sets `CAPYMOA_JVM_ARGS="-Xmx3g -Xss10M"` itself and uses at most 4 workers
  (`config.REAL_JVM_ARGS`, `REAL_MAX_WORKERS`).
- Every step is resumable: finished work is skipped.
- Long steps time one unit, print an estimate and wait for confirmation. `--yes` skips the
  prompt; it is required when running non-interactively.
- Each worker spawns its own JVM. On this machine (12 logical cores, 16 GB RAM) 6 workers is
  the sweet spot.
- capymoa lets every JVM grow to 8 GB by default. A 6-worker DDD run was killed for low
  memory here, so set `CAPYMOA_JVM_ARGS="-Xmx1g -Xss10M"`.

After the `select` step, `results/results_drift_ensemble/tuning/tuning_summary.csv` must
justify `CLUSTER_MEMBERS` in `drift_ensemble/config.py`. Since 2026-11-20 (re-tuned with
`MAX_DELAY = 500`) the split is HDDMWeighted/RDDM = ABRUPT and STEPD/SEED = GRADUAL.
- A detector always runs with its cluster's params, so it goes where those params hurt it least
  on the other stream type. The evidence is in config.py's comment and in `screening.csv`.
- "The mode with the higher tuning F1" no longer works as a rule: with the 500 window every
  detector scores higher on GRADUAL.
- The old 1000-window split, its tuning files and its runs were deleted on 2026-11-23 at the
  user's request. Don't bring them back up.
- The `select` step prints a reminder but does not stop. Phase 2 keeps this split.

## Critical: import order and sys.path

CapyMOA needs `JAVA_HOME`, which is unset on this machine. `drift_ensemble/config.py`
(`ensure_java_home`) derives it from `java -XshowSettings:properties` when it is imported. **Every
module that imports capymoa (directly or indirectly) must import `drift_ensemble.config`
first** (`ensemble_experiment.config` does so). `streams.stream_from_plan` calls
`ensure_java_home()` itself.

`drift_ensemble.config` also puts the repo root on `sys.path`, so `import streams` works.
`main.py` and pytest (`pythonpath = .` in `pytest.ini`) put it there too.

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
`STATS`, method roster). Tuning and test seeds are asserted disjoint there. Results go under
`results/results_drift_ensemble/`.

There are 60 test streams: 3 generators × {ABRUPT, GRADUAL, MIXED, RECURRING} × seeds 101–105.
- RECURRING was added on 2026-11-20. It cycles through 3 concepts (`RECURRING_N_CONCEPTS`).
- `STREAM_SIZES["n_concepts"]` is ignored by the other modes. All 63 earlier cached plans
  were verified unchanged.
- Tuning still uses ABRUPT and GRADUAL only.

Modules:
- `config.py`: every setting, plus `ensure_java_home`.
- `detectors.py`: detector factory, grids, `set_threshold`, the phase-2 gate, and the 3
  ensemble classes.
- `tuning.py`: phase 1, phase 2 and screening.
- `runs.py`: HT error cache, which params a method uses, one method on one stream, and run
  folders (save + `load_runs`).
- `stats.py`: scoring (`match_alarms`, `compute_metrics`, `delay_curve`) and the statistics.
- `analysis.py`: tables and plots.

- **Cache (`runs.py`, step `cache`; `test` caches the test streams):** runs one HT pass per
  stream and stores `errors.npy`, `ht_accuracy.npy` and `ground_truth.json` under `cache/`.
  HT's error sequence is deterministic given the stream, so **everything downstream replays
  these cached errors**. `tests/test_runs.py` proves equivalence with a live run.
- **Tuning phase 1 (`tuning.py`, steps `tune` + `select`):** grid × cached errors → best
  config per detector × mode, plus a LOGO check → `selected_params.json`. The singles and
  `ens_raw_resetall` run with it (both experiments), as do screening and the cluster-split
  check.
- **Tuning phase 2** (`tuning.py`, step `phase2`, added 2026-11-23 at the user's request):
  one shared minimum M per cluster, so a cluster's two
  detectors switch back on together after every reset. The user wanted M_A and M_G tuned
  separately; they're equal only if tuning says so.
  - Own minimums (probe, 0s then 1s after a reset): STEPD 2 × window_size, SEED
    2 × block_size, RDDM min_n_instances, HDDMWeighted none. With the phase-1 params they
    ranged from ~230 (HDDM_W) to 2,000 (RDDM, STEPD) and caused the mismatch.
  - `detectors.GatedDetector`: a fire counts only once the detector has seen M instances
    since its last reset or accepted fire. It keeps learning meanwhile. RDDM's
    min_n_instances is set to M. STEPD/SEED may only use configs with own minimum ≤ M.
  - For each M in `MIN_INSTANCES_GRID` (50–2000), the cluster's members are re-tuned on the
    cluster's own mode's tuning streams (phase-1 grids). The score is the mean of the two
    members' best F1, and the best M wins. Singles never get an explicit reset, so gated alarms
    come from one ungated replay (`gate_alarms`, tested equal to a gated run).
  - Output: `phase2_runs.csv`, `phase2_summary.csv` and `cluster_params.json`
    (`{"min_instances": {cluster: M}, cluster: {detector: params}}`).
  - **Only the two-vote methods use phase 2** (`config.PHASE2_METHODS`: `ens_cluster_vote`,
    `ens_cluster_sens`; the user's choice). The singles and `ens_raw_resetall` use phase 1
    with no minimum. This holds in both experiments, so the real data (`ens_raw_resetall`)
    uses phase 1. `runs.method_params` picks the file, and `build_detector` gates when the
    dict has `min_instances`. `meta.json` stores the params used.
- **Ensembles (`detectors.py`):** `RawResetAllEnsemble`, `ClusterVoteEnsemble` and
  `ClusterSensitivityEnsemble`. `runs.run_method` runs the 7 roster methods
  (`config.METHODS`), the same 7 as experiment 2:
  - `single_*` ×4.
  - `ens_raw_resetall`: OR; every alarm resets ALL detectors.
  - (`ens_raw`, OR with self-reset only, was dropped on 2026-11-23 at the user's request
    because DDD can't use it. Don't reintroduce it.)
  - `ens_cluster_vote`: two same-cluster votes within `VOTE["WINDOW"]` instances, or one
    full-weight vote, signal drift and reset all detectors. The earlier voter gains weight
    and the later loses it; weights are clamped to [0.5, 1.0].
  - `ens_cluster_sens`: same pair rule and reset-all. The later voter's threshold becomes
    more sensitive, scaled by its lag. Levels only rise, capped at `L_MAX`, and are applied
    mid-stream via `detectors.set_threshold`.
  - `runs.build_method_ensemble` builds the three ensemble classes. Experiment 2 reuses it.
- **Outputs:** `runs.py` defines the run-folder format (alarms.json, weights.csv,
  sensitivity.csv, raw_fires.csv, meta.json). `analysis.py` and `stats.py` read *only*
  result folders and must never recompute from streams.
- **Statistics:** `stats.py` is called by the `stats` step after `analysis.py`. It covers all 7
  methods and writes to `statistics/drift_ensemble/`:
  - `runs.csv` and `delays.csv`, which has one row per detected drift;
  - `metrics.csv`: per method × mode, the **mean and std** over runs of precision, recall
    and F1 (`*_mean`, `*_std`), and the **median** delay over the detected drifts
    (`delay_median`). The user chose this on 2026-11-23. The 15 runs are 3 generators ×
    5 seeds, so a median hides a failure on a whole generator (GRADUAL HDDMWeighted: F1
    0.16 on SEA, median ranked it 1st, mean 5th). Delays are skewed, so they use the
    median.
  - `delay_curve.csv`, plus `<MODE>/delay_curve.png`: means when every run is rescored
    for each max delay in `STATS["delay_grid"]` (0–500, step 25). The plot shows recall
    only (user's choice). The max delay counts from the drift END, so gradual curves start
    above 0 at max delay 0: alarms during the 2,000-instance transition count. The user
    confirmed this is correct.
  - one folder per mode with `singles.png` (the 4 detectors) and `ensembles.png` (the 3
    ensembles). They are bar charts of `metrics.csv`: per method, mean precision, recall
    and F1 bars with std error bars in the top row (0–1), and the median delay bar in the
    bottom row (in instances, since it's a different unit). Both charts of a mode share
    the delay scale.
  - `analysis/{tables,plots}` from `analysis.py`: detection_metrics (mean ± std, FAR,
    LaTeX), per_run_metrics, per_type_mixed, first_to_fire, diversity, significance,
    sensitivity_summary, runtime, plus weight, sensitivity and tuning-heatmap plots. Since
    2026-11-23 they're in `statistics/`, like experiment 2, because `results/` holds only
    run outcomes (cache, tuning, test runs, screening rows). analysis.py's own delay curves
    and `detection_medians.csv` were dropped because they duplicated `stats.py`.
  - The file is named `stats.py` rather than `statistics.py` so it doesn't shadow the
    stdlib module.
- **Detector screening** (`tuning.py`, step `screen`, added 2026-11-20, not part of the
  full run): it tunes DDM, HDDMAverage and ADWIN
  (`config.SCREENING_GRIDS`, our own ranges) as single detectors on the cached errors, then
  score them on the ABRUPT and GRADUAL test streams.
  - The study's 4 are scored the same way, each with its params for the stream's mode.
  - Raw rows go to `results/results_drift_ensemble/screening/`; the summary goes to
    `statistics/drift_ensemble/screening.csv`.
  - The roster, the clusters and the ensembles are unchanged.
- **Tests:** one file per module in `tests/` (`test_detectors`, `test_tuning`, `test_runs`,
  `test_stats`). `StubDetector` is defined in `test_detectors.py`. JVM tests carry the `jvm`
  marker.

## Experiment 2: `ensemble_experiment/`

All parameters are in `config.py`:
- streams: 8 drifts every 5,000; 4 modes; RECURRING cycles 3 concepts
- seeds: 201–210
- `DDD` dict
- 7 methods, the same as experiment 1. There is no `ens_raw`: DDD resets the whole signal
  on every alarm, which would make it identical to `ens_raw_resetall`.

The full grid is 840 runs; each takes about 17 s. Results go under
`results/results_ensemble/runs/` (`meta.json` written last means done).

- **`models.py`:** online bagging that drives the Java `moa_learner` of capymoa
  HoeffdingTrees (NB-adaptive leaves), and the drift signals: experiment 1's methods with
  the same params as there (phase 1 or phase 2 per method, `load_method_params`).
- **`ddd.py`:** the state machine, plus its parts (accuracy trackers, adaptive cap, concept
  pool). Ensembles and signal are injected, so the tests use fakes (`tests/test_ddd.py`).
- **`runs.py`:** one run → one folder, synthetic (step `ddd`) and real (step `real`).
  `RunLogger` matches DDD's behaviour against the ground truth: TP/FP, pool concept tags
  and recall hits.
- **`analysis.py`** (steps `ddd-stats`, `real`): reads run folders only (`correct.npy` holds every instance, so no
  re-runs). Writes statistics for the best method only (`config.STATS["method"]`,
  `ens_raw_resetall`) to `statistics/ensemble_experiment/`, outside `results/`, which
  holds only run outcomes. Per mode it writes `recovery.csv` and `drifts.csv` and plots
  the median accuracy per 200-instance block, one dot per block (`STATS["acc_block"]`):
  - ABRUPT and GRADUAL plots use the whole stream.
  - MIXED and RECURRING plots are aligned on each drift, with one line for abrupt and one
    for gradual drifts, because their drift positions differ per seed.

  Recovery (`config.RECOVERY`) uses only real drifts and the TP alarm that detected each one.
  It reports medians of:
  - accuracy over the stable period before the drift (from the previous drift's end, or
    from the end of warm-up);
  - accuracy from the drift to the detection;
  - instances after the detection until the accuracy counted from the detection reaches
    ≥ 95% of the before accuracy. It's only checked after 50 instances.
- **Live runs:** runs are live. No error cache is possible because DDD's errors depend on its
  own alarms.
- **Real data (step `real`, the full run's last step; runs in `runs.py`, statistics in
  `analysis.py`; added 2026-11-23):** extended DDD on Electricity, Airlines, Covtype and
  Poker (`config.REAL_DATASETS`). The step sets `CAPYMOA_JVM_ARGS="-Xmx3g -Xss10M"` itself,
  since the datasets have 540k–830k instances.
  - The user's choices: only `ens_raw_resetall`, ONE run (seed 201, `REAL_SEED`), no
    more seeds.
  - There are no drift positions, so only prequential accuracy (after warm-up), the
    number of detections and accuracy over time are reported.
  - A `no_change` baseline (predict the previous label) is included because Electricity
    is strongly autocorrelated.
  - The data comes from OpenML: 151 Electricity (= MOA's elecNormNew, same as capymoa's),
    1169 Airlines (MOA's), 150 Covtype (MOA's covtypeNorm, original order) and 155 Poker
    (MOA's poker-lsn, 829,201 instances, 10 classes). It's md5-checked and stored in
    `data/` (gitignored). `capymoa.datasets` can't be imported (it
    needs torchvision), and its Dropbox links are blocked on this network.
  - Runs go to `results/results_ensemble/real/`, statistics to
    `statistics/ensemble_experiment/real/<dataset>/`.
- **Choices beyond the user's spec:**
  - The signal is reset on every detection, and on any exit whose winner isn't new-low.
  - Recall may pick the entry frozen at the same detection; this is tracked as
    `recalled_just_added` and `recall_hit_rate_older`.

## capymoa 0.9.0 quirks (verified by probes, do not "fix")

- All four detectors restart their own statistics after `detected_change()`. They fire again
  without an explicit `reset()` (`DETECTOR_SELF_RESETS` in config.py).
- `reset()` only clears the Python side. `detectors.reset_detector` also calls MOA's
  `resetLearning()`.
- `GradualDrift(position=...)` means the drift **center**, so we pass `start`/`end` instead. Its
  constructor prints a stray "None", which is suppressed in `streams.stream_from_plan`.
- capymoa 0.9.0 imports torch unconditionally; the CPU build satisfies it.
- Changing a detector's MOA Option mid-stream does nothing, because thresholds are cached in
  private Java fields:
  - STEPD/RDDM read options only in `prepareForUse`.
  - HDDM_W/SEED re-read them on `resetLearning`.
  - `detectors.set_threshold` sets Option **and** field, which survives every reset path
    (`tests/test_detectors.py`).
- DDD frozen copies must use MOA `copy()`; Python `deepcopy` doesn't duplicate the Java tree.
- Poisson bagging trains k separate times. A single weight-k call is NOT equivalent for
  NB-adaptive trees (93–100% prediction agreement).
