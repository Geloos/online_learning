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
`test`, `stats` (experiment 1), then `ddd`, `ddd-stats` and `real` (experiment 2).
`screen` exists too but isn't part of the full run. Without a step, main.py runs the 8 in
order, each as a subprocess, and stops at the first failure.
- **Layout:** restructured on 2026-11-23 at the user's request, because they found it
  overengineered. It went from 66 to 23 `.py` files: no `scripts/` folders, one module per
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

**TF2 and the clusters (2026-11-24, the user's design).** F1 is gone everywhere, replaced by
**TF2 (timely F2)**, defined in `stats.py`:
- NDT_k = (t_detect − start_k) / (end_k − start_k + MAX_DELAY) per detected drift. It is
  normalized by the drift's whole TP window, so gradual drifts stay in [0, 1].
- Timely recall R_t = recall × (1 − mean NDT).
- TF2 = F2(precision, R_t) = 5·P·R_t / (4·P + R_t).
- The user asked for a recall- and speed-weighted score because DDD benefits from recall. Pure
  recall (or `delay_auc`) can't select anything: a detector that fires all the time maxes
  it. The earlier F2 and `delay_auc` columns and `select --objective` were removed with F1.

The `select` step picks each detector's config per mode by TF2, then the **clusters**: the
`CLUSTER_SIZE` = 2 detectors with the best mean tuning TF2 per mode.
- They go to `selected_params.json` under `"clusters"`.
- A detector may be in BOTH clusters (two members, two param sets). Members are named
  `<detector>_<cluster>`, and the roster is `runs.methods(selected)`: `single_<member>` × 4
  plus the 3 ensembles. It's dynamic, so `config.METHODS` / `CLUSTER_MEMBERS` no longer
  exist.
- There is no separate recall ensemble: the user decided `ens_raw_resetall` already is that,
  now over the TF2-picked members.
- The old hand-made split (HDDM_W + RDDM / STEPD + SEED, "where its params hurt it least")
  is gone.

## Critical: import order and sys.path

CapyMOA needs `JAVA_HOME`, which is unset on this machine. `drift_ensemble/config.py`
(`ensure_java_home`) derives it from `java -XshowSettings:properties` when it is imported. **Every
module that imports capymoa (directly or indirectly) must import `drift_ensemble.config`
first** (`ensemble_experiment.config` does so). `streams.stream_from_plan` calls
`ensure_java_home()` itself.

`drift_ensemble.config` also puts the repo root on `sys.path`, so `import streams` works.
`main.py` and pytest (`pythonpath = .` in `pytest.ini`) put it there too.

## Shared: `streams.py` (root)

- **Generators:** SEA, Agrawal and Sine (capymoa `SineGenerator`, functions 1–4, defaults).
  Sine replaced STAGGER on 2026-11-24 at the user's request; they found STAGGER a bad
  generator. Its drifts were near-total error jumps (median +0.45 in the tree's error right
  after a drift; Sine +0.23, Agrawal +0.11, SEA +0.03). Don't bring STAGGER back.
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

Seeds (2026-11-24, at the user's request; were 1–3 and 101–105): 102 tuning streams =
3 generators × {ABRUPT, GRADUAL} × seeds 1–17, and 153 test streams = 3 generators ×
{ABRUPT, GRADUAL, MIXED} × seeds 101–117.
- RECURRING was added on 2026-11-20. It cycles through 3 concepts (`RECURRING_N_CONCEPTS`).
- Since 2026-11-24, at the user's request, experiment 1's 51 RECURRING test streams are only
  generated and cached by `test` (`config.recurring_stream_specs`). No method runs on them
  and they get no statistics (`config.MODES` has no RECURRING). The user will say later
  what to do with them. DDD (experiment 2) keeps RECURRING.
- `STREAM_SIZES["n_concepts"]` is ignored by the other modes. All 63 earlier cached plans
  were verified unchanged.
- Tuning still uses ABRUPT and GRADUAL only.

Modules:
- `config.py`: every setting, plus `ensure_java_home`.
- `detectors.py`: detector factory, grids, `set_threshold`, and the 3 ensemble classes.
- `tuning.py`: tuning and screening.
- `runs.py`: HT error cache, building a method's detectors, one method on one stream, and run
  folders (save + `load_runs`).
- `stats.py`: scoring (`match_alarms`, `compute_metrics`, `delay_curve`) and every
  statistic: tables, plots, significance tests, tuning heatmaps.

- **Cache (`runs.py`, step `cache`; `test` caches the test streams):** runs one HT pass per
  stream and stores `errors.npy`, `ht_accuracy.npy` and `ground_truth.json` under `cache/`.
  HT's error sequence is deterministic given the stream, so **everything downstream replays
  these cached errors**. `tests/test_runs.py` proves equivalence with a live run.
- **Tuning (`tuning.py`, steps `tune` + `select`):** grid × cached errors → best config per
  detector × mode by TF2 (`select_params`), the clusters (`select_clusters`), plus a LOGO
  check → `selected_params.json` and `tuning_summary.csv` (rank, in_cluster). Every method
  of both experiments runs with it, as does screening.
- **Tuning phase 2 was DROPPED on 2026-11-24 at the user's request.** It gave each cluster
  one shared minimum M (`GatedDetector`, `cluster_params.json`) for the two-vote methods.
  Its code, tests and result files were deleted. Don't reintroduce it.
- **Ensembles (`detectors.py`):** `RawResetAllEnsemble`, `ClusterVoteEnsemble` and
  `ClusterSensitivityEnsemble`, keyed by member name (the threshold functions map a member
  to its detector type). `runs.run_method` runs the 7 roster methods
  (`runs.methods(selected)`), the same 7 as experiment 2:
  - `single_<member>` ×4, e.g. `single_HDDMWeighted_ABRUPT`.
  - `ens_raw_resetall`: OR; every alarm resets ALL members.
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
  sensitivity.csv, raw_fires.csv, meta.json). `stats.py` reads *only* result folders and
  must never recompute from streams. Its roster is the methods found in the runs; runs
  whose `meta.selected_params` differ from the current `selected_params.json` are skipped
  as stale.
- **Statistics:** `stats.py` (step `stats`) covers all 7 methods and writes to
  `statistics/drift_ensemble/`:
  - `runs.csv` (one row per run, including `ndt`, `tf2`, `far` and the duplicate alarms
    `n_dup`) and `delays.csv` (one row per detected drift, with its NDT);
  - `metrics.csv`: per method × mode, the **mean and std** over runs of precision, recall,
    TF2, NDT and far (`*_mean`, `*_std`), summed TP/FP/FN/n_dup, and the **median** delay
    over the detected drifts (`delay_median`). The user chose mean/median on 2026-11-23. The
    runs are 3 generators × the seeds, so a median hides a failure on a whole generator.
    Delays are skewed, so they use the median.
  - `delay_curve.csv`, plus `<MODE>/delay_curve.png`: means when every run is rescored
    for each max delay in `STATS["delay_grid"]` (0–500, step 25). The plot shows recall
    only (user's choice). The max delay counts from the drift END, so gradual curves start
    above 0 at max delay 0: alarms during the 2,000-instance transition count. The user
    confirmed this is correct.
  - one folder per mode (ABRUPT, GRADUAL, MIXED) with `singles.png` (the 4 members) and
    `ensembles.png` (the 3 ensembles). They are bar charts of `metrics.csv`: per method,
    mean precision, recall and TF2 bars with std error bars in the top row (0–1), and the
    median delay bar in the
    bottom row (in instances, since it's a different unit). Both charts of a mode share
    the delay scale.
  - `significance.csv` (Wilcoxon on TF2: each ensemble vs each single, each cluster
    ensemble vs the other ensembles), `per_type_mixed.csv` (abrupt vs gradual drifts in
    MIXED), `first_to_fire.csv` and `diversity.csv` (per member; Q-statistic and
    disagreement), `sensitivity_summary.csv` (ens_cluster_sens levels per member),
    `runtime.csv`, and `tuning_heatmaps/<detector>_<mode>.png` (tuning TF2).
  - These came from the old `analysis.py`, which was merged into `stats.py` on 2026-11-23
    at the user's request. The merged tables were verified identical to the old ones.
    - Dropped as duplicates: `detection_metrics` (same as `metrics.csv`, with a LaTeX
      copy), `per_run_metrics` (= `runs.csv`) and `run_index`. Their extra `far` and
      `n_dup` columns moved into `runs.csv` and `metrics.csv`.
    - Also dropped: the 120 per-run weight and sensitivity plots. Don't bring any of these
      back.
  - The file is named `stats.py` rather than `statistics.py` so it doesn't shadow the
    stdlib module.
- **Detector screening** (`tuning.py`, step `screen`, added 2026-11-20, not part of the
  full run): it tunes DDM, HDDMAverage, ADWIN and KSWIN (`config.SCREENING_GRIDS`) as
  single detectors on the cached errors, picks each one's config by mean tuning TF2, then
  scores them on the ABRUPT and GRADUAL test streams.
  - Grids: DDM is our own range. HDDMAverage and ADWIN use the user's values
    (2026-11-24), and ADWIN keeps delta 0.1. Rows of configs dropped from a grid stay in
    the CSV but are ignored (`tuning.only_tasks`).
  - KSWIN (added 2026-11-24) isn't in capymoa 0.9.0 or MOA. `detectors.KSWIN` is our
    own implementation from Raab et al. 2020, for 0/1 input: one hypergeometric draw
    replaces sampling positions, and an exact equal-size KS p-value is used. Seeded, so
    replays are deterministic.
  - The study's 4 are scored the same way, each with its params for the stream's mode.
  - Raw rows go to `results/results_drift_ensemble/screening/`; the summary goes to
    `statistics/drift_ensemble/screening.csv` (mean test TF2/P/R/NDT, median delay).
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
  the same params as there (`selected_params.json`, `models.load_params`).
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
  `analysis.py`; added 2026-11-23):** extended DDD on Electricity, Airlines, Covtype,
  Poker and (since 2026-11-24) 7 USP datasets (`config.REAL_DATASETS`). The step sets
  `CAPYMOA_JVM_ARGS="-Xmx3g -Xss10M"` itself, since the large datasets have 540k–830k
  instances.
  - The user's choices: only `ens_raw_resetall`, ONE run (seed 201, `REAL_SEED`), no
    more seeds.
  - There are no drift positions, so only prequential accuracy (after warm-up), the
    number of detections and accuracy over time are reported.
  - A `no_change` baseline (predict the previous label) is included because Electricity
    is strongly autocorrelated.
  - **`data/` is committed** (since 2026-11-24, user's request): all 11 ARFFs, up to
    98.6 MiB (covtype). That's under GitHub's 100 MB file limit, though GitHub warns above
    50 MB.
  - The data comes from OpenML: 151 Electricity (= MOA's elecNormNew, same as capymoa's),
    1169 Airlines (MOA's), 150 Covtype (MOA's covtypeNorm, original order) and 155 Poker
    (MOA's poker-lsn, 829,201 instances, 10 classes). It's md5-checked and stored in
    `data/`. `capymoa.datasets` can't be imported (it
    needs torchvision), and its Dropbox links are blocked on this network.
  - USP Data Stream Repository (added 2026-11-24, user's choice from my shortlist): INSECTS
    abrupt / gradual / incremental-reoccurring (balanced), NOAA, Rialto, GasSensor and
    Posture.
    - Their CSVs (headerless, class last) were converted once to ARFF. The ARFFs are
      committed in `data/` and md5-checked like the OpenML ones. The user then had the
      conversion code removed (2026-11-24). The 572 MiB zip stays out of git
      (`data/*.zip`).
    - Posture's CSV had one line with two rows glued together. It was split during the
      conversion, giving 164,860 rows.
    - Left out: duplicates of our OpenML data, sets too small for the 1,000 warm-up,
      KDDCUP99, SensorStream, Twitter, the high-dimensional "new datasets", and the
      imbalanced and pure-incremental INSECTS.
    - INSECTS drift positions are published in Souza et al. 2020 but are NOT in the zip, so
      INSECTS is scored like the other real data for now (accuracy, detections).
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
