# Drift-Detector Ensembles and Extended DDD on CapyMOA

This repo holds two experiments on synthetic concept-drift streams. Both use the same
stream generator, `streams.py`.

1. **`drift_ensemble/`: detection study.** A Hoeffding Tree learns the stream and is never
   reset. Four drift detectors (STEPD, SEED, HDDMWeighted, RDDM) watch its 0/1 error.
   Ensembles combine their alarms. Every alarm is scored against the known drift positions.
2. **`ensemble_experiment/`: extended DDD.** Here the learner *is* adapted. DDD (Minku &
   Yao 2011) keeps several online-bagging ensembles of Hoeffding trees and reacts to drift.
   It is extended with a fast/slow drift classification, an adaptive cap on the
   after-drift phase, and a pool of past concepts that can be recalled. DDD's drift signal
   is each of the detection methods from experiment 1, in turn.

## Setup

You need Python 3.12 and a JDK on PATH. `JAVA_HOME` is derived automatically.

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
```

The requirements are pinned to the versions the results were produced with. CPU-only torch
is enough; capymoa imports torch.

## Running

Run everything:

```
python main.py
```

`main.py` runs the drift-ensemble pipeline (steps 1–6), then the DDD experiment (steps 7–8)
and DDD on the real datasets (step 9).
Every step is resumable, so a re-run only does the work that is missing. Without `--yes`,
each long step prints a time estimate and asks for confirmation.

| `main.py` flag | Meaning |
|---|---|
| `--workers N` | Parallel worker processes, each with its own JVM (default 6) |
| `--yes` | Skip every confirmation prompt |
| `--only drift_ensemble` / `--only ensemble_experiment` | Run only one experiment |
| `--dry-run` | Print the steps without running them |

Each JVM may grow to 8 GB by default. On machines with less RAM, cap it before running:
`export CAPYMOA_JVM_ARGS="-Xmx1g -Xss10M"`. Step 9 always runs with `-Xmx3g -Xss10M` and at
most 4 workers, because the real datasets have up to 830k instances.

You can also run the steps one at a time:

| Step | Command | Flags |
|---|---|---|
| 1 | `python drift_ensemble/scripts/01_cache_tuning_streams.py` | `--workers`, `--yes` |
| 2 | `python drift_ensemble/scripts/02_tune.py` | `--workers`, `--yes`, `--limit N` (first N configs only) |
| 3 | `python drift_ensemble/scripts/03_select_params.py` | `--objective f1\|f2\|delay_auc` |
| 4 | `python drift_ensemble/scripts/03b_tune_min_instances.py` | `--workers`, `--yes` |
| 5 | `python drift_ensemble/scripts/04_run_test.py` | `--workers`, `--yes`, `--methods a,b` |
| 6 | `python drift_ensemble/scripts/05_analyze.py` | `--no-plots` |
| 7 | `python ensemble_experiment/run_experiment.py` | `--workers`, `--yes`, `--generators`, `--modes`, `--seeds`, `--methods` |
| 8 | `python ensemble_experiment/analyze.py` | none |
| 9 | `python ensemble_experiment/run_real.py` | `--workers`, `--yes`, `--datasets`, `--methods` |

Tests: `python -m pytest` runs both suites. `python -m pytest -m "not jvm"` runs only the fast
unit tests, which need no JVM.

## What runs

**Streams (`streams.py`).** The generators are SEA, Agrawal and STAGGER. A drift changes the
classification function. The seed fixes the whole stream.

| Mode | Drifts |
|---|---|
| ABRUPT | instant switches |
| GRADUAL | blend over a window |
| MIXED | half abrupt, half gradual, shuffled |
| RECURRING | as MIXED, but cycling through 3 concepts, so concepts come back |

**Experiment 1: drift ensemble.**
- **Streams:** 30 drifts per stream, one every 10,000 instances; gradual drifts are 2,000
  wide. The tree trains for 1,000 instances before the detectors start.
- **Caching:** the tree's 0/1 error sequence is computed once per stream and cached. Every
  detector configuration replays it.
- **Steps:**
  1. Cache the 18 tuning streams.
  2. Tuning phase 1: tune each detector's grid on them, and pick the best config separately
     for ABRUPT and for GRADUAL.
  3. Split the detectors into two clusters: ABRUPT = HDDMWeighted + RDDM,
     GRADUAL = STEPD + SEED. Each detector goes where its cluster's params hurt it least on
     the other stream type.
  4. Tuning phase 2: give each cluster one minimum M. After a reset, neither detector of the
     cluster may fire before M instances, so both switch back on together. For each M
     (50–2000), the cluster's two detectors are re-tuned on their mode's tuning streams, and
     the M with the best mean F1 of the two wins. The ABRUPT and GRADUAL clusters get their
     own M. Only the two methods that need two votes (`ens_cluster_vote`,
     `ens_cluster_sens`) use phase 2. The singles and `ens_raw_resetall` use phase 1.
  5. Run 7 methods on 60 test streams: 3 generators × 4 modes × 5 seeds. RECURRING uses the
     same tuned params as MIXED.
  6. Score them.
- **Methods:**

| Method | Alarm rule |
|---|---|
| `single_*` (×4) | the detector fires |
| `ens_raw_resetall` | any detector fires; all 4 reset |
| `ens_cluster_vote` | both detectors of a cluster fire within 250 instances (or one full-weight detector); the faster voter gains weight |
| `ens_cluster_sens` | same pair rule; the slower voter is made more sensitive (its threshold is changed mid-stream) |

- **Scoring:** an alarm is a true positive if it falls in `[drift start, drift end + 500]`.
  The metrics are precision, recall, F1, delay and false-alarm rate, with Wilcoxon tests on F1.
  `statistics/drift_ensemble/` holds, for every method × mode:
  - medians of precision, recall, F1 and delay;
  - curves of how they change as the allowed delay shrinks from 500 to 0;
  - histograms of each metric.

**Experiment 2: extended DDD.**
- **Streams:** 3 generators × 4 modes × 10 seeds, with 8 drifts per stream, one every 5,000
  instances; gradual drifts are 1,000 wide.
- **Drift signals:** 7 detection methods from experiment 1: the 4 singles,
  `ens_raw_resetall`, `ens_cluster_vote` and `ens_cluster_sens`. This gives 840 runs.
- **Learner:**
  - Each model is an online-bagging ensemble of 10 Hoeffding trees with Naive Bayes
    adaptive leaves.
  - After a drift alarm, up to 5 ensembles vote: new/old × low/high diversity, plus one
    recalled from the concept pool.
  - The after-drift phase ends when one ensemble clearly wins, or when the adaptive cap for
    that drift kind runs out.
- **Real data:** DDD with `ens_raw_resetall` on Electricity, Airlines, Covtype and Poker
  (OpenML, MOA versions), one run each, next to a no-change baseline.
- **Measured:**
  - prequential accuracy
  - detection F1 and delay
  - how each after-drift phase ended
  - fast/slow classification accuracy
  - how often a recalled concept matches the true one

All hyperparameters are in each experiment's `config.py`.

## Layout

```
main.py                runs both experiments end to end
streams.py             shared stream generator (all 4 modes)
drift_ensemble/        experiment 1
  config.py            every setting of the experiment (also sets JAVA_HOME)
  cache.py             one tree pass per stream -> cached 0/1 errors + ground truth
  detectors.py         detector factory, grids, mid-stream threshold control
  tuning.py, selection.py   grid evaluation, best config per detector x mode (phase 1)
  min_instances.py     phase 2: each cluster's shared minimum + re-tuned params
  ensemble.py          raw / reset-all / cluster-vote / cluster-sens ensembles
  workflow.py, runio.py     one method on one test stream, result folders
  evaluation.py        alarm <-> drift matching and metrics
  analysis.py          tables and plots (reads result folders only)
  stats.py             per-method statistics -> statistics/drift_ensemble/
  screening.py         DDM / HDDMAverage / ADWIN screened against the 4 (singles,
                       cached errors; scripts/screen_detectors.py)
  cli.py               shared CLI helpers (time estimate, workers)
  scripts/             01-05 pipeline steps (+ 03b, tuning phase 2)
  tests/
ensemble_experiment/   experiment 2
  config.py            every setting of the experiment
  bagging.py           online bagging over Hoeffding trees
  ddd.py               extended DDD state machine
  tracker.py, cap.py, pool.py   accuracy trackers, adaptive cap, concept pool
  detection.py         experiment 1's methods as DDD drift signals
  logger.py, runner.py      one run -> result folder
  analysis.py          best-method statistics (reads result folders only)
  real.py              DDD on the real datasets (data/ downloaded from OpenML)
  run_experiment.py, analyze.py, run_real.py   CLIs
  tests/
results/
  results_drift_ensemble/   cache/  tuning/  test/  screening/
  results_ensemble/         runs/<generator>,<mode>,<seed>,<method>/  real/
statistics/
  drift_ensemble/           runs.csv  delays.csv  metrics.csv  delay_curve.csv
                            screening.csv  <MODE>/{singles,ensembles,delay_curve}.png
                            analysis/{tables,plots}
  ensemble_experiment/      recovery.csv  drifts.csv  accuracy_200_<MODE>.png
                            real/<dataset>/{accuracy.csv,accuracy_over_time.png}
```

Only `results/results_drift_ensemble/tuning/` is in git. It holds the tuned detector
parameters: `selected_params.json` (phase 1) and `cluster_params.json` (phase 2), which
both experiments read. Everything else under `results/` is regenerated by
`main.py`.

## capymoa 0.9.0 notes

These were verified by probes.
- The four detectors reset their own statistics after `detected_change()`.
- `reset()` only clears the Python side, so a full reset also calls MOA's `resetLearning()`.
- Detector thresholds live in private Java fields: changing the MOA option alone mid-stream
  does nothing.
- `GradualDrift(position=...)` means the drift's center, so the code passes `start`/`end`.
- For the DDD trees:
  - Frozen copies use MOA's `copy()`, because Python's `deepcopy` does not duplicate the Java
    tree.
  - Poisson bagging trains k separate times. A single weight-k call gives different trees.
