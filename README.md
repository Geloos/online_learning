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
and DDD on the real datasets (step 9). Each step runs as its own process, and the run stops
at the first failure. Every step is resumable, so a re-run only does the work that is
missing. Without `--yes`, each long step prints a time estimate and asks for confirmation.

| `main.py` flag | Meaning |
|---|---|
| `--workers N` | Parallel worker processes, each with its own JVM (default 6) |
| `--yes` | Skip every confirmation prompt |
| `--only drift_ensemble` / `--only ensemble_experiment` | Run only one experiment |
| `--dry-run` | Print the steps without running them |

Each JVM may grow to 8 GB by default. On machines with less RAM, cap it before running:
`export CAPYMOA_JVM_ARGS="-Xmx1g -Xss10M"`. The `real` step always runs with
`-Xmx3g -Xss10M` and at most 4 workers, because the real datasets have up to 830k instances.

You can also run one step at a time: `python main.py STEP [--workers N] [--yes] [options]`.
`python main.py STEP -h` lists a step's options.

| # | Step | What it does | Options |
|---|---|---|---|
| 1 | `cache` | cache the tree errors of the 18 tuning streams | |
| 2 | `tune` | tuning phase 1: every grid config on the tuning streams | `--limit N` |
| 3 | `select` | tuning phase 1: best config per detector × mode | `--objective` |
| 4 | `phase2` | tuning phase 2: each cluster's shared minimum | |
| 5 | `test` | run the 7 methods on the 60 test streams | `--methods` |
| 6 | `stats` | statistics of experiment 1 | `--no-plots` |
| 7 | `ddd` | extended DDD on the synthetic streams | `--generators`, `--modes`, `--seeds`, `--methods` |
| 8 | `ddd-stats` | statistics of the DDD runs | |
| 9 | `real` | DDD on the real datasets, and their statistics | `--datasets`, `--methods` |
| – | `screen` | DDM, HDDMAverage and ADWIN screened against the 4 (not in the full run) | |

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
  - mean ± std of precision, recall and F1 over the runs, and the median delay;
  - recall as the allowed delay shrinks from 500 to 0;
  - bar charts per mode (singles and ensembles).

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
main.py                every step of both experiments (the only entry point)
streams.py             shared stream generator (all 4 modes)
drift_ensemble/        experiment 1
  config.py            every setting of the experiment (also sets JAVA_HOME)
  detectors.py         detectors (factory, grids, thresholds, phase-2 gate) and the
                       reset-all / cluster-vote / cluster-sens ensembles
  tuning.py            tuning phase 1 and 2, detector screening
  runs.py              tree-error cache, one method on one test stream, result folders
  stats.py             alarm <-> drift scoring, per-method statistics
  analysis.py          tables and plots (reads result folders only)
  tests/               test_detectors, test_tuning, test_runs, test_stats
ensemble_experiment/   experiment 2
  config.py            every setting of the experiment
  ddd.py               extended DDD state machine (+ trackers, adaptive cap, concept pool)
  models.py            online bagging over Hoeffding trees, drift signals
  runs.py              one run -> result folder (synthetic and real datasets)
  analysis.py          statistics (reads result folders only)
  tests/               test_ddd, test_runs, test_analysis
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
both experiments read. Everything else under `results/` is regenerated by `main.py`. The
real datasets are downloaded from OpenML into `data/` (not in git) on first use.

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
