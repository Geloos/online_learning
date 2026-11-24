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

`main.py` runs the drift-ensemble pipeline (steps 1–5), then the DDD experiment (steps 6–7)
and DDD on the real datasets (step 8). Each step runs as its own process, and the run stops
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
| 1 | `cache` | cache the tree errors of the 102 tuning streams | |
| 2 | `tune` | tuning: every grid config on the tuning streams | `--limit N` |
| 3 | `select` | tuning: best config per detector × mode, and the clusters (both by TF2) | |
| 4 | `test` | cache the 204 test streams, run the 7 methods on the 153 ABRUPT/GRADUAL/MIXED ones | `--methods` |
| 5 | `stats` | statistics of experiment 1 | |
| 6 | `ddd` | extended DDD on the synthetic streams | `--generators`, `--modes`, `--seeds`, `--methods` |
| 7 | `ddd-stats` | statistics of the DDD runs | |
| 8 | `real` | DDD on the real datasets, and their statistics | `--datasets`, `--methods` |
| – | `screen` | DDM, HDDMAverage, ADWIN and KSWIN screened against the 4 (not in the full run) | |

Tests: `python -m pytest` runs both suites. `python -m pytest -m "not jvm"` runs only the fast
unit tests, which need no JVM.

## What runs

**Streams (`streams.py`).** The generators are SEA, Agrawal and Sine. A drift changes the
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
  1. Cache the 102 tuning streams: 3 generators × ABRUPT/GRADUAL × 17 seeds (1–17).
  2. Tuning: tune each detector's grid on them, and pick its best config by TF2
     (see Scoring) separately for ABRUPT and for GRADUAL.
  3. Clusters: the 2 detectors with the best tuning TF2 on ABRUPT streams form the ABRUPT
     cluster, the 2 best on GRADUAL streams the GRADUAL cluster. Each member runs with its
     cluster's mode params, so one detector can be in both clusters with different params
     (e.g. `HDDMWeighted_ABRUPT` and `HDDMWeighted_GRADUAL`).
  4. Run 7 methods on 153 test streams: 3 generators × ABRUPT/GRADUAL/MIXED × 17 seeds
     (101–117). The 51 RECURRING test streams are generated and cached, but not run.
  5. Score them.
- **Methods:**

| Method | Alarm rule |
|---|---|
| `single_<detector>_<cluster>` (×4, one per cluster member) | the member fires |
| `ens_raw_resetall` | any member fires; all 4 reset |
| `ens_cluster_vote` | both detectors of a cluster fire within 250 instances (or one full-weight detector); the faster voter gains weight |
| `ens_cluster_sens` | same pair rule; the slower voter is made more sensitive (its threshold is changed mid-stream) |

- **Scoring:** an alarm is a true positive if it falls in `[drift start, drift end + 500]`.
  The one score is **TF2 (timely F2)**, which replaced F1:
  - NDT (normalized detection time) of a detected drift = (detection − drift start) /
    (drift end − drift start + 500), so 0 at the start and 1 at the end of its window;
  - timely recall = recall × (1 − mean NDT): a late detection earns less, a miss earns 0;
  - TF2 = 5 · P · R_t / (4 · P + R_t), the F2 of precision and timely recall (recall
    counts twice as much, and precision still stops a detector that fires all the time).

  `statistics/drift_ensemble/` holds, for every method × mode:
  - mean ± std of precision, recall, TF2, NDT and false-alarm rate over the runs, and the
    median delay;
  - recall as the allowed delay shrinks from 500 to 0;
  - bar charts per mode (singles and ensembles);
  - Wilcoxon tests on TF2, recall on abrupt vs gradual drifts in MIXED, which member
    fires first, how much the members agree, the sensitivity levels of
    `ens_cluster_sens`, run times, and heatmaps of the tuning TF2.

**Experiment 2: extended DDD.**
- **Streams:** 3 generators × 4 modes × 10 seeds, with 8 drifts per stream, one every 5,000
  instances; gradual drifts are 1,000 wide.
- **Drift signals:** the 7 detection methods of experiment 1, with the same tuning: the 4
  cluster-member singles, `ens_raw_resetall`, `ens_cluster_vote` and `ens_cluster_sens`.
  This gives 840 runs. DDD keeps its RECURRING streams.
- **Learner:**
  - Each model is an online-bagging ensemble of 10 Hoeffding trees with Naive Bayes
    adaptive leaves.
  - After a drift alarm, up to 5 ensembles vote: new/old × low/high diversity, plus one
    recalled from the concept pool.
  - The after-drift phase ends when one ensemble clearly wins, or when the adaptive cap for
    that drift kind runs out.
- **Real data:** DDD with `ens_raw_resetall`, one run per dataset, next to a no-change
  baseline:
  - from OpenML (MOA versions): Electricity, Airlines, Covtype and Poker;
  - from the USP Data Stream Repository (Souza et al.): INSECTS abrupt, gradual and
    incremental-reoccurring (balanced), NOAA, Rialto, GasSensor and Posture.
- **Measured:**
  - prequential accuracy
  - detection TF2, NDT and delay
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
  detectors.py         detectors (factory, grids, thresholds) and the
                       reset-all / cluster-vote / cluster-sens ensembles
  tuning.py            tuning, detector screening
  runs.py              tree-error cache, one method on one test stream, result folders
  stats.py             alarm <-> drift scoring, every statistic (reads result folders
                       only)
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
                            significance.csv  per_type_mixed.csv  first_to_fire.csv
                            diversity.csv  sensitivity_summary.csv  runtime.csv
                            screening.csv  <MODE>/{singles,ensembles,delay_curve}.png
                            tuning_heatmaps/
  ensemble_experiment/      recovery.csv  drifts.csv  accuracy_200_<MODE>.png
                            real/<dataset>/{accuracy.csv,accuracy_over_time.png}
```

Only `results/results_drift_ensemble/tuning/` is in git. It holds the tuned detector
parameters: `selected_params.json`, which both experiments read. Everything else under
`results/` is regenerated by `main.py`. The real datasets are committed as ARFF files in
`data/` (the OpenML ones would also be downloaded again if missing).

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
