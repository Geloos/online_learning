# Cluster Drift-Detector Ensembles on CapyMOA

A detection study. A Hoeffding Tree (HT) learns prequentially on synthetic drift streams
and is **never reset**. Four drift detectors (STEPD, SEED, HDDMWeighted, RDDM) watch the
HT's 0/1 error, and several ensembles combine their alarms. Every alarm is scored against
the known drift positions (ground truth).

The detectors are tuned per drift type and split into two clusters based on that tuning:
- **ABRUPT:** SEED + RDDM
- **GRADUAL:** STEPD + HDDMWeighted

The cluster ensembles raise an alarm only when both members of a cluster agree. Stream
design and tuning grids follow Cerqueira et al.'s drift-detector benchmark (CapyMOA), but
all code is written from scratch.

## Setup

Python 3.12 (the requirements are pinned for it) and a JDK 11+. If `JAVA_HOME` is unset, it is derived automatically from the
`java` on PATH.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # torch: CPU build is enough
.venv\Scripts\python -m pytest                            # 86 tests, ~10 s
.venv\Scripts\python -m pytest -m "not jvm"               # unit tests only, no JVM
```

## Pipeline

Run the scripts from the project root with the venv's Python.
- **Resumable:** every script skips work that is already finished.
- **Time estimate:** each long script times one unit of work, prints an estimate and asks
  for confirmation. `--yes` skips the prompt.
- **Parallelism:** `--workers N` sets the number of workers. Each worker starts its own JVM;
  6 is a good value on 12 cores / 16 GB.

```
python scripts/01_cache_tuning_streams.py   # build + cache the 18 tuning streams (one HT pass each)
python scripts/02_tune.py                   # 242 detector configs x 18 streams -> tuning_runs.csv
python scripts/03_select_params.py          # best config per detector x mode -> selected_params.json
#   STOP: check that tuning_summary.csv still justifies CLUSTER_MEMBERS in config.py
python scripts/04_run_test.py               # cache the 45 test streams, run 8 methods on each
python scripts/05_analyze.py                # tables + plots under results/analysis/
```

`04_run_test.py --methods a,b` runs only some of the methods.

## How it works

```
streams.py ─► cache.py (HT runs ONCE per stream) ─► results/cache/<stream>/errors.npy + ground_truth.json
                                                       │   (everything below replays these errors)
   02 tune ─► tuning.py: every detector config on the 18 tuning streams
   03 select ─► selection.py: best config per detector for ABRUPT and for GRADUAL -> clusters
   04 test ─► workflow.py: each method on each of the 45 test streams -> runio.py saves alarms
   05 analyze ─► analysis.py + evaluation.py: alarms vs ground truth -> tables, plots
```

1. **Streams (`streams.py`).** SEA, Agrawal and STAGGER generators.
   - A new concept (classification function) starts every 10,000 instances, giving 30 drifts
     per stream.
   - ABRUPT drifts switch instantly. GRADUAL drifts blend over 2,000 instances. MIXED
     streams have 15 of each, in random order.
   - The seed fixes the whole stream, so the same seed always gives the same stream.
   - Tuning seeds (1–3) and test seeds (101–105) are kept disjoint (asserted in
     `config.py`).
2. **The learner (`cache.py`).** The HT predicts each instance, then trains on it
   (prequential).
   - The first 1,000 instances are warm-up.
   - The detectors never touch the HT, so its 0/1 error sequence is fixed per stream. It is
     computed once and cached.
   - Every later step replays that cached sequence instead of re-running the HT.
     `tests/test_cache_replay.py` proves the replay is identical to a live run.
3. **Detectors (`detectors.py`).** A factory for the four capymoa detectors, plus the tuning
   grids.
   - `set_threshold` changes a detector's drift threshold mid-stream.
   - It sets both the MOA option and the private Java field; setting only the option has no
     effect. This is used by `ens_cluster_sens`.
4. **Tuning and selection (`tuning.py`, `selection.py`).** Every grid config is scored on
   the tuning streams.
   - For each detector, the best config by mean F1 is picked separately for ABRUPT and for
     GRADUAL.
   - A detector joins the cluster of the mode where its tuned F1 was higher.
   - In the ensembles, each detector runs with its own cluster's parameters.
5. **Ensembles (`ensemble.py`).** These take error values in and give alarms out. See
   [Methods](#methods).
6. **Test runs (`workflow.py`, `runio.py`).** One method on one test stream gives one
   result folder, `results/test/<generator>,<mode>,<seed>,<method>/`. It holds:
   - `alarms.json`, `raw_fires.csv`, `weights.csv`, `sensitivity.csv`, `verifications.csv`
   - `meta.json`, which is written last; its presence marks the run as complete.
7. **Scoring and analysis (`evaluation.py`, `analysis.py`).**
   - **Matching rule:** an alarm is a TP if it falls in `[drift start, drift end + MAX_DELAY]`,
     where `MAX_DELAY = 500`.
   - **Duplicates:** further alarms in the same window are duplicates and count as FP
     (`n_dup` in the tables).
   - **Other alarms** outside every window are FP.
   - **Metrics:** precision, recall, F1, delay (instances from the drift start to the
     matched alarm), false-alarm rate, and Wilcoxon signed-rank tests on F1.
   - `analysis.py` reads **only** result folders and never re-runs anything.

`config.py` holds every hyperparameter: streams, seeds, grids, clusters, `VOTE`, `SENS`,
and the method roster. Importing it also sets `JAVA_HOME`, so any module that imports
capymoa must import `drift_ensemble.config` first.

## Methods

These are the 8 methods in `config.METHODS`, each run on each of the 45 test streams
(15 ABRUPT, 15 GRADUAL, 15 MIXED):

| Method | Alarm rule | On alarm |
|---|---|---|
| `single_STEPD`, `single_SEED`, `single_HDDMWeighted`, `single_RDDM` | the detector fires | the detector self-resets |
| `ens_raw` | any detector fires (OR) | only the firing detector self-resets |
| `ens_raw_resetall` | any detector fires (OR) | all 4 detectors reset |
| `ens_cluster_vote` | both members of one cluster fire within `VOTE["WINDOW"]` = 250 instances, or one full-weight detector fires | all 4 reset; the earlier voter of a pair gains weight, the later one loses it (weights in [0.5, 1.0]) |
| `ens_cluster_sens` | both members of one cluster fire within `SENS["WINDOW"]` = 250 instances | all 4 reset; the **later** voter becomes more sensitive by `LATE_STEP × gap / WINDOW` levels (cap `L_MAX` = 3). Its threshold is changed mid-stream via `set_threshold`. |

The clusters never influence each other's weights or sensitivity.

**Dropped methods.** `ens_verified`, `ens_verified_resetall` and `ens_cooldown` are still
runnable through `workflow.run_method`, but they are no longer in the roster.
- `ens_verified*` used Naive Bayes + one-sided McNemar verification (`verifier.py`).
  It wrongly rejected about 28% of real alarms.
- `ens_cooldown` added a silence window after each alarm.

`analysis.py` ignores result folders of methods that are not in the roster.

## Layout

```
drift_ensemble/
  config.py      every hyperparameter; also sets JAVA_HOME (import it before capymoa)
  _java.py       derives JAVA_HOME from `java -XshowSettings:properties`
  streams.py     stream plans (pure Python, reproducible) + capymoa DriftStream build
  cache.py       one HT pass per stream -> errors.npy / ht_accuracy.npy / ground_truth.json
  detectors.py   detector factory, tuning grids, cached-error replay, set_threshold
  tuning.py      grid evaluation on cached errors (pool workers)
  selection.py   best config per detector x mode (+ LOGO check)
  ensemble.py    RawResetAll, ClusterVote, ClusterSensitivity (+ dropped Verified, Cooldown)
  verifier.py    one-sided McNemar + NB verification state machine (dropped methods only)
  workflow.py    run_method: one method on one cached test stream
  runio.py       result-folder I/O
  evaluation.py  alarm<->drift matching, metrics, delay curve, per-type metrics
  analysis.py    all tables and plots (reads result folders only)
scripts/         thin CLIs: 01 cache, 02 tune, 03 select, 04 test runs, 05 analyze
tests/           unit tests with fake detectors (tests/fakes.py) + `jvm`-marked integration tests
results/         cache/  tuning/  test/  analysis/{tables,plots}
```

## Tests

The tests check the code; they are not part of the experiment.

- **Unit tests** drive the ensembles with `StubDetector` fakes that fire at scripted
  instants, so they need no JVM.
  - Ensembles: `test_cluster_vote.py`, `test_cluster_sens.py`, `test_ensemble.py`
  - Verifier and McNemar: `test_verifier.py`, `test_mcnemar.py`
  - Scoring: `test_evaluation.py`
  - Configuration: `test_config.py`
- **`jvm`-marked tests** start real capymoa:
  - the cached replay equals a live HT run (`test_cache_replay.py`)
  - `reset_detector` really resets MOA state (`test_reset.py`)
  - `set_threshold` changes sensitivity mid-stream and survives resets
    (`test_set_threshold.py`)
  - streams and ground truth are built correctly (`test_streams.py`)

## Second experiment: extended DDD (`ensemble_experiment/`)

Here the learner **is** adapted. Each model is an online-bagging ensemble of 10 Hoeffding
trees, and DDD (Minku & Yao 2011) keeps up to 5 of them: new-low, new-high, old-low,
old-high, and a model recalled from the concept pool. It is extended with:
- a fast/slow drift classification,
- an adaptive per-kind cap on the after-drift phase,
- a concept pool of frozen past ensembles.

DDD's drift signal is one of this study's detection methods. Each of 7 methods drives its
own runs:
- the 4 singles,
- `ens_raw_resetall`,
- `ens_cluster_vote`,
- `ens_cluster_sens`.

The streams are this study's generators, shortened to 8 drifts every 5,000 instances. A new
RECURRING mode revisits 3 concepts. All parameters are in `ensemble_experiment/config.py`.

The full grid is 3 generators × 4 modes × 10 seeds × 7 methods = 840 runs. Each run takes
about 17 s.

### Running it on another machine

```
git clone <repo-url> && cd <repo>
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
.venv/bin/python -m pytest ensemble_experiment/tests    # 47 tests

# Each worker starts a JVM. capymoa lets every JVM grow to 8 GB by default; cap it.
# If a run fails with OutOfMemoryError, raise -Xmx1g to -Xmx2g.
export CAPYMOA_JVM_ARGS="-Xmx1g -Xss10M"                # Windows: set CAPYMOA_JVM_ARGS=-Xmx1g -Xss10M

.venv/bin/python ensemble_experiment/run_experiment.py --workers N --yes   # resumable
```

- **Workers:** set N to about the number of physical cores, with roughly 1.5 GB of RAM per
  worker.
- **Requirements:** you need a JDK on PATH. `results/tuning/selected_params.json` (the tuned
  detector parameters) is part of the repo.

Each run writes `ensemble_experiment/results/runs/<generator>,<mode>,<seed>,<method>/`,
which holds:
- `correct.npy`: 0/1 correctness for every instance.
- `meta.json`: drift ground truth and the run summary.
- `episodes.csv`: one row per alarm.
- `steps.csv`: the state every 100 instances.
- `alarms.json`.

The outputs are git-ignored. To bring them back through git:

```
git add -f ensemble_experiment/results/runs && git commit -m "DDD results" && git push
```

Then build the tables and plots under `ensemble_experiment/results/analysis/` with
`python ensemble_experiment/analyze.py`.

## capymoa 0.9.0 quirks

These were verified by probes; do not "fix" them.

- **Self-reset.** All four detectors restart their own statistics after
  `detected_change()` (`DETECTOR_SELF_RESETS`).
- **Full reset.** capymoa's `reset()` only clears the Python-side flags.
  `ensemble.reset_detector` also calls MOA's `resetLearning()`.
- **Mid-stream thresholds.** Thresholds live in private Java fields, so changing a MOA
  option mid-stream does nothing. `set_threshold` sets the option and the field together.
- **`GradualDrift`.** `GradualDrift(position=...)` means the drift *center*, so the code
  passes `start`/`end` instead. Its constructor also prints a stray `None`, which is
  suppressed.
- **torch.** capymoa imports torch unconditionally; the CPU build is enough.
