# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis ("ptyxiakh") experiment: a verified heterogeneous drift-detector ensemble on CapyMOA.
A Hoeffding Tree learns prequentially on synthetic drift streams and is **never reset**. Four
detectors (STEPD, SEED, HDDMWeighted, RDDM) watch its 0/1 error; when one fires, two small
Naive Bayes models + a one-sided McNemar test verify whether the concept really changed, and
the verdict only mutes/unmutes detectors (one ablation also resets detectors on a confirmed
drift) — the Hoeffding Tree is never adapted. Stream design and
tuning grids follow Cerqueira et al.'s benchmark, but all code is written from scratch (their
repo is unlicensed — never copy from it).

## Commands

Always use the venv's Python from the project root:

```
.venv\Scripts\python.exe -m pytest                       # full suite (~10 s)
.venv\Scripts\python.exe -m pytest -m "not jvm"          # unit tests only, no JVM startup
.venv\Scripts\python.exe -m pytest tests/test_verifier.py::test_name   # single test
```

Pipeline (each script is resumable — finished work is skipped on re-run):

```
.venv\Scripts\python.exe scripts/01_cache_tuning_streams.py
.venv\Scripts\python.exe scripts/02_tune.py
.venv\Scripts\python.exe scripts/03_select_params.py
.venv\Scripts\python.exe scripts/04_run_test.py
.venv\Scripts\python.exe scripts/05_analyze.py
```

Every long-running script times one unit of work, prints an estimate, and waits for
confirmation — pass `--yes` to skip the prompt (required when running non-interactively)
and `--workers N` to set parallelism. Each worker spawns its own JVM; on this machine
(12 logical cores, 16 GB RAM) 6 workers is the sweet spot.

After `03_select_params.py`, STOP: `results/tuning/tuning_summary.csv` must justify the
`CLUSTER_MEMBERS` assignment in `config.py` (currently the evidence-based split:
SEED/RDDM = ABRUPT cluster, STEPD/HDDMWeighted = GRADUAL cluster) before running 04.

## Critical: import order

CapyMOA needs `JAVA_HOME`, which is unset on this machine. `drift_ensemble/_java.py`
derives it from `java -XshowSettings:properties` when `drift_ensemble.config` is imported.
**Every module that imports capymoa (directly or indirectly) must import
`drift_ensemble.config` first.** A bare `import capymoa` in a fresh process will fail.

## Architecture

Data flows one way through the package; `config.py` holds *every* hyperparameter (streams,
seeds, grids, cluster assignment, verification N/D/E/alpha, weights) — no magic numbers
anywhere else, and tuning/test seeds are asserted disjoint there.

- `streams.py` builds reproducible stream *plans* in pure Python, then materializes them as
  capymoa DriftStreams.
- `cache.py` runs one HT pass per stream and stores `errors.npy` / `ht_accuracy.npy` /
  `ground_truth.json` under `results/cache/`. Because HT's error sequence is deterministic
  given the stream, **everything downstream replays these cached errors instead of re-running
  HT** (`USE_CACHED_ERRORS_FOR_TEST_RUNS`); `tests/test_cache_replay.py` proves equivalence
  against a live run. This is why tuning 242 configs is cheap: detectors replay a cached
  error array, no JVM learner in the loop.
- `detectors.py` (factory + grids) → `tuning.py` (grid × cached errors, pool workers) →
  `selection.py` (best config per detector × mode, LOGO check).
- `verifier.py` is the NB + McNemar verification state machine; `ensemble.py` wires it into
  `VerifiedDetectorEnsemble` (plus `RawResetAllEnsemble`, `CooldownEnsemble` and
  `ClusterVoteEnsemble`); `workflow.py` runs the prequential test for the 8 roster methods
  (`config.METHODS`): `single_*` ×4, `ens_raw` (OR; each detector resets only itself via
  the capymoa self-reset), `ens_raw_resetall` (OR + every alarm resets ALL detectors),
  `ens_cluster_vote` (two votes from the same cluster within `VOTE["WINDOW"]` instances —
  or one vote from a detector at full weight — signal drift and reset ALL detectors; the
  earlier voter of a pair gains weight proportional to its speed advantage, the later one
  loses it, weights clamped to [0.5, 1.0]), `ens_cluster_sens` (same pair rule and
  reset-all, no weights: the later voter of a pair gets a more sensitive drift threshold,
  scaled by its lag; levels only rise, capped at `L_MAX` — the earlier "stricter on an
  unpartnered vote" rule was dropped; per-cluster levels in `config.SENS`, applied mid-stream via `detectors.set_threshold`,
  logged to `sensitivity.csv`). The dropped methods `ens_cooldown`,
  `ens_verified`, `ens_verified_resetall` (verification muting underperformed: ~28% of
  real alarms wrongly rejected, shadow verifications suppressed recall) are still runnable
  in `workflow.run_method`, but `analysis.py` ignores their stale result folders.
- `runio.py` defines the run-folder format (alarms.json, verifications.csv, weights.csv,
  meta.json); `analysis.py` reads *only* those result folders to produce tables/plots —
  it must never recompute from streams.
- `scripts/` are thin CLIs over the package; shared gate/parallelism logic is in
  `scripts/_common.py`.

Tests use fakes (`tests/fakes.py`) for unit coverage; JVM-dependent integration tests carry
the `jvm` marker.

## Second experiment: `ensemble_experiment/` (extended DDD)

Separate package, same venv. It implements the user's extended DDD (Minku & Yao 2011, plus a
fast/slow drift classification, an adaptive per-kind cap on the after-drift mode, and a concept
pool with recall). Unlike the detector study, the learner IS adapted here: online-bagging ensembles of CapyMOA
Hoeffding trees, driven directly through their Java `moa_learner`. DDD's drift signal is one of the
detector study's methods (`config.METHODS`, 7 entries; `ens_raw` is omitted because DDD resets the
whole signal on every alarm, which makes it identical to `ens_raw_resetall`). It reuses
`drift_ensemble` streams (via `streams.stream_from_plan`), detectors and tuned params
(`workflow.build_method_ensemble`, `results/tuning/selected_params.json`) and the matching rule
(`evaluation`). The one new stream mode is RECURRING (3 concepts revisited, half abrupt/half gradual). All its
params are in `ensemble_experiment/config.py` (short streams: 8 drifts every 5,000; seeds 201–210).

```
.venv\Scripts\python.exe -m pytest ensemble_experiment/tests          # its own suite
.venv\Scripts\python.exe ensemble_experiment/run_experiment.py --workers 6 [--yes]
.venv\Scripts\python.exe ensemble_experiment/analyze.py
```

The tree runs are live: no error cache is possible, because DDD's errors depend on its own alarms.
Each run is about 17 s. Results go under `ensemble_experiment/results/runs/` (`meta.json` written
last = done). Quirks:
- Frozen copies must use MOA `copy()`; Python `deepcopy` doesn't duplicate the Java tree.
- Poisson bagging trains k separate times. A single weight-k call is NOT equivalent for
  NB-adaptive trees (93–100% agreement).

## capymoa 0.9.0 quirks (verified by probes, do not "fix")

- All four detectors restart their own statistics after `detected_change()` — they fire
  again without an explicit `reset()` (`DETECTOR_SELF_RESETS` in config.py).
- `GradualDrift(position=...)` means the drift **center**; we pass `start`/`end` instead.
- `GradualDrift`'s constructor prints a stray "None" — suppressed in `streams.make_stream`.
- capymoa 0.9.0 imports torch unconditionally; the CPU build satisfies it.
- Changing a detector's MOA Option mid-stream does nothing: thresholds are cached in
  private Java fields (STEPD/RDDM read options only in `prepareForUse`; HDDM_W/SEED
  re-read them on `resetLearning`). `detectors.set_threshold` sets Option **and** field,
  which survives every reset path (`tests/test_set_threshold.py`).
