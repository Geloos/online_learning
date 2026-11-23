"""Configuration of the extended-DDD experiment. Every hyperparameter of this
experiment lives here; values shared with the detector study (generators,
warm-up, delay window, vote/sensitivity settings) are re-exported from
drift_ensemble.config so the two experiments cannot silently diverge.

Importing this module imports drift_ensemble.config first, which sets
JAVA_HOME (needed by capymoa) and puts the repo root (home of the shared
streams.py) on sys.path.
"""

from drift_ensemble import config as base

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = base.PROJECT_ROOT / "results" / "results_ensemble"  # run outcomes
RUNS_DIR = RESULTS_DIR / "runs"
# Statistics computed from the runs (analysis.py) live outside results/.
ANALYSIS_DIR = base.PROJECT_ROOT / "statistics" / "ensemble_experiment"
# Detector params tuned by the drift-ensemble experiment
# (drift_ensemble/scripts/03_select_params.py).
SELECTED_PARAMS_PATH = base.TUNING_DIR / "selected_params.json"

# ---------------------------------------------------------------------------
# Extended DDD (Minku & Yao 2011 + drift-type-aware adaptive cap + pool)
# ---------------------------------------------------------------------------
DDD = {
    "n_learners": 10,          # trees per ensemble
    "lambda_low": 1.0,         # Poisson lambda, low diversity
    "lambda_high": 0.05,       # Poisson lambda, high diversity
    "grace_period": 50,        # Hoeffding tree grace period
    "leaf_prediction": "NaiveBayesAdaptive",
    "W": 1.0,                  # multiplier on the old-low vote weight
    "min_steps": 40,           # no exit / classification before this
    "ref_window": 200,         # sliding window for acc_ref of the main model
    "tau": 0.10,               # accuracy drop > tau -> "fast"
    "cap_init": 500,           # cap until a kind has enough history
    "cap_floor": 100,
    "cap_ceil": 2000,
    "cap_margin": 1.5,         # cap = margin * p90(natural exit times)
    "cap_growth": 1.25,        # cap *= growth on a forced exit
    "cap_history": 10,         # natural exit times kept per kind
    "cap_min_history": 3,      # natural exits needed before learning the cap
    "pool_size": 5,            # max frozen ensembles in the concept pool
    "forced_exit": True,       # False -> plain DDD (no cap)
}

# ---------------------------------------------------------------------------
# Streams (shared streams.py): the drift-ensemble experiment's generators,
# shorter streams (DDD runs live), plus the RECURRING mode.
# ---------------------------------------------------------------------------
GENERATORS = list(base.GENERATORS)
MODES = ["ABRUPT", "GRADUAL", "MIXED", "RECURRING"]
RECURRING_N_CONCEPTS = 3  # concepts revisited in a RECURRING stream
STREAM_SIZES = {  # streams.plan_stream keyword arguments
    "n_drifts": 8,
    "drift_every_n": 5_000,
    "gradual_width": 1_000,
    "n_concepts": RECURRING_N_CONCEPTS,  # used by RECURRING only
}

# ---------------------------------------------------------------------------
# Seeds (disjoint from the detector study's tuning and test seeds)
# ---------------------------------------------------------------------------
SEEDS = list(range(201, 211))
assert not set(SEEDS) & set(base.TUNING_SEEDS), "seeds overlap tuning seeds"
assert not set(SEEDS) & set(base.TEST_SEEDS), "seeds overlap test seeds"

# ---------------------------------------------------------------------------
# Detection: the error stream of DDD's new-low ensemble is watched by one of
# the detector study's 7 methods. There is no ens_raw: DDD resets the whole
# signal on every alarm (the watched model is replaced), which would make it
# identical to ens_raw_resetall (it was dropped from experiment 1 too).
# ---------------------------------------------------------------------------
METHODS = [
    "single_STEPD",
    "single_SEED",
    "single_HDDMWeighted",
    "single_RDDM",
    "ens_raw_resetall",
    "ens_cluster_vote",
    "ens_cluster_sens",
]
MIN_TRAINING_SIZE = base.MIN_TRAINING_SIZE  # detector warm-up (instances)
MAX_DELAY = base.MAX_DELAY  # alarm <-> drift matching window
DUPLICATES_AS_FP = base.DUPLICATES_AS_FP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_EVERY = 100  # steps.csv keeps every LOG_EVERY-th step

# ---------------------------------------------------------------------------
# Analysis only (analysis.py): recovery after a detected real drift = how
# many instances after the detection the accuracy counted from the detection
# first reaches `level` x the accuracy of the stable period before the drift.
# It is checked only after `min_instances` instances (after 1 instance the
# accuracy is 0% or 100%).
# ---------------------------------------------------------------------------
RECOVERY = {"level": 0.95, "min_instances": 50}

# Statistics are reported for one method: the one with the best DDD
# accuracy. acc_block = the plots show one point per block of this many
# instances; MIXED and RECURRING plots are aligned on each drift, from `pre`
# before to `post` after its start (their drift positions differ per seed).
STATS = {"method": "ens_raw_resetall", "acc_block": 200,
         "pre": 1000, "post": 5000}


def snapshot() -> dict:
    """The values that define a run, stored in each run's meta.json."""
    import streams

    return {
        "DDD": DDD,
        "STREAM_SIZES": STREAM_SIZES,
        "MIN_TRAINING_SIZE": MIN_TRAINING_SIZE,
        "MAX_DELAY": MAX_DELAY,
        "DUPLICATES_AS_FP": DUPLICATES_AS_FP,
        "INIT_FUNCTION": streams.INIT_FUNCTION,
        "CLUSTER_MEMBERS": base.CLUSTER_MEMBERS,
        "DETECTOR_ORDER": base.DETECTOR_ORDER,
        "VOTE": base.VOTE,
        "SENS": base.SENS,
        "LOG_EVERY": LOG_EVERY,
    }
