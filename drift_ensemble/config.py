"""Central configuration. Every hyperparameter of the study lives here.

Importing this module also sets JAVA_HOME (needed by capymoa) — modules that
import capymoa must import drift_ensemble.config first.
"""

from pathlib import Path

from drift_ensemble._java import ensure_java_home

ensure_java_home()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
TUNING_DIR = RESULTS_DIR / "tuning"
TEST_DIR = RESULTS_DIR / "test"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
GENERATORS = ["Agrawal", "SEA", "STAGGER"]
FUNCTIONS = {
    "SEA": [1, 2, 3, 4],
    "Agrawal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "STAGGER": [1, 2, 3],
}
N_DRIFTS = 30
DRIFT_EVERY_N = 10_000
GRADUAL_WIDTH = 2_000
MIXED_SPLIT = (15, 15)  # (abrupt, gradual) drifts in a MIXED stream
# All 3 generators (15 MIXED test streams, was Agrawal only = 5): with 5
# pairs Wilcoxon cannot reach p < 0.0625, so MIXED could never be significant.
MIXED_GENERATORS = ["Agrawal", "SEA", "STAGGER"]
INIT_FUNCTION = 1
MODES = ["ABRUPT", "GRADUAL", "MIXED"]

# ---------------------------------------------------------------------------
# Seeds (tuning and test seeds must never overlap — asserted below)
# ---------------------------------------------------------------------------
TUNING_SEEDS = [1, 2, 3]
TEST_SEEDS = [101, 102, 103, 104, 105]
assert not set(TUNING_SEEDS) & set(TEST_SEEDS), "tuning and test seeds overlap"

# ---------------------------------------------------------------------------
# Learner / detection
# ---------------------------------------------------------------------------
MIN_TRAINING_SIZE = 1000  # detectors (and HT error logging) start after this
MAX_DELAY = 500
DELAY_GRID = list(range(50, 501, 50))  # [50, 100, ..., 500]
DUPLICATES_AS_FP = True  # extra alarms inside an already-matched window = FP

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
TUNE_OBJECTIVE = "f1"  # options: 'f1', 'f2', 'delay_auc'
SEED_RANDOM_CONFIGS = 100  # SEED grid has 700 combos -> sample 100 (seeded)
GRID_SAMPLE_SEED = 20240  # rng seed used to subsample oversized grids

# Grids follow Cerqueira et al. (drift detector benchmark, CapyMOA).
# Keys use the paper's parameter names; drift_ensemble.detectors translates
# them to capymoa constructor kwargs.
TUNING_GRIDS = {
    "STEPD": {  # 36 combos, full grid
        "window_size": [30, 50, 100, 300, 500, 1000],
        "alpha_drift": [0.001, 0.002, 0.003, 0.005, 0.01, 0.1],
    },
    "SEED": {  # 700 combos -> SEED_RANDOM_CONFIGS sampled (seeded)
        "delta": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "epsilon_prime": [0.0025, 0.005, 0.0075, 0.01],
        "block_size": [32, 50, 100, 150, 256],
        "alpha": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    },
    "HDDMWeighted": {  # 50 combos, full grid
        "drift_confidence": [0.0001, 0.001, 0.002, 0.005, 0.01],
        "lambda_": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "test_type": ["One-sided", "Two-sided"],
    },
    "RDDM": {  # 56 combos, full grid
        "min_n_instances": [30, 50, 100, 300, 500, 1000, 2000],
        "drift_level": [1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 3.0],
    },
}

# ---------------------------------------------------------------------------
# Clusters (membership justified by results/tuning/tuning_summary.csv)
# Abrupt-cluster detectors run with their ABRUPT-tuned params, gradual-cluster
# detectors with their GRADUAL-tuned params (see selected_params.json).
#
# Assignment chosen from the tuning results (each detector goes to the mode
# where its tuned config reached the higher mean F1 over the 9 tuning
# streams of that mode; tuning_summary.csv, 2026-11-12):
#   STEPD        F1 0.427 ABRUPT vs 0.474 GRADUAL -> GRADUAL
#   SEED         F1 0.425 ABRUPT vs 0.417 GRADUAL -> ABRUPT
#   HDDMWeighted F1 0.571 ABRUPT vs 0.691 GRADUAL -> GRADUAL
#   RDDM         F1 0.580 ABRUPT vs 0.533 GRADUAL -> ABRUPT
# (The spec's provisional {ABRUPT: STEPD+SEED, GRADUAL: HDDM+RDDM} was only
# half-supported by the data; user picked the evidence-based split.)
# ---------------------------------------------------------------------------
CLUSTER_MEMBERS = {
    "ABRUPT": ["SEED", "RDDM"],
    "GRADUAL": ["STEPD", "HDDMWeighted"],
}
# Fixed tie-break order for choosing the trigger when several detectors fire
# on the same instance.
DETECTOR_ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]

# ---------------------------------------------------------------------------
# Verification (chosen by the cluster of the TRIGGER detector)
# N = training samples per NB model, D = gap skipped before the alarm,
# E = unseen test samples.
# ---------------------------------------------------------------------------
VERIFY = {
    "ABRUPT": {"N": 250, "D": 250, "E": 250, "alpha": 0.05},
    "GRADUAL": {"N": 500, "D": 500, "E": 500, "alpha": 0.05},
}

# ---------------------------------------------------------------------------
# Cluster-vote ensemble (ens_cluster_vote): a drift needs two votes from the
# same cluster within WINDOW instances (or one vote from a detector whose
# weight reached W_MAX). The earlier voter of a pair gains STEP scaled by how
# much earlier it fired; the later one loses the same amount. Weights live in
# [W_FLOOR, W_MAX].
# ---------------------------------------------------------------------------
VOTE = {
    "WINDOW": 250,
    "W_INIT": 0.75,
    "W_MAX": 1.0,
    "W_FLOOR": 0.5,
    "STEP": 0.1,
}

# ---------------------------------------------------------------------------
# Cluster-sensitivity ensemble (ens_cluster_sens): same alarm rule as
# ens_cluster_vote (two same-cluster votes within WINDOW -> alarm, reset all
# 4) but no weights. Instead each detector carries a sensitivity LEVEL
# (0 = its tuned params) that moves its own drift threshold mid-stream: the
# LATER voter of a confirming pair gains LATE_STEP * gap / WINDOW levels
# (more sensitive; a same-instant tie changes nothing). Levels only ever
# rise (no stricter rule — dropped 2026-11-17, it drove every detector to
# the strictest level and cut recall) and are capped at L_MAX. SCALE maps
# one level to a threshold change per detector: "mul" -> base *
# factor**level (p-value-like thresholds, capped at PROB_MAX), "add" ->
# base + step * level (RDDM's drift_level in std units, floored at
# RDDM_LEVEL_MIN). Positive level is always the more sensitive direction.
# ---------------------------------------------------------------------------
SENS = {
    "WINDOW": 250,
    "LATE_STEP": 1.0,
    "L_MAX": 3.0,
    "SCALE": {
        "STEPD": ("alpha_drift", "mul", 2.0),
        "SEED": ("delta", "mul", 2.0),
        "HDDMWeighted": ("drift_confidence", "mul", 2.0),
        "RDDM": ("drift_level", "add", -0.15),
    },
    "PROB_MAX": 0.5,
    "RDDM_LEVEL_MIN": 1.0,
}

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
W_INIT = 1.0
BETA = 0.5
W_MIN = 0.5  # muted when w <= W_MIN (one strike with BETA = 0.5)

# Phase 0.4 probe result (capymoa 0.9.0, fed 2000x0, 500x1, 2000x0, 500x1):
# every detector fired again after its first detection WITHOUT an explicit
# reset() call, i.e. all four restart their own statistics after
# detected_change() is True. So "reset only the signaling detector" already
# happens by itself in every method; the *_resetall methods add the explicit
# reset of ALL detectors (via ensemble.reset_detector).
DETECTOR_SELF_RESETS = {
    "STEPD": True,
    "SEED": True,
    "HDDMWeighted": True,
    "RDDM": True,
}

# ---------------------------------------------------------------------------
# Test-run methods
# ---------------------------------------------------------------------------
METHODS = [
    "single_STEPD",
    "single_SEED",
    "single_HDDMWeighted",
    "single_RDDM",
    "ens_raw",
    "ens_raw_resetall",
    "ens_cluster_vote",
    "ens_cluster_sens",
]

def snapshot() -> dict:
    """The config values that define an experiment, for meta.json files."""
    return {
        "N_DRIFTS": N_DRIFTS,
        "DRIFT_EVERY_N": DRIFT_EVERY_N,
        "GRADUAL_WIDTH": GRADUAL_WIDTH,
        "MIXED_SPLIT": list(MIXED_SPLIT),
        "INIT_FUNCTION": INIT_FUNCTION,
        "MIN_TRAINING_SIZE": MIN_TRAINING_SIZE,
        "MAX_DELAY": MAX_DELAY,
        "DELAY_GRID": DELAY_GRID,
        "DUPLICATES_AS_FP": DUPLICATES_AS_FP,
        "TUNE_OBJECTIVE": TUNE_OBJECTIVE,
        "SEED_RANDOM_CONFIGS": SEED_RANDOM_CONFIGS,
        "GRID_SAMPLE_SEED": GRID_SAMPLE_SEED,
        "CLUSTER_MEMBERS": CLUSTER_MEMBERS,
        "DETECTOR_ORDER": DETECTOR_ORDER,
        "VERIFY": VERIFY,
        "VOTE": VOTE,
        "SENS": SENS,
        "W_INIT": W_INIT,
        "BETA": BETA,
        "W_MIN": W_MIN,
        "USE_CACHED_ERRORS_FOR_TEST_RUNS": USE_CACHED_ERRORS_FOR_TEST_RUNS,
    }


# Engineering choice (validated by tests/test_cache_replay.py): HT is
# deterministic given the instance sequence, so test runs read the 0/1 error
# sequence from the per-stream cache instead of re-running HT inside every
# method. Verified-ensemble runs still iterate the real stream for the NB
# verification models.
USE_CACHED_ERRORS_FOR_TEST_RUNS = True
