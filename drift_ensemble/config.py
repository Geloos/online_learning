"""Configuration of the drift-ensemble experiment. Every hyperparameter of
this study lives here (the stream generators themselves are defined in the
shared root module streams.py).

Importing this module also sets JAVA_HOME (needed by capymoa; see
ensure_java_home below) — modules that
import capymoa must import drift_ensemble.config first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ===========================================================================
# JAVA_HOME
#
# JAVA_HOME bootstrap.
#
# CapyMOA runs MOA through a JVM (via jpype) and needs JAVA_HOME. On this
# machine java is on PATH but JAVA_HOME is not set, so we derive it from the
# JVM's own ``java.home`` property before any capymoa import happens.
#
# Import order contract: every module that (directly or indirectly) imports
# capymoa must import ``drift_ensemble.config`` first, which calls
# ``ensure_java_home()`` at import time.
# ===========================================================================
def ensure_java_home() -> str:
    """Set os.environ['JAVA_HOME'] from the running JVM if it is unset."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        return java_home
    try:
        proc = subprocess.run(
            ["java", "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "JAVA_HOME is not set and 'java' was not found on PATH. "
            "Install a JDK 11+ or set JAVA_HOME."
        ) from exc
    # -XshowSettings prints the properties on stderr.
    for line in proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("java.home"):
            java_home = line.split("=", 1)[1].strip()
            os.environ["JAVA_HOME"] = java_home
            return java_home
    raise RuntimeError(
        "Could not derive JAVA_HOME from 'java -XshowSettings:properties'. "
        "Set JAVA_HOME manually."
    )

ensure_java_home()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # the shared streams.py lives there
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results" / "results_drift_ensemble"
CACHE_DIR = RESULTS_DIR / "cache"
TUNING_DIR = RESULTS_DIR / "tuning"
TEST_DIR = RESULTS_DIR / "test"
# results/ holds only run outcomes (cache, tuning, test runs, screening
# rows). Everything computed from them (stats.py: tables, bar charts, delay
# curves, tuning heatmaps) lives under statistics/.
STATISTICS_DIR = PROJECT_ROOT / "statistics" / "drift_ensemble"

# ---------------------------------------------------------------------------
# Streams (built by the shared streams.py)
# ---------------------------------------------------------------------------
GENERATORS = ["Agrawal", "SEA", "STAGGER"]
N_DRIFTS = 30  # MIXED streams: half abrupt, half gradual (15 / 15)
DRIFT_EVERY_N = 10_000
GRADUAL_WIDTH = 2_000
RECURRING_N_CONCEPTS = 3  # concepts cycled by a RECURRING stream
STREAM_SIZES = {
    "n_drifts": N_DRIFTS,
    "drift_every_n": DRIFT_EVERY_N,
    "gradual_width": GRADUAL_WIDTH,
    # Used by RECURRING only; plan_stream ignores it for the other modes, so
    # adding it left every earlier plan and cached stream unchanged.
    "n_concepts": RECURRING_N_CONCEPTS,
}
# All 3 generators (15 MIXED test streams, was Agrawal only = 5): with 5
# pairs Wilcoxon cannot reach p < 0.0625, so MIXED could never be significant.
MIXED_GENERATORS = ["Agrawal", "SEA", "STAGGER"]
# RECURRING (added 2026-11-20): like MIXED, but cycling through
# RECURRING_N_CONCEPTS concepts. Test-only, like MIXED.
RECURRING_GENERATORS = ["Agrawal", "SEA", "STAGGER"]
MODES = ["ABRUPT", "GRADUAL", "MIXED", "RECURRING"]
PURE_MODES = ("ABRUPT", "GRADUAL")  # tuning uses the pure modes only

# ---------------------------------------------------------------------------
# Seeds (tuning and test seeds must never overlap — asserted below)
# ---------------------------------------------------------------------------
TUNING_SEEDS = [1, 2, 3]
TEST_SEEDS = [101, 102, 103, 104, 105]
assert not set(TUNING_SEEDS) & set(TEST_SEEDS), "tuning and test seeds overlap"


def tuning_stream_specs() -> list[tuple[str, str, int]]:
    return [(g, mode, s) for g in GENERATORS for mode in PURE_MODES
            for s in TUNING_SEEDS]


def test_stream_specs() -> list[tuple[str, str, int]]:
    pure = [(g, mode, s) for g in GENERATORS for mode in PURE_MODES
            for s in TEST_SEEDS]
    mixed = [(g, "MIXED", s) for g in MIXED_GENERATORS for s in TEST_SEEDS]
    recurring = [(g, "RECURRING", s) for g in RECURRING_GENERATORS
                 for s in TEST_SEEDS]
    return pure + mixed + recurring

# ---------------------------------------------------------------------------
# Learner / detection
# ---------------------------------------------------------------------------
MIN_TRAINING_SIZE = 1000  # detectors (and HT error logging) start after this
MAX_DELAY = 500
DELAY_GRID = list(range(50, 501, 50))  # [50, 100, ..., 500]
DUPLICATES_AS_FP = True  # extra alarms inside an already-matched window = FP

# Statistics (drift_ensemble/stats.py -> STATISTICS_DIR): medians per
# method x mode, and the delay curves rescore every run for each max delay
# in delay_grid (0 = alarm must fall inside the drift itself).
STATS = {"delay_grid": list(range(0, 501, 25))}

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

# capymoa 0.9.0's constructor defaults for the tuned parameters (grid names).
# Every grid (tuning and screening) also evaluates this config
# (detectors.iter_configs appends it when the grid misses it: SEED if not
# sampled, RDDM always), so "the recommended settings" are scored too.
DEFAULT_PARAMS = {
    "STEPD": {"window_size": 30, "alpha_drift": 0.003},
    "SEED": {"delta": 0.05, "epsilon_prime": 0.01, "block_size": 32,
             "alpha": 0.8},
    "HDDMWeighted": {"drift_confidence": 0.001, "lambda_": 0.05,
                     "test_type": "Two-sided"},
    "RDDM": {"min_n_instances": 129, "drift_level": 2.258},
    "DDM": {"min_n_instances": 30, "out_control_level": 3.0},
    "HDDMAverage": {"drift_confidence": 0.001, "test_type": "Two-sided"},
    "ADWIN": {"delta": 0.002},
}

# ---------------------------------------------------------------------------
# Tuning phase 2: a shared minimum per cluster
# (tuning.py, `python main.py phase2`).
#
# Each detector has its own minimum number of instances after a reset
# before it can fire (measured by probe on 2026-11-23, 0s then 1s right
# after a reset): STEPD 2 x window_size, SEED 2 x block_size, RDDM
# min_n_instances, HDDMWeighted none (a statistical test only). With the
# phase-1 params the two detectors of a cluster therefore switched back on
# at different times after every alarm (e.g. RDDM 2000 vs HDDMWeighted
# ~230).
#
# Phase 2 gives each cluster ONE minimum M (M_A for ABRUPT, M_G for
# GRADUAL, tuned separately; equal only if tuning says so):
#   - a gate: after a reset (stream start, reset-all) or after an accepted
#     fire, a detector's fires are ignored until it has seen M instances.
#     The detector keeps updating its statistics meanwhile.
#   - RDDM's min_n_instances is set to M; STEPD and SEED may only use
#     configs whose own minimum is <= M (so the gate decides when they
#     start, not their params). HDDMWeighted can use every config.
# For each M in the grid, the cluster's two detectors are re-tuned on the
# cluster's own mode (ABRUPT or GRADUAL tuning streams, phase-1 grids).
# The cluster's score at M is the mean of its two detectors' best mean
# TUNE_OBJECTIVE; the M with the highest score wins. Output:
# tuning/cluster_params.json, used by the two-vote methods only
# (PHASE2_METHODS: ens_cluster_vote, ens_cluster_sens) in both experiments.
# ---------------------------------------------------------------------------
MIN_INSTANCES_GRID = [50, 100, 200, 300, 500, 1000, 2000]
CLUSTER_PARAMS_PATH = TUNING_DIR / "cluster_params.json"

# ---------------------------------------------------------------------------
# Detector screening (`python main.py screen`, tuning.py): extra
# capymoa detectors tuned and tested as singles on the CACHED errors only, to
# see whether one beats the study's 4 on ABRUPT / GRADUAL F1. They are NOT
# in the roster, the clusters or the ensembles. These grids are our own
# ranges around capymoa's defaults (not from Cerqueira et al.). Only the
# drift threshold matters here: warning levels are left at their defaults.
# ---------------------------------------------------------------------------
SCREENING_DIR = RESULTS_DIR / "screening"
SCREENING_GRIDS = {
    "DDM": {  # 49 combos
        "min_n_instances": [30, 50, 100, 300, 500, 1000, 2000],
        "out_control_level": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    },
    "HDDMAverage": {  # 16 combos
        "drift_confidence": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
                             0.05, 0.1],
        "test_type": ["One-sided", "Two-sided"],
    },
    "ADWIN": {  # 10 combos
        "delta": [1e-6, 1e-5, 1e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.05,
                  0.1],
    },
}

# ---------------------------------------------------------------------------
# Clusters (membership justified by tuning/tuning_summary.csv)
# Abrupt-cluster detectors run with their ABRUPT-tuned params, gradual-cluster
# detectors with their GRADUAL-tuned params (see selected_params.json).
#
# Assignment (2026-11-20, tuning with MAX_DELAY = 500): a detector always
# runs with its cluster's params, so it goes to the cluster whose params hurt
# it least on the other stream type. Median test F1 (ABRUPT / GRADUAL
# streams) with the ABRUPT-tuned vs the GRADUAL-tuned params
# (statistics/drift_ensemble/screening.csv, results/.../screening/):
#   HDDMWeighted 0.596 / 0.536  vs  0.074 / 0.553  -> ABRUPT
#   STEPD        0.345 / 0.383  vs  0.178 / 0.516  -> GRADUAL
#   SEED         0.317 / 0.441  vs  0.328 / 0.438  -> GRADUAL (no difference)
#   RDDM         same params for both modes        -> ABRUPT (2nd-best
#                detector on ABRUPT, tuning F1 0.476)
# The earlier rule ("the mode with the higher tuning F1") no longer works:
# with the 500 window every detector scores higher on GRADUAL (a gradual
# drift's window is GRADUAL_WIDTH + MAX_DELAY wide).
# ---------------------------------------------------------------------------
CLUSTER_MEMBERS = {
    "ABRUPT": ["HDDMWeighted", "RDDM"],
    "GRADUAL": ["STEPD", "SEED"],
}
# Fixed tie-break order for choosing the trigger when several detectors fire
# on the same instance.
DETECTOR_ORDER = ["STEPD", "SEED", "HDDMWeighted", "RDDM"]

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

# Probe result (capymoa 0.9.0, fed 2000x0, 500x1, 2000x0, 500x1): every
# detector fired again after its first detection WITHOUT an explicit
# reset() call, i.e. all four restart their own statistics after
# detected_change() is True. So "reset only the signaling detector" already
# happens by itself in every method; ens_raw_resetall and the cluster
# ensembles add the explicit reset of ALL detectors (ensemble.reset_detector).
DETECTOR_SELF_RESETS = {
    "STEPD": True,
    "SEED": True,
    "HDDMWeighted": True,
    "RDDM": True,
    # Screening detectors (probe 2026-11-20: 10% -> 60% error twice; each
    # fired again on the second rise without reset(). ADWIN fires a few
    # times per drift as its window shrinks.)
    "DDM": True,
    "HDDMAverage": True,
    "ADWIN": True,
}

# ---------------------------------------------------------------------------
# Test-run methods. ens_raw (OR, each detector only self-resets) was dropped
# on 2026-11-23: DDD (experiment 2) can't use it, so both experiments share
# the same 7 methods.
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
# Which tuning each method runs with (user's choice, 2026-11-23): the two
# methods that need two votes from a cluster use tuning phase 2
# (cluster_params.json: the cluster's shared minimum + its re-tuned
# params). The singles and ens_raw_resetall use phase 1
# (selected_params.json: each detector tuned alone, its cluster's mode
# params, no shared minimum), in both experiments.
PHASE2_METHODS = ["ens_cluster_vote", "ens_cluster_sens"]

def snapshot() -> dict:
    """The config values that define an experiment, for meta.json files."""
    import streams

    return {
        "N_DRIFTS": N_DRIFTS,
        "DRIFT_EVERY_N": DRIFT_EVERY_N,
        "GRADUAL_WIDTH": GRADUAL_WIDTH,
        "RECURRING_N_CONCEPTS": RECURRING_N_CONCEPTS,
        "INIT_FUNCTION": streams.INIT_FUNCTION,
        "MIN_TRAINING_SIZE": MIN_TRAINING_SIZE,
        "MAX_DELAY": MAX_DELAY,
        "DELAY_GRID": DELAY_GRID,
        "DUPLICATES_AS_FP": DUPLICATES_AS_FP,
        "TUNE_OBJECTIVE": TUNE_OBJECTIVE,
        "SEED_RANDOM_CONFIGS": SEED_RANDOM_CONFIGS,
        "GRID_SAMPLE_SEED": GRID_SAMPLE_SEED,
        "MIN_INSTANCES_GRID": MIN_INSTANCES_GRID,
        "PHASE2_METHODS": PHASE2_METHODS,
        "CLUSTER_MEMBERS": CLUSTER_MEMBERS,
        "DETECTOR_ORDER": DETECTOR_ORDER,
        "VOTE": VOTE,
        "SENS": SENS,
    }
