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
GENERATORS = ["Agrawal", "SEA", "Sine"]
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
# All 3 generators (was Agrawal only): with 5 pairs Wilcoxon cannot reach
# p < 0.0625, so MIXED could never be significant.
MIXED_GENERATORS = ["Agrawal", "SEA", "Sine"]
# RECURRING (added 2026-11-20): like MIXED, but cycling through
# RECURRING_N_CONCEPTS concepts. Since 2026-11-24 (user's request) its test
# streams are only generated and cached by the `test` step: no method runs
# on them and they get no statistics (recurring_stream_specs).
RECURRING_GENERATORS = ["Agrawal", "SEA", "Sine"]
MODES = ["ABRUPT", "GRADUAL", "MIXED"]  # modes the methods run and are scored on
PURE_MODES = ("ABRUPT", "GRADUAL")  # tuning uses the pure modes only

# ---------------------------------------------------------------------------
# Seeds (tuning and test seeds must never overlap — asserted below)
# ---------------------------------------------------------------------------
# 17 seeds each (2026-11-24, was 3 and 5): 3 generators x 2 modes x 17 =
# 102 tuning streams; 3 generators x 3 modes x 17 = 153 test streams, plus
# 51 RECURRING streams that are only cached.
TUNING_SEEDS = list(range(1, 18))
TEST_SEEDS = list(range(101, 118))
assert not set(TUNING_SEEDS) & set(TEST_SEEDS), "tuning and test seeds overlap"


def tuning_stream_specs() -> list[tuple[str, str, int]]:
    return [(g, mode, s) for g in GENERATORS for mode in PURE_MODES
            for s in TUNING_SEEDS]


def test_stream_specs() -> list[tuple[str, str, int]]:
    """The test streams the methods run on (ABRUPT, GRADUAL, MIXED)."""
    pure = [(g, mode, s) for g in GENERATORS for mode in PURE_MODES
            for s in TEST_SEEDS]
    mixed = [(g, "MIXED", s) for g in MIXED_GENERATORS for s in TEST_SEEDS]
    return pure + mixed


def recurring_stream_specs() -> list[tuple[str, str, int]]:
    """RECURRING test streams: cached by the `test` step, not run on."""
    return [(g, "RECURRING", s) for g in RECURRING_GENERATORS
            for s in TEST_SEEDS]

# ---------------------------------------------------------------------------
# Learner / detection
# ---------------------------------------------------------------------------
MIN_TRAINING_SIZE = 1000  # detectors (and HT error logging) start after this
MAX_DELAY = 500
DUPLICATES_AS_FP = True  # extra alarms inside an already-matched window = FP

# Statistics (drift_ensemble/stats.py -> STATISTICS_DIR): the delay curves
# rescore every run for each max delay in delay_grid (0 = alarm must fall
# inside the drift itself).
STATS = {"delay_grid": list(range(0, 501, 25))}

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
# The one selection score (2026-11-24, replaced F1): TF2 = F2(precision,
# recall x (1 - NDT)), see stats.py. It picks every detector's params per
# mode AND the clusters.
TUNE_OBJECTIVE = "tf2"
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
    # not in capymoa: the defaults of scikit-multiflow's and River's KSWIN
    "KSWIN": {"alpha": 0.005, "window_size": 100, "stat_size": 30},
}

# ---------------------------------------------------------------------------
# Detector screening (`python main.py screen`, tuning.py): extra
# detectors tuned and tested as singles on the CACHED errors only, to
# see whether one beats the study's 4 on ABRUPT / GRADUAL TF2. They are NOT
# in the roster, the clusters or the ensembles. Only the drift threshold
# matters here: warning levels are left at their defaults.
#   - DDM: our own ranges around capymoa's defaults.
#   - HDDMAverage, ADWIN: the values the user gave on 2026-11-24 (same
#     drift_confidence values as HDDMWeighted's grid); ADWIN also keeps
#     delta 0.1, which won with our earlier, wider grid.
#   - KSWIN (2026-11-24, not in capymoa 0.9.0 or MOA: our own
#     implementation, detectors.KSWIN): our own ranges around the usual
#     defaults. Configs with window_size < 2 * stat_size are left out
#     (the random sample is drawn from the older window_size - stat_size).
# ---------------------------------------------------------------------------
SCREENING_DIR = RESULTS_DIR / "screening"
SCREENING_GRIDS = {
    "DDM": {  # 49 combos
        "min_n_instances": [30, 50, 100, 300, 500, 1000, 2000],
        "out_control_level": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    },
    "HDDMAverage": {  # 10 combos
        "drift_confidence": [0.0001, 0.001, 0.002, 0.005, 0.01],
        "test_type": ["One-sided", "Two-sided"],
    },
    "ADWIN": {  # 7 combos
        "delta": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1],
    },
    "KSWIN": {  # 60 combos, 55 valid
        "alpha": [0.0001, 0.001, 0.005, 0.01, 0.05],
        "window_size": [100, 300, 500, 1000],
        "stat_size": [30, 50, 100],
    },
}

# ---------------------------------------------------------------------------
# Clusters (2026-11-24, user's rule; replaced the hand-made split): the
# `select` step puts the CLUSTER_SIZE detectors with the best mean tuning
# TF2 on ABRUPT streams in the ABRUPT cluster, and the best on GRADUAL
# streams in the GRADUAL cluster (ties: precision, then DETECTOR_ORDER).
# Each member runs with the params tuned for its cluster's mode, so one
# detector can be in BOTH clusters as two members with different params
# (e.g. HDDMWeighted_ABRUPT and HDDMWeighted_GRADUAL). The result is in
# selected_params.json ("clusters"); runs.members() lists the members.
# ---------------------------------------------------------------------------
CLUSTER_SIZE = 2  # the vote ensembles pair the two members of a cluster
# Fixed detector order: tuning ties, member order inside a cluster, and the
# order the ensembles poll their members in.
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
    "KSWIN": True,  # by construction: keeps only stat_size inputs
}

# ---------------------------------------------------------------------------
# Test-run methods, the same 7 in both experiments: one single_<member> per
# cluster member (4, named after the tuning: runs.methods) and these 3
# ensembles over the 4 members. ens_raw (OR, each detector only
# self-resets) was dropped on 2026-11-23: DDD (experiment 2) can't use it.
# ---------------------------------------------------------------------------
ENSEMBLE_METHODS = [
    "ens_raw_resetall",
    "ens_cluster_vote",
    "ens_cluster_sens",
]


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
        "DUPLICATES_AS_FP": DUPLICATES_AS_FP,
        "TUNE_OBJECTIVE": TUNE_OBJECTIVE,
        "SEED_RANDOM_CONFIGS": SEED_RANDOM_CONFIGS,
        "GRID_SAMPLE_SEED": GRID_SAMPLE_SEED,
        "CLUSTER_SIZE": CLUSTER_SIZE,
        "DETECTOR_ORDER": DETECTOR_ORDER,
        "VOTE": VOTE,
        "SENS": SENS,
    }
