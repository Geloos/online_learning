"""Streams of the DDD experiment.

ABRUPT / GRADUAL / MIXED are the detector study's streams
(drift_ensemble.streams), shortened with config.STREAM_OVERRIDES. RECURRING
is new: a few concepts (classification functions of one generator) are
revisited. A repeated concept uses the SAME function with a FRESH instance
seed, so the decision boundary recurs but the samples do not repeat.

RECURRING RNG consumption order for seed s (np.random.default_rng(s)):
  1. permutation of the half-abrupt / half-gradual width list,
  2. the concept set (n_concepts functions, without replacement); the
     stream starts with the first of them,
  3. instance seed of segment 0,
  4. per drift i: next concept (uniform over the set minus the current
     one), then instance seed of segment i+1.
Drift positions follow drift_ensemble.streams: pos += drift_every_n, the
drift occupies [pos, pos + width], then pos += width.
"""

from __future__ import annotations

import numpy as np

from ensemble_experiment import config
from drift_ensemble import streams as base_streams


def run_specs(generators=None, modes=None, seeds=None, methods=None):
    """All (generator, mode, seed, method) runs, in a fixed order."""
    generators = config.GENERATORS if generators is None else generators
    modes = config.MODES if modes is None else modes
    seeds = config.SEEDS if seeds is None else seeds
    methods = config.METHODS if methods is None else methods
    return [(g, m, s, meth) for g in generators for m in modes
            for s in seeds for meth in methods]


def plan_recurring(
    generator: str,
    seed: int,
    *,
    n_drifts: int,
    drift_every_n: int,
    gradual_width: int,
    n_concepts: int | None = None,
) -> dict:
    """Deterministic RECURRING plan, same shape as plan_stream's."""
    if generator not in config.FUNCTIONS:
        raise ValueError(f"unknown generator: {generator!r}")
    n_concepts = (config.RECURRING_N_CONCEPTS if n_concepts is None
                  else n_concepts)
    available = config.FUNCTIONS[generator]
    if not 2 <= n_concepts <= len(available):
        raise ValueError(f"{generator} has {len(available)} functions; "
                         f"cannot recur over {n_concepts}")

    rng = np.random.default_rng(seed)
    widths = base_streams._make_widths("MIXED", n_drifts, gradual_width, rng)
    concepts = [int(f) for f in rng.choice(available, size=n_concepts,
                                           replace=False)]
    functions = [concepts[0]]
    instance_seeds = [int(rng.integers(0, 2**31 - 1))]
    for _ in range(n_drifts):
        candidates = [c for c in concepts if c != functions[-1]]
        functions.append(int(rng.choice(candidates)))
        instance_seeds.append(int(rng.integers(0, 2**31 - 1)))

    ground_truth = []
    drifts = []
    pos = 0
    for i, w in enumerate(widths):
        pos += drift_every_n
        kind = "ABRUPT" if w == 0 else "GRADUAL"
        drifts.append((kind, pos, w))
        ground_truth.append({
            "start": pos,
            "end": pos + w,
            "type": kind,
            "function_before": functions[i],
            "function_after": functions[i + 1],
        })
        pos += w

    length = n_drifts * drift_every_n + sum(widths) + drift_every_n
    return {
        "generator": generator,
        "mode": "RECURRING",
        "seed": seed,
        "widths": widths,
        "concepts": concepts,
        "functions": functions,
        "instance_seeds": instance_seeds,
        "drifts": drifts,
        "ground_truth": ground_truth,
        "length": length,
    }


def plan_for(generator: str, mode: str, seed: int,
             overrides: dict | None = None) -> dict:
    overrides = config.STREAM_OVERRIDES if overrides is None else overrides
    if mode == "RECURRING":
        return plan_recurring(generator, seed, **overrides)
    return base_streams.plan_stream(generator, mode, seed, **overrides)


def mark_recurring(plan: dict) -> list[dict]:
    """Ground truth with a 'recurring' flag per drift: True if the concept
    the drift leads to was already seen earlier in the stream."""
    seen = {plan["functions"][0]}
    out = []
    for d in plan["ground_truth"]:
        out.append({**d, "recurring": d["function_after"] in seen})
        seen.add(d["function_after"])
    return out


def concept_ids(plan: dict) -> np.ndarray:
    """True concept (function) per step. Abrupt drifts switch at their
    start; gradual drifts at their center (where both concepts are equally
    likely)."""
    ids = np.full(plan["length"], plan["functions"][0], dtype=np.int16)
    for d in plan["ground_truth"]:
        switch = d["start"] + (d["end"] - d["start"]) // 2
        ids[switch:] = d["function_after"]
    return ids


def make_experiment_stream(generator: str, mode: str, seed: int,
                           overrides: dict | None = None):
    """(capymoa DriftStream, plan) for a spec of this experiment."""
    plan = plan_for(generator, mode, seed, overrides)
    return base_streams.stream_from_plan(plan), plan
