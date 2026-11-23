"""Synthetic drift streams shared by both experiments.

``plan_stream`` is pure Python (testable without a JVM): it fixes the drift
widths, the concept (classification function) of every segment, the
per-segment instance seeds and the ground truth. ``stream_from_plan`` /
``make_stream`` turn a plan into a capymoa DriftStream. Stream SIZES are
not defined here: each experiment's config passes its own.

Modes
  ABRUPT     every drift switches instantly (width 0)
  GRADUAL    every drift blends over ``gradual_width`` instances
  MIXED      half abrupt, half gradual, in random order
  RECURRING  like MIXED, but the segments cycle through ``n_concepts``
             functions, so concepts come back. A returning concept uses the
             SAME function with a FRESH instance seed: the decision boundary
             recurs, the samples do not.

RNG consumption order for seed s (np.random.default_rng(s)) — fixed so a
seed always reproduces the same stream:
  1. MIXED / RECURRING: permutation of the width list,
  2. RECURRING only: the concept set (n_concepts functions, no
     replacement); the stream starts with the first of them,
  3. instance seed of segment 0,
  4. per drift i: next function (uniform over the candidates other than
     the current one: the generator's other functions, or for RECURRING
     the other members of the concept set), then instance seed of
     segment i+1.

Drift positions: pos += drift_every_n, the drift occupies
[pos, pos + width] (width 0 for abrupt), then pos += width. capymoa's
GradualDrift is built with start/end (its ``position`` kwarg means the
center, verified on capymoa 0.9.0).
"""

from __future__ import annotations

import numpy as np

# Classification functions of each generator (capymoa 0.9.0).
FUNCTIONS = {
    "SEA": [1, 2, 3, 4],
    "Agrawal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "STAGGER": [1, 2, 3],
}
INIT_FUNCTION = 1  # first concept of ABRUPT / GRADUAL / MIXED streams
MODES = ("ABRUPT", "GRADUAL", "MIXED", "RECURRING")


def stream_id(generator: str, mode: str, seed: int) -> str:
    return f"{generator},{mode},{seed}"


def parse_stream_id(sid: str) -> tuple[str, str, int]:
    generator, mode, seed = sid.split(",")
    return generator, mode, int(seed)


def _make_widths(mode: str, n_drifts: int, gradual_width: int,
                 rng: np.random.Generator) -> list[int]:
    if mode == "ABRUPT":
        return [0] * n_drifts
    if mode == "GRADUAL":
        return [gradual_width] * n_drifts
    if mode in ("MIXED", "RECURRING"):
        n_abrupt = n_drifts // 2
        widths = [0] * n_abrupt + [gradual_width] * (n_drifts - n_abrupt)
        return [int(w) for w in rng.permutation(widths)]
    raise ValueError(f"unknown mode: {mode!r}")


def plan_stream(
    generator: str,
    mode: str,
    seed: int,
    *,
    n_drifts: int,
    drift_every_n: int,
    gradual_width: int,
    n_concepts: int | None = None,
) -> dict:
    """Deterministic plan: concepts, drifts, ground truth and length.
    ``n_concepts`` is required for RECURRING and ignored otherwise."""
    if generator not in FUNCTIONS:
        raise ValueError(f"unknown generator: {generator!r}")
    available = FUNCTIONS[generator]

    rng = np.random.default_rng(seed)
    widths = _make_widths(mode, n_drifts, gradual_width, rng)

    if mode == "RECURRING":
        if n_concepts is None or not 2 <= n_concepts <= len(available):
            raise ValueError(f"{generator} has {len(available)} functions; "
                             f"cannot recur over {n_concepts}")
        concepts = [int(f) for f in rng.choice(available, size=n_concepts,
                                               replace=False)]
    else:
        concepts = list(available)

    functions = [concepts[0] if mode == "RECURRING" else INIT_FUNCTION]
    instance_seeds = [int(rng.integers(0, 2**31 - 1))]
    for _ in range(n_drifts):
        candidates = [f for f in concepts if f != functions[-1]]
        functions.append(int(rng.choice(candidates)))
        instance_seeds.append(int(rng.integers(0, 2**31 - 1)))

    ground_truth = []
    drifts = []  # (kind, start, width) for the stream builder
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

    plan = {
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "widths": widths,
        "functions": functions,
        "instance_seeds": instance_seeds,
        "drifts": drifts,
        "ground_truth": ground_truth,
        "length": n_drifts * drift_every_n + sum(widths) + drift_every_n,
    }
    if mode == "RECURRING":
        plan["concepts"] = concepts
    return plan


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
    """True concept (function) per instance. Abrupt drifts switch at their
    start, gradual drifts at their center (both concepts equally likely)."""
    ids = np.full(plan["length"], plan["functions"][0], dtype=np.int16)
    for d in plan["ground_truth"]:
        switch = d["start"] + (d["end"] - d["start"]) // 2
        ids[switch:] = d["function_after"]
    return ids


def _make_concept(generator: str, function: int, instance_seed: int):
    from capymoa.stream.generator import (
        SEA,
        AgrawalGenerator,
        STAGGERGenerator,
    )

    if generator == "SEA":
        return SEA(instance_random_seed=instance_seed, function=function)
    if generator == "Agrawal":
        return AgrawalGenerator(instance_random_seed=instance_seed,
                                classification_function=function)
    if generator == "STAGGER":
        return STAGGERGenerator(instance_random_seed=instance_seed,
                                classification_function=function)
    raise ValueError(f"unknown generator: {generator!r}")


def stream_from_plan(plan: dict):
    """Materialize a plan as a capymoa DriftStream."""
    import contextlib
    import io

    # capymoa needs JAVA_HOME before its first import.
    from drift_ensemble.config import ensure_java_home

    ensure_java_home()
    from capymoa.stream.drift import AbruptDrift, DriftStream, GradualDrift

    generator = plan["generator"]
    # capymoa 0.9.0's GradualDrift constructor prints a stray "None".
    with contextlib.redirect_stdout(io.StringIO()):
        parts = [_make_concept(generator, plan["functions"][0],
                               plan["instance_seeds"][0])]
        for i, (kind, start, width) in enumerate(plan["drifts"]):
            if kind == "ABRUPT":
                parts.append(AbruptDrift(position=start))
            else:
                parts.append(GradualDrift(start=start, end=start + width))
            parts.append(_make_concept(generator, plan["functions"][i + 1],
                                       plan["instance_seeds"][i + 1]))
        stream = DriftStream(stream=parts)
    return stream


def make_stream(generator: str, mode: str, seed: int, **sizes):
    """(capymoa DriftStream, ground_truth, plan); ``sizes`` are the
    plan_stream keyword arguments."""
    plan = plan_stream(generator, mode, seed, **sizes)
    return stream_from_plan(plan), plan["ground_truth"], plan
