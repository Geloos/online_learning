"""Stream factory (pure + mixed drift streams) and ground truth (PHASE 2).

``plan_stream`` is pure Python (fully testable without a JVM): it fixes the
drift widths, the concept functions, the per-concept instance seeds and the
ground truth. ``make_stream`` turns a plan into a capymoa DriftStream.

RNG consumption order for seed s (np.random.default_rng(s)) — fixed so a
seed always reproduces the same stream:
  1. MIXED only: permutation of the width list,
  2. instance seed of concept 0,
  3. per drift i: next function (uniform over the other functions of the
     generator), then instance seed of concept i+1.

Drift positions follow the spec: pos += DRIFT_EVERY_N, the drift occupies
[pos, pos + width] (width 0 for abrupt), then pos += width. capymoa's
GradualDrift is built with start/end (its ``position`` kwarg means the
center, verified on capymoa 0.9.0).
"""

from __future__ import annotations

import numpy as np

from drift_ensemble import config

PURE_MODES = ("ABRUPT", "GRADUAL")


def stream_id(generator: str, mode: str, seed: int) -> str:
    return f"{generator},{mode},{seed}"


def parse_stream_id(sid: str) -> tuple[str, str, int]:
    generator, mode, seed = sid.split(",")
    return generator, mode, int(seed)


def tuning_stream_specs() -> list[tuple[str, str, int]]:
    return [
        (g, mode, s)
        for g in config.GENERATORS
        for mode in PURE_MODES
        for s in config.TUNING_SEEDS
    ]


def test_stream_specs() -> list[tuple[str, str, int]]:
    pure = [
        (g, mode, s)
        for g in config.GENERATORS
        for mode in PURE_MODES
        for s in config.TEST_SEEDS
    ]
    mixed = [(g, "MIXED", s) for g in config.MIXED_GENERATORS
             for s in config.TEST_SEEDS]
    return pure + mixed


def _make_widths(mode: str, n_drifts: int, gradual_width: int,
                 rng: np.random.Generator) -> list[int]:
    if mode == "ABRUPT":
        return [0] * n_drifts
    if mode == "GRADUAL":
        return [gradual_width] * n_drifts
    if mode == "MIXED":
        if n_drifts == sum(config.MIXED_SPLIT):
            n_abrupt, n_gradual = config.MIXED_SPLIT
        else:  # shrunken streams in tests keep a half/half split
            n_abrupt = n_drifts // 2
            n_gradual = n_drifts - n_abrupt
        widths = [0] * n_abrupt + [gradual_width] * n_gradual
        return [int(w) for w in rng.permutation(widths)]
    raise ValueError(f"unknown mode: {mode!r}")


def plan_stream(
    generator: str,
    mode: str,
    seed: int,
    *,
    n_drifts: int | None = None,
    drift_every_n: int | None = None,
    gradual_width: int | None = None,
) -> dict:
    """Deterministic plan: concepts, drifts, ground truth and length."""
    if generator not in config.FUNCTIONS:
        raise ValueError(f"unknown generator: {generator!r}")
    n_drifts = config.N_DRIFTS if n_drifts is None else n_drifts
    drift_every_n = (config.DRIFT_EVERY_N if drift_every_n is None
                     else drift_every_n)
    gradual_width = (config.GRADUAL_WIDTH if gradual_width is None
                     else gradual_width)

    rng = np.random.default_rng(seed)
    widths = _make_widths(mode, n_drifts, gradual_width, rng)

    functions = [config.INIT_FUNCTION]
    instance_seeds = [int(rng.integers(0, 2**31 - 1))]
    for _ in range(n_drifts):
        candidates = [f for f in config.FUNCTIONS[generator]
                      if f != functions[-1]]
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

    length = n_drifts * drift_every_n + sum(widths) + drift_every_n
    return {
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "widths": widths,
        "functions": functions,
        "instance_seeds": instance_seeds,
        "drifts": drifts,
        "ground_truth": ground_truth,
        "length": length,
    }


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
    """Materialize a plan (from plan_stream or any plan of the same shape:
    generator, functions, instance_seeds, drifts) as a capymoa DriftStream."""
    import contextlib
    import io

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


def make_stream(generator: str, mode: str, seed: int, **plan_overrides):
    """Build (capymoa DriftStream, ground_truth, plan) for a spec."""
    plan = plan_stream(generator, mode, seed, **plan_overrides)
    return stream_from_plan(plan), plan["ground_truth"], plan
