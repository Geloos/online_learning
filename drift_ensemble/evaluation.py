"""Alarm <-> drift matching and detection metrics.

The same matching rule is used everywhere: tuning, single detectors and the
ensemble.

Matching rule
-------------
The acceptable window of drift k is [start_k, end_k + max_delay].
For each drift, the FIRST alarm inside its window is a TP with
delay = alarm - start_k. Further alarms inside the same window are
duplicates: counted as FP if ``duplicates_as_fp`` else ignored. Alarms
outside every window are FP. Drifts with no alarm in their window are FN.
Windows must not overlap (asserted); alarms are assigned to the earliest
unmatched drift whose window contains them.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    delays: list[int] = field(default_factory=list)  # one per TP, drift order
    matched: list[int | None] = field(default_factory=list)  # alarm per drift
    tp_alarms: list[int] = field(default_factory=list)
    fp_alarms: list[int] = field(default_factory=list)  # outside every window
    duplicate_alarms: list[int] = field(default_factory=list)


def match_alarms(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    duplicates_as_fp: bool = True,
) -> MatchResult:
    """Apply the matching rule above. ``ground_truth`` is a list of dicts with
    at least 'start' and 'end' (end == start for abrupt drifts)."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    starts = [d["start"] for d in truth]
    ends = [d["end"] + max_delay for d in truth]
    for i in range(len(truth) - 1):
        assert ends[i] < starts[i + 1], (
            f"overlapping drift windows: [{starts[i]}, {ends[i]}] and "
            f"[{starts[i + 1]}, {ends[i + 1]}] — not supported"
        )

    matched: list[int | None] = [None] * len(truth)
    tp_alarms: list[int] = []
    fp_alarms: list[int] = []
    duplicate_alarms: list[int] = []

    for a in sorted(alarms):
        k = bisect_right(starts, a) - 1  # last drift with start <= a
        if k >= 0 and a <= ends[k]:
            if matched[k] is None:
                matched[k] = a
                tp_alarms.append(a)
            else:
                duplicate_alarms.append(a)
        else:
            fp_alarms.append(a)

    tp = sum(m is not None for m in matched)
    fn = len(truth) - tp
    fp = len(fp_alarms) + (len(duplicate_alarms) if duplicates_as_fp else 0)
    delays = [m - d["start"] for m, d in zip(matched, truth) if m is not None]
    return MatchResult(
        tp=tp, fp=fp, fn=fn, delays=delays, matched=matched,
        tp_alarms=tp_alarms, fp_alarms=fp_alarms,
        duplicate_alarms=duplicate_alarms,
    )


def _fbeta(precision: float, recall: float, beta: float) -> float:
    denom = beta * beta * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denom


def compute_metrics(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    n_instances: int,
    min_training_size: int,
    duplicates_as_fp: bool = True,
) -> dict:
    """TP/FP/FN, precision, recall, F1, F2, mean delay (TPs only) and false
    alarm rate per 1000 instances (after the warm-up period)."""
    m = match_alarms(alarms, ground_truth, max_delay, duplicates_as_fp)
    precision = m.tp / (m.tp + m.fp) if (m.tp + m.fp) else 0.0
    recall = m.tp / (m.tp + m.fn) if (m.tp + m.fn) else 0.0
    monitored_k = (n_instances - min_training_size) / 1000.0
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": precision,
        "recall": recall,
        "f1": _fbeta(precision, recall, 1.0),
        "f2": _fbeta(precision, recall, 2.0),
        "mean_delay": float(sum(m.delays) / len(m.delays)) if m.delays else math.nan,
        "far": m.fp / monitored_k if monitored_k > 0 else math.nan,
    }


def delay_curve(
    alarms: list[int],
    ground_truth: list[dict],
    delay_grid: list[int],
    duplicates_as_fp: bool = True,
) -> dict:
    """Precision, recall and F1 for each max-delay m in the grid; delay_auc
    is the mean recall over the grid (normalized to [0, 1])."""
    points = []
    for m in delay_grid:
        res = match_alarms(alarms, ground_truth, m, duplicates_as_fp)
        precision = res.tp / (res.tp + res.fp) if (res.tp + res.fp) else 0.0
        recall = res.tp / (res.tp + res.fn) if (res.tp + res.fn) else 0.0
        points.append({
            "max_delay": m,
            "precision": precision,
            "recall": recall,
            "f1": _fbeta(precision, recall, 1.0),
        })
    auc = sum(p["recall"] for p in points) / len(points) if points else math.nan
    return {"points": points, "delay_auc": auc}


def per_type(
    alarms: list[int],
    ground_truth: list[dict],
    max_delay: int,
    duplicates_as_fp: bool = True,
) -> dict:
    """Recall and mean delay restricted to the drifts of each type.

    Precision is only defined overall: an FP cannot be assigned to a type.
    """
    truth = sorted(ground_truth, key=lambda d: d["start"])
    m = match_alarms(alarms, truth, max_delay, duplicates_as_fp)
    out: dict[str, dict] = {}
    for dtype in ("ABRUPT", "GRADUAL"):
        idx = [i for i, d in enumerate(truth) if d["type"] == dtype]
        if not idx:
            continue
        hits = [i for i in idx if m.matched[i] is not None]
        delays = [m.matched[i] - truth[i]["start"] for i in hits]
        out[dtype] = {
            "n_drifts": len(idx),
            "recall": len(hits) / len(idx),
            "mean_delay": float(sum(delays) / len(delays)) if delays else math.nan,
        }
    return out
