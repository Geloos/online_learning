"""Per-step, per-episode and per-run logging of an ExtendedDDD run.

The logger is the only place where DDD's behaviour meets the ground truth:
it tags pool entries with the true concept they represent, checks recalls
against the true current concept and labels detections TP / duplicate / FP
with the detector study's matching rule (evaluation.match_alarms).

Pool-entry tag = the majority true concept over the period the stored
model was main (from the previous exit, or the stream start, to the
detection). A recalled entry that wins its episode is re-tagged with the
majority concept of that episode.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter

import numpy as np

from drift_ensemble.evaluation import compute_metrics, match_alarms
from ensemble_experiment.ddd import VOTERS

STEP_FIELDS = (["t", "mode", "n_active", "correct", "acc_global",
                "acc_main"]
               + [f"acc_{v}" for v in VOTERS]
               + [f"w_{v}" for v in VOTERS]
               + ["kind", "cap_fast", "cap_slow", "pool_size", "concept"])

EPISODE_FIELDS = [
    "t_detect", "mode_at_detection", "label", "drift_idx", "delay",
    "true_type", "true_recurring", "kind", "kind_correct", "drop", "acc_ref",
    "steps_after", "t_exit", "exit_type", "winner", "cap_used",
    "cap_fast_after", "cap_slow_after", "pool_add_id", "pool_add_tag",
    "pool_evicted_id", "recalled_pool_id", "recalled_just_added",
    "recall_score", "recalled_tag", "true_concept_at_recall", "recall_hit",
]

EXIT_TYPES = ["natural_new_low", "natural_old_high", "natural_recalled",
              "forced", "interrupted", "unfinished"]


def _majority(ids: np.ndarray) -> int | None:
    if len(ids) == 0:
        return None
    return int(Counter(ids.tolist()).most_common(1)[0][0])


def label_alarms(alarms, ground_truth, max_delay, duplicates_as_fp=True):
    """alarm t -> (label, drift index | None, delay | None) under the
    detector study's matching rule."""
    truth = sorted(ground_truth, key=lambda d: d["start"])
    m = match_alarms(alarms, truth, max_delay, duplicates_as_fp)
    first = {a: k for k, a in enumerate(m.matched) if a is not None}
    starts = [d["start"] for d in truth]
    out = {}
    for a in alarms:
        if a in first:
            k = first[a]
            out[a] = ("TP", k, a - starts[k])
            continue
        k = bisect_right(starts, a) - 1
        if k >= 0 and a <= truth[k]["end"] + max_delay:
            out[a] = ("duplicate", k, a - starts[k])
        else:
            out[a] = ("FP", None, None)
    return out


class RunLogger:
    def __init__(self, concepts: np.ndarray, *, min_training_size: int,
                 log_every: int):
        self.concepts = concepts
        self.min_training_size = min_training_size
        self.log_every = log_every
        n = len(concepts)
        self.correct = np.zeros(n, dtype=np.int8)
        self.n_active = np.zeros(n, dtype=np.int8)
        self.rows: list[dict] = []
        self.n_seen = 0
        self.n_correct = 0
        self.ddd = None

        self.pool_tags: dict[int, int | None] = {}
        self.pool_add_tag: dict[int, int | None] = {}  # t_detect -> tag
        self.recall_info: dict[int, dict] = {}  # t_detect -> recall record
        self.main_since = 0
        self.t_detect = None

    def attach(self, ddd) -> None:
        self.ddd = ddd

    # ------------------------------------------------------------------
    def on_event(self, ev: dict) -> None:
        t = ev["t"]
        kind = ev["type"]
        if kind == "pool_add":
            tag = _majority(self.concepts[self.main_since:t + 1])
            self.pool_tags[ev["id"]] = tag
            self.pool_add_tag[t] = tag
        elif kind == "detection":
            self.t_detect = t
        elif kind == "recall":
            tag = self.pool_tags.get(ev["id"])
            truth = int(self.concepts[t])
            self.recall_info[self.t_detect] = {
                "recalled_tag": tag,
                "true_concept_at_recall": truth,
                "recall_hit": tag is not None and tag == truth,
            }
        elif kind == "pool_replace":
            self.pool_tags[ev["id"]] = _majority(
                self.concepts[self.t_detect:t + 1])
        elif kind == "exit":
            self.main_since = t + 1

    # ------------------------------------------------------------------
    def record(self, t: int, info: dict) -> None:
        ok = bool(info["correct"])
        self.correct[t] = ok
        d = self.ddd
        self.n_active[t] = d.n_active_ensembles()
        if t >= self.min_training_size:
            self.n_seen += 1
            self.n_correct += ok
        if t % self.log_every:
            return
        after = d.mode == "after"
        active = set(d.active_voters()) if after else set()
        row = {
            "t": t,
            "mode": d.mode,
            "n_active": int(self.n_active[t]),
            "correct": int(ok),
            "acc_global": (self.n_correct / self.n_seen
                           if self.n_seen else None),
            "acc_main": d.main_window.acc if not after else None,
            "kind": d.kind,
            "cap_fast": d.cap.get("fast"),
            "cap_slow": d.cap.get("slow"),
            "pool_size": len(d.pool),
            "concept": int(self.concepts[t]),
        }
        for v in VOTERS:
            row[f"acc_{v}"] = d.trackers[v].acc if v in active else None
            row[f"w_{v}"] = d.last_weights.get(v) if v in active else None
        self.rows.append(row)

    # ------------------------------------------------------------------
    def episode_rows(self, ground_truth, max_delay, duplicates_as_fp):
        episodes = self.ddd.episodes
        labels = label_alarms([e["t_detect"] for e in episodes],
                              ground_truth, max_delay, duplicates_as_fp)
        truth = sorted(ground_truth, key=lambda d: d["start"])
        rows = []
        for e in episodes:
            label, k, delay = labels[e["t_detect"]]
            true_type = truth[k]["type"] if k is not None else None
            kind_correct = None
            if label == "TP" and e["kind"] is not None:
                kind_correct = ((e["kind"] == "fast")
                                == (true_type == "ABRUPT"))
            rec = self.recall_info.get(e["t_detect"], {})
            rid = e.get("recalled_pool_id")
            rows.append({
                **{f: e.get(f) for f in EPISODE_FIELDS if f in e},
                "label": label,
                "drift_idx": k,
                "delay": delay,
                "true_type": true_type,
                "true_recurring": (truth[k].get("recurring")
                                   if k is not None else None),
                "kind_correct": kind_correct,
                "pool_add_tag": self.pool_add_tag.get(e["t_detect"]),
                # The entry frozen at THIS detection is a copy of old-low:
                # recalling it is not a recall of an earlier concept.
                "recalled_just_added": (None if rid is None
                                        else rid == e.get("pool_add_id")),
                "recalled_tag": rec.get("recalled_tag"),
                "true_concept_at_recall": rec.get("true_concept_at_recall"),
                "recall_hit": rec.get("recall_hit"),
            })
        return rows

    def summary(self, episode_rows, alarms, ground_truth, length,
                max_delay, duplicates_as_fp) -> dict:
        n = len(self.correct)
        warm = self.correct[self.min_training_size:]
        out = {
            "acc": float(warm.mean()) if len(warm) else float("nan"),
            "acc_all": float(self.correct.mean()) if n else float("nan"),
            "n_detections": len(alarms),
        }
        out.update(compute_metrics(alarms, ground_truth, max_delay, length,
                                   self.min_training_size, duplicates_as_fp))
        out["frac_after"] = float((self.n_active >= 4).mean())
        out["frac_4"] = float((self.n_active == 4).mean())
        out["frac_5"] = float((self.n_active == 5).mean())
        for et in EXIT_TYPES:
            out[f"exit_{et}"] = sum(r["exit_type"] == et
                                    for r in episode_rows)
        graded = [r for r in episode_rows if r["kind_correct"] is not None]
        out["n_kind_graded"] = len(graded)
        out["kind_acc"] = (sum(r["kind_correct"] for r in graded)
                           / len(graded) if graded else float("nan"))
        recalls = [r for r in episode_rows if r["recall_hit"] is not None]
        out["n_recalls"] = len(recalls)
        out["recall_hit_rate"] = (sum(r["recall_hit"] for r in recalls)
                                  / len(recalls) if recalls
                                  else float("nan"))
        older = [r for r in recalls if not r["recalled_just_added"]]
        out["n_recalls_older"] = len(older)
        out["recall_hit_rate_older"] = (sum(r["recall_hit"] for r in older)
                                        / len(older) if older
                                        else float("nan"))
        steps = [r["steps_after"] for r in episode_rows
                 if r["exit_type"] not in ("interrupted", "unfinished")]
        out["mean_steps_after"] = (float(np.mean(steps)) if steps
                                   else float("nan"))
        cap = self.ddd.cap
        out["final_cap_fast"] = cap.get("fast")
        out["final_cap_slow"] = cap.get("slow")
        return out
