"""HT error cache.

HT is never reset and the detectors never influence it, so HT's 0/1 error
sequence on a stream is the same whatever detector is attached. It is
computed ONCE per stream and reused for every detector configuration.

errors[t] = -1 during the warm-up (t < MIN_TRAINING_SIZE), else 0/1.
"""

from __future__ import annotations

import json
import time

import numpy as np

from drift_ensemble import config
from streams import make_stream, stream_id


def stream_cache_dir(sid: str):
    return config.CACHE_DIR / sid


def is_cached(sid: str) -> bool:
    d = stream_cache_dir(sid)
    return (d / "meta.json").exists() and (d / "errors.npy").exists()


def cache_stream(
    generator: str,
    mode: str,
    seed: int,
    *,
    force: bool = False,
    min_training_size: int | None = None,
    **plan_overrides,
) -> str:
    """Run HT prequentially once over the stream; cache the 0/1 errors, the
    prequential accuracy curve and the ground truth. Returns the stream id.
    Skips work if the cache already exists (resumable)."""
    sid = stream_id(generator, mode, seed)
    if is_cached(sid) and not force:
        return sid
    if min_training_size is None:
        min_training_size = config.MIN_TRAINING_SIZE

    from capymoa.classifier import HoeffdingTree

    t0 = time.perf_counter()
    stream, ground_truth, plan = make_stream(
        generator, mode, seed, **{**config.STREAM_SIZES, **plan_overrides})
    n = plan["length"]
    ht = HoeffdingTree(schema=stream.get_schema())
    errors = np.full(n, -1, dtype=np.int8)
    accuracy = np.full(n, np.nan, dtype=np.float32)
    seen = 0
    correct = 0
    for t in range(n):
        inst = stream.next_instance()
        if t >= min_training_size:
            pred = ht.predict(inst)
            err = int(pred is None or int(pred) != int(inst.y_index))
            errors[t] = err
            seen += 1
            correct += 1 - err
            accuracy[t] = correct / seen
        ht.train(inst)

    out = stream_cache_dir(sid)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "errors.npy", errors)
    np.save(out / "ht_accuracy.npy", accuracy)
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=1))
    meta = {
        "generator": generator,
        "mode": mode,
        "seed": seed,
        "length": n,
        "min_training_size": min_training_size,
        "plan_overrides": plan_overrides,
        "final_accuracy": float(accuracy[n - 1]),
        "runtime_s": time.perf_counter() - t0,
    }
    # meta.json is written last: its presence marks the cache as complete.
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return sid


def cache_stream_spec(spec: tuple[str, str, int]) -> str:
    """Pool-friendly wrapper: cache one (generator, mode, seed) spec."""
    return cache_stream(*spec)


def load_cache(sid: str) -> tuple[np.ndarray, list[dict], dict]:
    """(errors, ground_truth, meta) for a cached stream."""
    d = stream_cache_dir(sid)
    if not is_cached(sid):
        raise FileNotFoundError(
            f"stream {sid!r} is not cached yet — run "
            f"drift_ensemble/scripts/01 (tuning) or 04 (test) first"
        )
    errors = np.load(d / "errors.npy")
    ground_truth = json.loads((d / "ground_truth.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return errors, ground_truth, meta
