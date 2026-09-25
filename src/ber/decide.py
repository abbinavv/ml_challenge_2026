"""Turning pair probabilities into per-entity match lists.

1. One-owner rule: in train every S2/S3 record belongs to at most one S1 entity,
   so each candidate is kept only for the S1 query that gives it the highest
   probability.
2. Threshold: keep the remaining candidates whose probability is >= threshold.
   The threshold is tuned for macro F0.5 on validation (precision-heavy).
Entities with nothing above the threshold get an empty list (singleton-safe).
"""

import numpy as np
import polars as pl


def one_owner(scored):
    """Keep, for each candidate id, only the pair(s) with the highest probability."""
    return scored.filter(pl.col("prob") == pl.col("prob").max().over("cand_id"))


def select(scored, threshold, owner=True):
    """Return {s1_id: [cand ids]} for pairs above `threshold` (after one-owner)."""
    df = one_owner(scored) if owner else scored
    df = df.filter(pl.col("prob") >= threshold)
    return {s1: ids for s1, ids in df.group_by("s1_id").agg("cand_id").iter_rows()}


def tune_threshold(scored, truths, entities, metric, grid=None, owner=True):
    """Grid-search the probability threshold maximising `metric` over `entities`."""
    grid = np.round(np.arange(0.20, 0.96, 0.025), 3) if grid is None else grid
    base = one_owner(scored) if owner else scored
    results = []
    for t in grid:
        preds = select(base, t, owner=False)
        results.append((float(t), metric(preds, truths, entities)))
    best = max(results, key=lambda r: r[1])
    return best, results
