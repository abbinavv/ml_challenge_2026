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


def select_expected(scored, floor=0.0, empty_weight=1.0, owner=True):
    """Per-entity selection maximising expected F0.5 (instead of a global threshold).

    For an entity with candidates sorted by probability p1 >= p2 >= ..., keeping the
    top k has expected F0.5 ~= 1.25 * (p1+..+pk) / (0.25 * sum(p) + k), treating the
    probabilities as calibrated and sum(p) as the expected number of true matches.
    The empty prediction scores 1.0 only if nothing matches, with probability
    ~= prod(1 - p). The best of the two is chosen per entity.
    floor        : candidates below this probability are never kept
    empty_weight : >1 favours empty predictions (more conservative)
    """
    df = one_owner(scored) if owner else scored
    df = df.sort(["s1_id", "prob"], descending=[False, True]).with_columns(
        pl.col("prob").cum_count().over("s1_id").alias("k"),
        pl.col("prob").cum_sum().over("s1_id").alias("cs"),
        pl.col("prob").sum().over("s1_id").alias("S"),
        (1 - pl.col("prob").clip(0, 0.999999)).log().sum().over("s1_id").exp().alias("p_empty"),
    ).with_columns((1.25 * pl.col("cs") / (0.25 * pl.col("S") + pl.col("k"))).alias("ef"))
    df = df.with_columns(pl.col("ef").max().over("s1_id").alias("ef_max"))
    df = df.with_columns(pl.when(pl.col("ef") == pl.col("ef_max")).then(pl.col("k"))
                         .min().over("s1_id").alias("k_best"))
    keep = df.filter((pl.col("k") <= pl.col("k_best")) & (pl.col("prob") >= floor)
                     & (pl.col("ef_max") > empty_weight * pl.col("p_empty")))
    return {s1: ids for s1, ids in keep.group_by("s1_id").agg("cand_id").iter_rows()}
