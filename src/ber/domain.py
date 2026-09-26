"""Label-free decoy detection by density ratio (domain classifier).

True variants are generated the same way in train and test; test adds decoys. A
classifier D(x) = P(test | x) trained to separate test pairs from validation pairs
(both scored by the same first-stage model, validation re-weighted so that equal
per-S1 densities give D = 0.5) estimates the density ratio r(x) = D / (1 - D).
Where test has r times the validation density, only 1/r of the test pairs can be
true, so the test-calibrated probability is  p_test(x) = p_val(x) / r(x).
"""

import polars as pl
from rapidfuzz import fuzz, process

from ber.io import load_source
from ber.rivals import _legal_expr

DOMAIN_FEATURES = ["prob", "p1", "nums_cat", "name_cat", "legal_cat", "raw_ratio", "addr_ratio",
                   "cand_blank", "s1_blank", "is_s3", "len_diff", "prob_rank", "n_hi", "n_mid"]


def describe(pairs, split):
    """Cheap pair descriptors for owned pairs (s1_id, cand_id, prob, p1)."""
    s1 = load_source(split, 1).select(
        pl.col("entity_id").alias("s1_id"), "country", pl.col("name_key").alias("k1"), pl.col("name_compact").alias("c1"),
        pl.col("addr_nums").fill_null("").alias("n1"), pl.col("addr_norm").fill_null("").alias("a1"),
        pl.col("name").fill_null("").str.to_lowercase().alias("r1"), _legal_expr("name").alias("l1"))
    pool = pl.concat([load_source(split, 2), load_source(split, 3)]).select(
        pl.col("entity_id").alias("cand_id"), pl.col("name_key").alias("k2"), pl.col("name_compact").alias("c2"),
        pl.col("addr_nums").fill_null("").alias("n2"), pl.col("addr_norm").fill_null("").alias("a2"),
        pl.col("name").fill_null("").str.to_lowercase().alias("r2"), _legal_expr("name").alias("l2"))
    d = pairs.join(s1, on="s1_id").join(pool, on="cand_id")
    ov = pl.col("n1").str.split(" ").list.set_intersection(pl.col("n2").str.split(" ")).list.len() > 0
    d = d.with_columns(
        pl.when(pl.col("a2") == "").then(0).when((pl.col("n1") == "") | (pl.col("n2") == "")).then(1)
          .when(pl.col("n1") == pl.col("n2")).then(2).when(ov).then(3).otherwise(4).alias("nums_cat"),
        pl.when(pl.col("c1") == pl.col("c2")).then(0).when(pl.col("k1") == pl.col("k2")).then(1).otherwise(2).alias("name_cat"),
        pl.when((pl.col("l1") == "") | (pl.col("l2") == "")).then(0).when(pl.col("l1") == pl.col("l2")).then(1).otherwise(2).alias("legal_cat"),
        (pl.col("a2") == "").cast(pl.Int8).alias("cand_blank"), (pl.col("a1") == "").cast(pl.Int8).alias("s1_blank"),
        pl.col("cand_id").str.starts_with("S3").cast(pl.Int8).alias("is_s3"),
        (pl.col("r1").str.len_chars() - pl.col("r2").str.len_chars()).alias("len_diff"),
        pl.col("prob").rank("ordinal", descending=True).over("s1_id").alias("prob_rank"),
        (pl.col("prob") >= 0.97).sum().over("s1_id").alias("n_hi"),
        ((pl.col("prob") >= 0.3) & (pl.col("prob") < 0.97)).sum().over("s1_id").alias("n_mid"))
    d = d.with_columns(
        pl.Series("raw_ratio", process.cpdist(d["r1"].to_list(), d["r2"].to_list(), scorer=fuzz.ratio, workers=-1), dtype=pl.Float64),
        pl.Series("addr_ratio", process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1), dtype=pl.Float64))
    return d.select("s1_id", "cand_id", "country", *DOMAIN_FEATURES, *[c for c in ("label",) if c in d.columns])
