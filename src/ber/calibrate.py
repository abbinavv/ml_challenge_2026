"""Test-side calibration from pair densities (no test labels needed).

The process that creates TRUE variants of a business is the same in train and
test; test adds more DECOYS. So in any first-stage score band, the number of true
pairs per S1 entity is the same on validation and on test, and the extra test pairs
in that band are decoys:

    test_precision(band) = true_pairs_per_S1_on_validation(band) / test_pairs_per_S1(band)

Checked on the top bands (validation 2.53 vs test 2.53 pairs/S1 for India, 2.67 vs 2.68
for US: identical where there are no decoys). France has no validation data and
uses the mean of US and India.
"""

import numpy as np
import polars as pl

BANDS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.8664, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999]


def owned(df):
    return df.filter(pl.col("prob") == pl.col("prob").max().over("cand_id"))


def band(col="prob"):
    return pl.col(col).cut(BANDS).cast(pl.Utf8).alias("band")


def density_table(val, n_val, test, n_test):
    """val: owned validation pairs (s1_id, prob, label, country); test: owned test pairs
    (s1_id, prob, country); n_*: {country: number of S1 entities}. Returns per
    (country, band): val precision, true pairs/S1, test pairs/S1, test precision."""
    a = val.with_columns(band()).group_by("country", "band").agg(pl.len().alias("v_pairs"), pl.col("label").sum().alias("v_true"))
    a = a.with_columns((pl.col("v_true") / pl.col("country").replace_strict(n_val, return_dtype=pl.Float64)).alias("true_per_s1"),
                       (pl.col("v_true") / pl.col("v_pairs")).alias("val_prec"))
    pooled = a.group_by("band").agg(pl.col("true_per_s1").mean().alias("pooled_tps"), pl.col("val_prec").mean().alias("pooled_prec"))
    b = test.with_columns(band()).group_by("country", "band").agg(pl.len().alias("t_pairs"))
    b = b.with_columns((pl.col("t_pairs") / pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)).alias("test_per_s1"))
    j = b.join(a.select("country", "band", "true_per_s1", "val_prec"), on=["country", "band"], how="left").join(pooled, on="band", how="left")
    j = j.with_columns(pl.col("true_per_s1").fill_null(pl.col("pooled_tps")), pl.col("val_prec").fill_null(pl.col("pooled_prec")))
    return j.with_columns((pl.col("true_per_s1") / pl.col("test_per_s1")).clip(0.0, 1.0).fill_null(0.0).alias("test_prec"))


def expected_macro_f05(pairs, selected, entities, miss_per_s1):
    """Expected macro F0.5 from calibrated probabilities `pc` (independence approx.).
    pairs: all owned candidate pairs (s1_id, pc). selected: (s1_id, cand_id, pc) chosen.
    entities: list of all S1 ids. miss_per_s1: expected true matches outside the
    candidate set per entity (blocking/filter misses, from validation)."""
    tot = pairs.group_by("s1_id").agg(pl.col("pc").sum().alias("T"), (1 - pl.col("pc")).log().sum().alias("lognone"))
    sel = selected.group_by("s1_id").agg(pl.col("pc").sum().alias("tp"), pl.len().alias("k"))
    e = pl.DataFrame({"s1_id": entities}).join(tot, on="s1_id", how="left").join(sel, on="s1_id", how="left").fill_null(0)
    e = e.with_columns((pl.col("T") + miss_per_s1).alias("T"))
    f = pl.when(pl.col("k") > 0).then(1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("k"))) \
          .otherwise((pl.col("lognone") - miss_per_s1).exp())
    return float(e.select(f.mean()).item())
