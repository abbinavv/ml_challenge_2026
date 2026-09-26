"""Replace first-stage probabilities by test-calibrated ones (label-free).

Usage: python src/domain_adjust.py <val_domain.parquet> <test_domain.parquet>
                                    <test_scored_pairs.parquet> <out.parquet>

Trains the validation-vs-test domain classifier (src/ber/domain.py) on US + India pairs
with first-stage prob >= 0.3, pair-level features only (entity-level context leaks the
decoy density of a business onto its true pairs), and writes the scored pairs with
prob = prob / max(r, 1), r = D / (1 - D). Countries without validation data (France)
keep the first-stage probability; finalize.py gives them their own threshold.
"""

import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.domain import DOMAIN_FEATURES
from ber.io import load_source

PAIR_FEATURES = [f for f in DOMAIN_FEATURES if f not in ("n_hi", "n_mid", "prob_rank")]
LABELLED = ["US", "India"]


def main():
    val_p, test_p, scored_p, out = sys.argv[1:5]
    import json
    va_n = pl.read_parquet(val_p).group_by("country").agg(pl.col("s1_id").n_unique()).rows()
    # per-country S1 counts: validation entities (all, incl. those without pairs) and test
    c = pl.read_parquet("cache/train_cands_k30_plus.parquet", columns=["s1_id"]).unique()
    e = np.array(sorted(c["s1_id"].to_list())); np.random.default_rng(7).shuffle(e); va = e[2000000:2150000].tolist()
    s1tr = load_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    n_val = dict(s1tr.filter(pl.col("s1_id").is_in(va)).group_by("country").len().iter_rows())
    n_test = dict(load_source("test", 1).group_by("country").len().iter_rows())
    v = pl.read_parquet(val_p).filter((pl.col("prob") >= 0.3) & pl.col("country").is_in(LABELLED))
    t = pl.read_parquet(test_p)
    tt = t.filter((pl.col("prob") >= 0.3) & pl.col("country").is_in(LABELLED))
    wv = v["country"].replace_strict({k: n_test[k] / n_val[k] for k in LABELLED}, return_dtype=pl.Float64).to_numpy()
    X = np.vstack([v.select(PAIR_FEATURES).to_numpy(), tt.select(PAIR_FEATURES).to_numpy()])
    y = np.r_[np.zeros(v.height), np.ones(tt.height)]
    w = np.r_[wv, np.ones(tt.height)]
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                       verbose=-1, seed=7, num_threads=10), lgb.Dataset(X, label=y, weight=w), num_boost_round=400)
    lab = t.filter(pl.col("country").is_in(LABELLED))
    D = np.clip(m.predict(lab.select(PAIR_FEATURES).to_numpy()), 1e-4, 1 - 1e-4)
    r = D / (1 - D)
    adj = lab.select("s1_id", "cand_id", pl.Series("r", r), pl.Series("pc", np.clip(lab["prob"].to_numpy() / np.maximum(r, 1.0), 0, 1)))
    s = pl.read_parquet(scored_p).join(adj, on=["s1_id", "cand_id"], how="left")
    s1c = load_source("test", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    s = s.join(s1c, on="s1_id")
    # labelled countries: calibrated prob for the owner pairs, 0 for non-owner pairs;
    # other countries keep the first-stage probability
    s = s.with_columns(pl.col("prob").alias("prob_stage1"),
                       pl.when(pl.col("country").is_in(LABELLED)).then(pl.col("pc").fill_null(0.0)).otherwise(pl.col("prob")).alias("prob"))
    s.select("s1_id", "cand_id", "p1", "prob", "prob_stage1", "r").write_parquet(out)
    print(f"wrote {out}: {s.height:,} pairs; labelled-country pairs adjusted: {adj.height:,}")


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("domain_adjust")
    main()
