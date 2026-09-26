"""Per-pair test-calibrated probabilities from validation-vs-test densities (label-free).

Usage: python src/test_calibrate.py <val_scored.parquet> <test_scored_pairs.parquet>
                                    <test_pair_tags.parquet> <out.parquet>

Continuous version of src/group_rules.py. True variants are generated the same way in
train and test and test adds decoys, so for a pair description x

    P_test(true | x) = P_val(true | x) * p_val(x) / p_test(x)      (per-S1 densities)

p_test / p_val is estimated by a classifier separating test pairs from validation pairs
(US + India, first-stage prob >= 0.3, pair-level features only: entity-level context
leaks a business's decoy density onto its true pairs). Features: first-stage prob, the
group labels (house-number relation incl. change kind and overlap subset/mixed, name
change kind) and fine similarities. P_val(true | x) is the first-stage probability.
Writes s1_id, cand_id, country, prob, r, pc for owned US/India pairs with prob >= 0.3.
"""

import sys

sys.path.insert(0, "src")

import importlib.util

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.calibrate import owned
from ber.domain import describe
from ber.io import load_source

LABELLED = ["US", "India"]
N_VAL = {"US": 90123, "India": 59877}     # validation entities per country (v8 split)
NUM = ["prob", "p1", "raw_ratio", "addr_ratio", "cand_blank", "s1_blank", "is_s3", "len_diff"]


def _tagger():
    spec = importlib.util.spec_from_file_location("gr", "src/group_rules.py")
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    return gr.tag


def main():
    val_p, test_p, tags_p, out = sys.argv[1:5]
    n_test = dict(load_source("test", 1).group_by("country").len().iter_rows())
    tag = _tagger()
    v = owned(pl.read_parquet(val_p, columns=["s1_id", "cand_id", "prob", "p1", "label"])).filter(pl.col("prob") >= 0.3)
    v = describe(v, "train").join(tag(v.select("s1_id", "cand_id", "prob"), "train").select("s1_id", "cand_id", "nc", "nk"), on=["s1_id", "cand_id"])
    t = owned(pl.read_parquet(test_p, columns=["s1_id", "cand_id", "prob", "p1"])).filter(pl.col("prob") >= 0.3)
    t = describe(t, "test").join(pl.read_parquet(tags_p), on=["s1_id", "cand_id"], how="left")
    v = v.filter(pl.col("country").is_in(LABELLED))
    tl = t.filter(pl.col("country").is_in(LABELLED))
    cats = {c: sorted(set(v[c].drop_nulls().to_list()) | set(tl[c].drop_nulls().to_list())) for c in ("nc", "nk")}

    def X(df):
        enc = [df[c].replace_strict({k: i for i, k in enumerate(cats[c])}, default=-1, return_dtype=pl.Int32).to_numpy() for c in ("nc", "nk")]
        country = (df["country"] == "India").cast(pl.Int8).to_numpy()
        return np.column_stack([df.select(NUM).to_numpy()] + enc + [country])

    wv = v["country"].replace_strict({k: n_test[k] / N_VAL[k] for k in LABELLED}, return_dtype=pl.Float64).to_numpy()
    Xa = np.vstack([X(v), X(tl)])
    y = np.r_[np.zeros(v.height), np.ones(tl.height)]
    w = np.r_[wv, np.ones(tl.height)]
    n_num = len(NUM)
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, verbose=-1,
                       seed=7, num_threads=10), lgb.Dataset(Xa, label=y, weight=w, categorical_feature=[n_num, n_num + 1, n_num + 2]),
                  num_boost_round=500)
    D = np.clip(m.predict(X(tl)), 1e-4, 1 - 1e-4)
    r = D / (1 - D)
    pc = np.clip(tl["prob"].to_numpy() / r, 0.0, 0.999)
    res = tl.select("s1_id", "cand_id", "country", "prob").with_columns(pl.Series("r", r), pl.Series("pc", pc))
    res.write_parquet(out)
    # sanity: validation-weighted mean of D should equal the test share
    Dv = m.predict(X(v))
    print(f"mean D on validation (weighted) {np.average(Dv, weights=wv):.3f}; on test {D.mean():.3f}")
    print(res.with_columns(pl.col("prob").cut([0.6, 0.8664, 0.97]).alias("band")).group_by("country", "band")
             .agg(pl.len(), pl.col("pc").mean().round(3).alias("mean_pc"), (pl.col("pc") >= 0.78).mean().round(3).alias("keep_share"))
             .sort("country", "band"))


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("test_calibrate")
    main()
