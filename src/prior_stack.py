"""Prior-corrected second stage: P_test(true | x) learned from validation labels.

Usage: python src/prior_stack.py <val_scored.parquet> <test_scored_pairs.parquet>
                                 <test_pair_tags.parquet> <out.parquet>

Validation pairs (held-out entities scored by the first-stage model, owner pairs with
prob >= 0.3) carry true labels, but their negatives are rarer than on test: test adds
decoys. Per group g = (country, number relation, name-change kind) the test negatives per
S1 are  test_pairs_per_S1(g) - validation_true_per_S1(g)  (true variants are generated the
same way in both), so validation negatives are re-weighted by
    w(g) = test_negatives_per_S1(g) / validation_negatives_per_S1(g)
and a classifier on fine pair features learns the TEST posterior, splitting groups that
the group rules can only treat as a whole. 5-fold out-of-fold predictions on validation
report the re-weighted log-loss vs the first-stage probability.
Writes s1_id, cand_id, country, prob, pc for US/India test owner pairs with prob >= 0.3.
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
N_VAL = {"US": 90123, "India": 59877}
NUM = ["prob", "p1", "raw_ratio", "addr_ratio", "cand_blank", "s1_blank", "is_s3", "len_diff"]


def main():
    val_p, test_p, tags_p, out = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("gr", "src/group_rules.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    n_test = dict(load_source("test", 1).group_by("country").len().iter_rows())
    v = owned(pl.read_parquet(val_p, columns=["s1_id", "cand_id", "prob", "p1", "label"])).filter(pl.col("prob") >= 0.3)
    v = describe(v, "train").join(gr.tag(v.select("s1_id", "cand_id", "prob"), "train").select("s1_id", "cand_id", "nc", "nk"), on=["s1_id", "cand_id"])
    v = v.filter(pl.col("country").is_in(LABELLED))
    t = owned(pl.read_parquet(test_p, columns=["s1_id", "cand_id", "prob", "p1"])).filter(pl.col("prob") >= 0.3)
    t = describe(t, "test").join(pl.read_parquet(tags_p), on=["s1_id", "cand_id"], how="left").filter(pl.col("country").is_in(LABELLED))
    # group weights for validation negatives
    nv = pl.col("country").replace_strict(N_VAL, return_dtype=pl.Float64)
    nt = pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)
    a = v.group_by("country", "nc", "nk").agg((pl.col("label") == 1).sum().alias("vt"), (pl.col("label") == 0).sum().alias("vn")) \
         .with_columns((pl.col("vt") / nv).alias("tps"), (pl.col("vn") / nv).alias("nps"))
    b = t.group_by("country", "nc", "nk").agg(pl.len().alias("tn")).with_columns((pl.col("tn") / nt).alias("tt"))
    g = a.join(b, on=["country", "nc", "nk"], how="left").fill_null(0.0)
    g = g.with_columns(((pl.col("tt") - pl.col("tps")).clip(0.0, None) / pl.col("nps")).alias("w_neg"))
    g = g.with_columns(pl.when(pl.col("vn") >= 5).then(pl.col("w_neg")).otherwise(None).clip(0.2, 60.0).fill_null(1.0).alias("w_neg"))
    v = v.join(g.select("country", "nc", "nk", "w_neg"), on=["country", "nc", "nk"], how="left").with_columns(pl.col("w_neg").fill_null(1.0))
    w = np.where(v["label"].to_numpy() == 1, 1.0, v["w_neg"].to_numpy())
    cats = {c: sorted(set(v[c].drop_nulls().to_list()) | set(t[c].drop_nulls().to_list())) for c in ("nc", "nk")}

    def X(df):
        enc = [df[c].replace_strict({k: i for i, k in enumerate(cats[c])}, default=-1, return_dtype=pl.Int32).to_numpy() for c in ("nc", "nk")]
        return np.column_stack([df.select(NUM).to_numpy()] + enc + [(df["country"] == "India").cast(pl.Int8).to_numpy()])

    Xv, y = X(v), v["label"].to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, lambda_l2=5.0,
                  verbose=-1, seed=7, num_threads=10)
    cat_idx = [len(NUM), len(NUM) + 1, len(NUM) + 2]
    ents = v["s1_id"].to_numpy()
    uniq = np.unique(ents); rng = np.random.default_rng(3); fold_of = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    fold = np.array([fold_of[e] for e in ents])
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(Xv[tr], label=y[tr], weight=w[tr], categorical_feature=cat_idx), num_boost_round=400)
        oof[te] = m.predict(Xv[te])
    def wll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w)
    print(f"re-weighted (test-like) log-loss on validation, out-of-fold: first stage {wll(v['prob'].to_numpy()):.4f} -> prior-corrected {wll(oof):.4f}")
    # test-like precision / recall of the decision pc >= 0.78 vs group-rule style prob >= 0.97, out-of-fold
    for name, sel in (("prob >= 0.97", v["prob"].to_numpy() >= 0.97), ("pc >= 0.78", oof >= 0.78)):
        tp = np.sum(w * sel * y); fp = np.sum(w * sel * (1 - y)); pos = np.sum(w * y)
        print(f"  {name:13s}: test-like precision {tp / (tp + fp):.4f}, recall among pairs >= 0.3 {tp / pos:.4f}")
    v.select("s1_id", "cand_id", "country", "prob", "label", "nc", "nk").with_columns(pl.Series("w", w), pl.Series("oof", oof)) \
     .write_parquet(out.replace(".parquet", "_val_oof.parquet"))
    m = lgb.train(params, lgb.Dataset(Xv, label=y, weight=w, categorical_feature=cat_idx), num_boost_round=400)
    t = t.with_columns(pl.Series("pc", m.predict(X(t))))
    t.select("s1_id", "cand_id", "country", "prob", "pc").write_parquet(out)
    print(t.with_columns(pl.col("prob").cut([0.6, 0.8664, 0.97]).alias("band")).group_by("country", "band")
           .agg(pl.len(), pl.col("pc").mean().round(3).alias("mean_pc"), (pl.col("pc") >= 0.78).mean().round(3).alias("keep")).sort("country", "band"))


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("prior_stack")
    main()
