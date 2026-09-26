"""Prior-corrected second stage: P_test(true | x) learned from validation labels.

Usage: python src/prior_stack.py <val_scored.parquet>[,<more.parquet>...] <test_scored_pairs.parquet>[,<more>...]
                                 <test_pair_tags.parquet | -> <out.parquet>
Several validation files (e.g. cross-fitted halves) are concatenated; several test files are
averaged (prob, p1). With '-' the test pair tags are computed here.

Validation pairs (held-out entities scored by the first-stage model, owner pairs with
prob >= 0.3) carry true labels, but their negatives are rarer than on test: test adds
decoys. Per group g = (country, number relation, name-change kind) the test negatives per
S1 are  test_pairs_per_S1(g) - validation_true_per_S1(g)  (true variants are generated the
same way in both), so validation negatives are re-weighted per score band by
    w(g, band) = test_negatives_per_S1(g, band) / validation_negatives_per_S1(g, band)
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
# --band-weights: per-band, uncapped negative weights (tried 27 Sep). Default: the per-group,
# capped weighting that produced sub26 (leaderboard 0.967148).
BAND_WEIGHTS = "--band-weights" in sys.argv
N_VAL = {"US": 90123, "India": 59877}
NUM = ["prob", "p1", "raw_ratio", "addr_ratio", "cand_blank", "s1_blank", "is_s3", "len_diff"]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    val_ps, test_ps, tags_p, out = args[0].split(","), args[1].split(","), args[2], args[3]
    spec = importlib.util.spec_from_file_location("gr", "src/group_rules.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    n_test = dict(load_source("test", 1).group_by("country").len().iter_rows())
    vall = pl.concat([pl.read_parquet(p, columns=["s1_id", "cand_id", "prob", "p1", "label"]) for p in val_ps])
    nv_flag = [x for x in sys.argv if x.startswith("--n-val-json=")]
    if nv_flag:     # entity counts of a sampled validation set
        import json as _json
        n_val = _json.load(open(nv_flag[0].split("=", 1)[1]))
    elif len(val_ps) == 1 and "val_v8" in val_ps[0]:
        n_val = N_VAL
    else:   # every train entity is in some out-of-fold file: count them per country
        s1c = load_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
        cand_ents = pl.read_parquet("cache/train_cands_k30_plus.parquet", columns=["s1_id"]).unique()
        n_val = dict(cand_ents.join(s1c, on="s1_id").group_by("country").len().iter_rows())
    print(f"validation: {vall['s1_id'].n_unique():,} entities, {vall.height:,} pairs; entity counts {n_val}")
    v = owned(vall).filter(pl.col("prob") >= 0.3)
    v = describe(v, "train").join(gr.tag(v.select("s1_id", "cand_id", "prob"), "train").select("s1_id", "cand_id", "nc", "nk"), on=["s1_id", "cand_id"])
    v = v.filter(pl.col("country").is_in(LABELLED))
    ts = [pl.read_parquet(p, columns=["s1_id", "cand_id", "prob", "p1"]) for p in test_ps]
    t = ts[0] if len(ts) == 1 else pl.concat(ts).group_by("s1_id", "cand_id").agg(pl.col("prob").mean(), pl.col("p1").mean())
    t = owned(t).filter(pl.col("prob") >= 0.3)
    tags = gr.tag(t.select("s1_id", "cand_id", "prob"), "test").select("s1_id", "cand_id", "nc", "nk") if tags_p == "-" else pl.read_parquet(tags_p)
    t = describe(t, "test").join(tags, on=["s1_id", "cand_id"], how="left").filter(pl.col("country").is_in(LABELLED))
    if BAND_WEIGHTS:
        # group weights for validation negatives, per score band: decoys concentrate in the
        # middle bands (the leaderboard confirmed band-specific group rules, sub22 > sub08),
        # so band-blind weights under-count them
        def bandx():
            e = pl.lit(None, dtype=pl.Int32)
            for i, (lo, hi) in enumerate([(0.3, 0.6), (0.6, 0.8664), (0.8664, 0.97), (0.97, 1.01)]):
                e = pl.when((pl.col("prob") >= lo) & (pl.col("prob") < hi)).then(pl.lit(i, dtype=pl.Int32)).otherwise(e)
            return e.alias("band")
        v = v.with_columns(bandx()); t = t.with_columns(bandx())
        nv = pl.col("country").replace_strict(n_val, return_dtype=pl.Float64)
        nt = pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)
        def weights(keys):
            a = v.group_by(keys).agg((pl.col("label") == 1).sum().alias("vt"), (pl.col("label") == 0).sum().alias("vn")) \
                 .with_columns((pl.col("vt") / nv).alias("tps"), (pl.col("vn") / nv).alias("nps"))
            b = t.group_by(keys).agg(pl.len().alias("tn")).with_columns((pl.col("tn") / nt).alias("tt"))
            g = a.join(b, on=keys, how="left").with_columns(pl.col("tt").fill_null(0.0))
            g = g.with_columns(((pl.col("tt") - pl.col("tps")).clip(0.0, None) / pl.col("nps")).alias("w"))
            return g.with_columns(pl.when(pl.col("vn") >= 5).then(pl.col("w")).otherwise(None).alias("w")).select(*keys, "w")
        fine = weights(["country", "band", "nc", "nk"]).rename({"w": "w_fine"})
        coarse = weights(["country", "band"]).rename({"w": "w_coarse"})
        v = v.join(fine, on=["country", "band", "nc", "nk"], how="left").join(coarse, on=["country", "band"], how="left")
        # no upper cap: groups with few validation negatives need large factors (e.g. US top-band
        # one-digit neighbours ~130x); a cap makes their decoys look true
        v = v.with_columns(pl.coalesce("w_fine", "w_coarse", pl.lit(1.0)).clip(0.2, None).alias("w_neg"))
    else:
        # the submitted sub26 weighting: per group (band-blind), capped at 60
        nv = pl.col("country").replace_strict(n_val, return_dtype=pl.Float64)
        nt = pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)
        a = v.group_by("country", "nc", "nk").agg((pl.col("label") == 1).sum().alias("vt"), (pl.col("label") == 0).sum().alias("vn")) \
             .with_columns((pl.col("vt") / nv).alias("tps"), (pl.col("vn") / nv).alias("nps"))
        b = t.group_by("country", "nc", "nk").agg(pl.len().alias("tn")).with_columns((pl.col("tn") / nt).alias("tt"))
        g = a.join(b, on=["country", "nc", "nk"], how="left").with_columns(pl.col("tt").fill_null(0.0))
        g = g.with_columns(((pl.col("tt") - pl.col("tps")).clip(0.0, None) / pl.col("nps")).alias("w_neg"))
        g = g.with_columns(pl.when(pl.col("vn") >= 5).then(pl.col("w_neg")).otherwise(None).clip(0.2, 60.0).fill_null(1.0).alias("w_neg"))
        v = v.join(g.select("country", "nc", "nk", "w_neg"), on=["country", "nc", "nk"], how="left").with_columns(pl.col("w_neg").fill_null(1.0), pl.lit(None, dtype=pl.Int32).alias("band"))
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
    v.select("s1_id", "cand_id", "country", "prob", "label", "nc", "nk", "band").with_columns(pl.Series("w", w), pl.Series("oof", oof)) \
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
