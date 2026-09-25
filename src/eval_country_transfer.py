"""Leave-one-country-out check: how well does a model trained on one country
transfer to a country it never saw? Proxy for France (test-only).

Usage: python src/eval_country_transfer.py <train_cands.parquet>
Trains on US-only and on India-only entities, scores held-out India entities
with both, and reports the macro F0.5 gap (each with its own tuned threshold and
with the source-country threshold, which is what France will get).
"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from ber.decide import select, tune_threshold
from ber.features import COMPETITION_FEATURES, FEATURES, add_group_features
from ber.io import load_ground_truth, load_source
from ber.metrics import macro_f05
from train_model import featurize



def main():
    """Entry point (kept out of module level: macOS worker processes re-import this file)."""
    feats = [f for f in FEATURES if f not in COMPETITION_FEATURES]
    cands = add_group_features(pl.read_parquet(sys.argv[1]))
    s1 = load_source("train", 1)
    pool = pl.concat([load_source("train", 2), load_source("train", 3)])
    gt, _ = load_ground_truth()
    labels = gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1).alias("label"))
    country = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    ents = sorted(cands["s1_id"].unique().to_list())
    rng = np.random.default_rng(3); rng.shuffle(ents)
    us = [e for e in ents if country[e] == "US"][:20000]
    india = [e for e in ents if country[e] == "India"]
    in_tr, in_va = india[:20000], india[20000:30000]

    def data(ids):
        df = featurize(cands.filter(pl.col("s1_id").is_in(ids)), s1, pool, feats)
        return df.join(labels, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))

    va = data(in_va)
    truths = {s: set(m) for s, m in gt.filter(pl.col("s1_id").is_in(in_va)).group_by("s1_id").agg("match_id").iter_rows()}
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  num_threads=10, verbose=-1, seed=7)
    for name, ids in (("trained on US only", us), ("trained on India", in_tr)):
        tr = data(ids)
        m = lgb.train(params, lgb.Dataset(tr.select(feats).to_numpy(), label=tr["label"].to_numpy()), 300)
        scored = va.with_columns(pl.Series("prob", m.predict(va.select(feats).to_numpy())))
        (t, f), _ = tune_threshold(scored, truths, in_va, macro_f05)
        tr_s = tr.with_columns(pl.Series("prob", m.predict(tr.select(feats).to_numpy())))
        tr_truth = {s: set(x) for s, x in gt.filter(pl.col("s1_id").is_in(ids)).group_by("s1_id").agg("match_id").iter_rows()}
        (t_src, _), _ = tune_threshold(tr_s, tr_truth, ids, macro_f05)
        f_src = macro_f05(select(scored, t_src), truths, in_va)
        print(f"India validation, {name:18}: best-threshold F0.5={f:.4f} (t={t}) | source-tuned threshold t={t_src}: F0.5={f_src:.4f}", flush=True)


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("eval_country_transfer")
    main()
