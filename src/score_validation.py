"""Re-score a trained model's validation entities, keeping features and labels.

Usage: python src/score_validation.py <train_cands.parquet> <model_dir> <out.parquet>
                                      [--n-train 2000000] [--n-val 150000]

Reproduces train_model.py's entity split (same shuffle, seed 7) so the entities are
exactly the ones the model never trained on, then runs the same cascade as predict.py
(stage-1 features -> filter -> stage-2 features -> main model). The output has every
feature, p1, prob and the true label, for error analysis of what the model misses.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.features import add_group_features, build_pairs, compute_stage1_features, compute_stage2_features
from ber.io import load_ground_truth, load_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--n-train", type=int, default=2000000)
    ap.add_argument("--n-val", type=int, default=150000)
    a = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(os.path.join(a.model_dir, "meta.json")))
    m1 = lgb.Booster(model_file=os.path.join(a.model_dir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(a.model_dir, "model.txt"))
    f1, cut, feats = meta["stage1"]["features"], meta["stage1"]["cutoff"], meta["features"]

    cands = pl.read_parquet(a.cands)
    ents = np.array(sorted(cands["s1_id"].unique().to_list()))
    np.random.default_rng(7).shuffle(ents)
    va = ents[a.n_train:a.n_train + a.n_val].tolist()
    cands = add_group_features(cands)
    cands = cands.filter(pl.col("s1_id").is_in(va))
    print(f"{len(va):,} validation entities, {cands.height:,} blocking pairs ({time.time()-t0:.0f}s)", flush=True)

    s1 = load_source("train", 1)
    pool = pl.concat([load_source("train", 2), load_source("train", 3)])
    pairs = compute_stage1_features(build_pairs(cands, s1, pool))
    pairs = pairs.with_columns(pl.Series("p1", m1.predict(pairs.select(f1).to_numpy())))
    gt, _ = load_ground_truth()
    lab = gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1).alias("label"))
    allp = pairs.select("s1_id", "cand_id", "p1").join(lab, on=["s1_id", "cand_id"], how="left").fill_null(0)
    pairs = compute_stage2_features(pairs.filter(pl.col("p1") >= cut))
    pairs = pairs.with_columns(pl.Series("prob", m2.predict(pairs.select(feats).to_numpy())))
    pairs = pairs.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    pairs.write_parquet(a.out)
    allp.write_parquet(a.out.replace(".parquet", "_prefilter.parquet"))
    print(f"wrote {a.out}: {pairs.height:,} pairs, {int(pairs['label'].sum()):,} true "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("score_validation")
    main()
