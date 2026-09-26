"""Second-stage model: re-score pairs using how each candidate fits this S1 compared
with its rivals (other S1 entities that list it) and with the S1's own other
candidates.

Usage:
  python src/stack_rivals.py train <val_scored.parquet> <train_cands.parquet> <model_dir>
  python src/stack_rivals.py apply <scored_pairs.parquet> <test_cands.parquet> <model_dir> <out.parquet>

train: fits on held-out validation entities scored by the first-stage model
       (src/score_validation.py), so the first-stage probabilities are out-of-sample,
       and saves <model_dir>/stack.txt + stack.json.
apply: adds the same features to test scored pairs and writes them with `prob`
       replaced by the second-stage probability (first-stage kept as `prob_stage1`),
       ready for src/finalize.py.

Measured on validation (75K entities trained, 75K held out): macro F0.5 0.9746 ->
0.9772. The strongest new signals are raw_margin / raw_ratio: many same-name S1
entities are branches at different addresses, and a record without an address can
only be assigned by how closely its exact name text fits each namesake.
"""

import json
import os
import sys
import time

sys.path.insert(0, "src")

import lightgbm as lgb
import polars as pl

from ber.io import load_source
from ber.rivals import CONTEXT_FEATURES, RIVAL_FEATURES, context_features, rival_features

FEATS = ["prob", "p1"] + RIVAL_FEATURES + CONTEXT_FEATURES
CONFLICT_NEG_WEIGHT = 3.0


def add_features(scored, cands_path, split, n_chunks):
    allc = pl.read_parquet(cands_path, columns=["s1_id", "cand_id", "cos"])
    s1 = load_source(split, 1)
    pool = pl.concat([load_source(split, 2), load_source(split, 3)])
    scored = rival_features(scored, allc, s1, pool, n_chunks=n_chunks)
    del allc
    return context_features(scored, pool)


def main():
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "train":
        val, cands, model_dir = sys.argv[2:5]
        v = add_features(pl.read_parquet(val), cands, "train", 1)
        # same correction as the first stage: test has far more same-street look-alikes
        # with a different house number, so conflicting-number negatives weigh x3
        w = ((v["label"] == 0) & (v["nums_conflict"] > 0)).cast(pl.Float64).to_numpy() * (CONFLICT_NEG_WEIGHT - 1) + 1
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                           verbose=-1, seed=7, num_threads=10),
                      lgb.Dataset(v.select(FEATS).to_numpy(), label=v["label"].to_numpy(), weight=w), num_boost_round=600)
        m.save_model(os.path.join(model_dir, "stack.txt"))
        json.dump({"features": FEATS, "conflict_neg_weight": CONFLICT_NEG_WEIGHT, "trained_on": val, "pairs": v.height}, open(os.path.join(model_dir, "stack.json"), "w"), indent=2)
        print(f"stack model trained on {v.height:,} pairs ({time.time()-t0:.0f}s)")
    else:
        scored, cands, model_dir, out = sys.argv[2:6]
        m = lgb.Booster(model_file=os.path.join(model_dir, "stack.txt"))
        s = add_features(pl.read_parquet(scored), cands, "test", 8)
        s = s.with_columns(pl.col("prob").alias("prob_stage1"))
        s = s.with_columns(pl.Series("prob", m.predict(s.select(FEATS).to_numpy())))
        s.select("s1_id", "cand_id", "p1", "prob", "prob_stage1").write_parquet(out)
        print(f"wrote {out}: {s.height:,} pairs ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("stack_rivals")
    main()
