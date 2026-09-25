"""Step 3: train the pair classifier on train candidates and tune the threshold.

Usage: python src/train_model.py <train_cands.parquet> <out_dir>

Splits the sampled S1 entities 70/30 into model-train / validation (by entity,
so no entity leaks across), trains LightGBM on candidate pairs, then tunes the
decision threshold for macro F0.5 on validation over ALL validation entities —
including those whose true matches were missed by blocking, and singletons.
"""

import json
import os
import sys
import time

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.decide import select, tune_threshold
from ber.features import FEATURES, add_group_features, build_pairs, compute_features
from ber.io import load_ground_truth, load_source
from ber.metrics import macro_f05, report

# Competition features need blocking over ALL S1 queries; excluded while train
# candidates come from a sample of S1 (their distribution would differ from test).
SAMPLE_EXCLUDED = {"cand_n_lists", "cand_best_cos", "cos_minus_cand_best", "is_cand_best"}


def log(msg):
    print(msg, flush=True)


def main(cands_path, out_dir, full_query_set=False):
    t0 = time.time()
    feats = [f for f in FEATURES if full_query_set or f not in SAMPLE_EXCLUDED]
    cands = add_group_features(pl.read_parquet(cands_path))
    s1 = load_source("train", 1)
    pool = pl.concat([load_source("train", 2), load_source("train", 3)])
    gt, _ = load_ground_truth()

    pairs = compute_features(build_pairs(cands, s1, pool))
    pairs = pairs.join(gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1).alias("label")),
                       on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    log(f"pairs={pairs.height:,} positives={pairs['label'].sum():,} ({time.time()-t0:.0f}s)")

    # entity-level split
    ents = np.array(sorted(cands["s1_id"].unique().to_list()))
    rng = np.random.default_rng(7)
    rng.shuffle(ents)
    cut = int(len(ents) * 0.7)
    tr_ents, va_ents = set(ents[:cut]), list(ents[cut:])
    is_tr = pairs["s1_id"].is_in(list(tr_ents))
    tr, va = pairs.filter(is_tr), pairs.filter(~is_tr)

    dtr = lgb.Dataset(tr.select(feats).to_numpy(), label=tr["label"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(va.select(feats).to_numpy(), label=va["label"].to_numpy(), reference=dtr)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1, seed=7)
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
    log(f"trained {model.best_iteration} rounds ({time.time()-t0:.0f}s)")

    va = va.with_columns(pl.Series("prob", model.predict(va.select(feats).to_numpy(),
                                                          num_iteration=model.best_iteration)))
    truths = {s: set(m) for s, m in gt.filter(pl.col("s1_id").is_in(va_ents))
              .group_by("s1_id").agg("match_id").iter_rows()}
    n_true = sum(len(v) for v in truths.values())
    found = va.filter(pl.col("label") == 1).height
    log(f"validation: {len(va_ents):,} entities, blocking recall={found / n_true:.2%}")

    (best_t, best_f), grid = tune_threshold(va, truths, va_ents, macro_f05)
    rep = report(select(va, best_t), truths, va_ents)
    log(f"best threshold={best_t} -> {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})}")
    no_owner = report(select(va, best_t, owner=False), truths, va_ents)["macro_f05"]
    log(f"  (without one-owner rule: macro_f05={no_owner:.4f})")

    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "model.txt"), num_iteration=model.best_iteration)
    meta = {"features": feats, "threshold": best_t, "validation": rep,
            "threshold_grid": grid, "blocking_recall": found / n_true,
            "feature_importance_gain": [(f, float(g)) for f, g in imp]}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("top features: " + ", ".join(f for f, _ in imp[:8]))
    log(f"saved to {out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
