"""Step 3: train the pair classifier and choose the decision rule on validation.

Usage:
  python src/train_model.py <train_cands.parquet> <out_dir> [--full] [--n-train N] [--n-val N]

--full : the candidate file covers ALL train S1 queries, so the competition features
         (how strongly other S1 entities claim a candidate) are valid and are used.
         Without it (a sampled query set) they are excluded, since their distribution
         would differ from test.

Entities are split into disjoint model-train / validation samples (no entity leaks
across). The decision rule — global threshold or per-entity expected-F0.5 — is
chosen by macro F0.5 on validation over ALL validation entities, including those
whose true matches blocking missed, and singletons. Results are also broken down
per country.
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

from ber.decide import select, select_expected, tune_threshold
from ber.features import (COMPETITION_FEATURES, FEATURES, add_group_features,
                          build_pairs, compute_features)

# Cheap features for the candidate filter: blocking/competition scores plus a few
# fast string similarities.
STAGE1_FEATURES = ["cos", "rank", "cos_gap", "cos_ratio", "n_cands", "cand_n_lists",
                   "cand_best_cos", "cos_minus_cand_best", "is_cand_best",
                   "name_tset", "addr_tset", "nums_jacc", "compact_eq", "key_tset"]
from ber.io import load_ground_truth, load_source
from ber.metrics import macro_f05, report


def log(msg):
    print(msg, flush=True)


def featurize(cands, s1, pool, feats, chunk_entities=100000):
    """Features for candidate pairs, computed in chunks of whole entities."""
    ents = cands["s1_id"].unique().to_list()
    parts = []
    for i in range(0, len(ents), chunk_entities):
        sub = cands.filter(pl.col("s1_id").is_in(ents[i:i + chunk_entities]))
        pairs = compute_features(build_pairs(sub, s1, pool))
        parts.append(pairs.select(["s1_id", "cand_id"] + feats))
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("out_dir")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-train", type=int, default=300000)
    ap.add_argument("--n-val", type=int, default=100000)
    ap.add_argument("--stage1-keep", type=float, default=0.995,
                    help="share of blocking-found true matches the candidate filter must keep")
    args = ap.parse_args()
    t0 = time.time()

    feats = [f for f in FEATURES if args.full or f not in COMPETITION_FEATURES]
    cands = add_group_features(pl.read_parquet(args.cands))  # on the FULL candidate set
    s1 = load_source("train", 1)
    pool = pl.concat([load_source("train", 2), load_source("train", 3)])
    gt, _ = load_ground_truth()
    country = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))

    # disjoint entity samples (drawn from all S1 queries, so entities without candidates count too)
    ents = np.array(sorted(cands["s1_id"].unique().to_list()))
    rng = np.random.default_rng(7)
    rng.shuffle(ents)
    tr_ents = ents[:args.n_train].tolist()
    va_ents = ents[args.n_train:args.n_train + args.n_val].tolist()

    labels = gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1).alias("label"))
    def labelled(ids):
        sub = cands.filter(pl.col("s1_id").is_in(ids))
        df = featurize(sub, s1, pool, feats)
        return df.join(labels, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    tr, va = labelled(tr_ents), labelled(va_ents)
    log(f"features: train pairs={tr.height:,} (pos {tr['label'].sum():,}), "
        f"val pairs={va.height:,} ({time.time()-t0:.0f}s)")

    # ---- Stage B: cheap candidate filter -> candidate_pairs.tsv --------------------
    # The organisers rank smaller candidate sets higher. A small model on cheap
    # features prunes obvious non-matches; its cut-off keeps `stage1_keep` of the true
    # matches that blocking found (measured on validation). Survivors are the
    # candidate set, and the main model is trained and applied on survivors only.
    s1_feats = [f for f in STAGE1_FEATURES if f in feats]
    p1 = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200,
              num_threads=10, verbose=-1, seed=7)
    m1 = lgb.train(p1, lgb.Dataset(tr.select(s1_feats).to_numpy(), label=tr["label"].to_numpy()),
                   num_boost_round=300)
    tr = tr.with_columns(pl.Series("p1", m1.predict(tr.select(s1_feats).to_numpy())))
    va = va.with_columns(pl.Series("p1", m1.predict(va.select(s1_feats).to_numpy())))
    pos = np.sort(va.filter(pl.col("label") == 1)["p1"].to_numpy())
    n_blocked_pos = len(pos)
    log(f"candidate filter (stage B) on validation — blocking gives {va.height / len(va_ents):.1f} candidates/entity:")
    for keep in (0.98, 0.99, 0.995, 0.998, 0.999):
        t = float(pos[int(len(pos) * (1 - keep))])
        log(f"  keep {keep:.1%} of found matches: cut-off {t:.4f} -> "
            f"{va.filter(pl.col('p1') >= t).height / len(va_ents):.2f} candidates/entity")
    t1 = float(pos[int(len(pos) * (1 - args.stage1_keep))])
    tr, va = tr.filter(pl.col("p1") >= t1), va.filter(pl.col("p1") >= t1)
    cands_per_entity = va.height / len(va_ents)
    log(f"chosen cut-off {t1:.4f}: {cands_per_entity:.2f} candidates/entity, "
        f"{va['label'].sum() / n_blocked_pos:.2%} of blocking-found matches kept")

    # ---- Stage C: main model on the candidate set ---------------------------------
    dtr = lgb.Dataset(tr.select(feats).to_numpy(), label=tr["label"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(va.select(feats).to_numpy(), label=va["label"].to_numpy(), reference=dtr)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1, seed=7)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    log(f"trained {model.best_iteration} rounds ({time.time()-t0:.0f}s)")

    va = va.with_columns(pl.Series("prob", model.predict(va.select(feats).to_numpy(),
                                                          num_iteration=model.best_iteration)))
    truths = {s: set(m) for s, m in gt.filter(pl.col("s1_id").is_in(va_ents))
              .group_by("s1_id").agg("match_id").iter_rows()}
    n_true = sum(len(v) for v in truths.values())
    recall = va["label"].sum() / n_true
    log(f"validation: {len(va_ents):,} entities, blocking recall={recall:.2%}")

    # decision rules
    (best_t, best_tf), grid = tune_threshold(va, truths, va_ents, macro_f05)
    options = {("threshold", best_t): best_tf}
    for ew in (1.0, 1.5, 2.0):
        for floor in (0.0, 0.3, 0.5):
            f = macro_f05(select_expected(va, floor=floor, empty_weight=ew), truths, va_ents)
            options[("expected", ew, floor)] = f
    for k, v in sorted(options.items(), key=lambda kv: -kv[1])[:5]:
        log(f"  {k}: macro_f05={v:.4f}")
    best = max(options, key=options.get)
    if best[0] == "threshold":
        decision = {"method": "threshold", "threshold": best[1]}
        preds = select(va, best[1])
    else:
        decision = {"method": "expected", "empty_weight": best[1], "floor": best[2]}
        preds = select_expected(va, floor=best[2], empty_weight=best[1])
    rep = report(preds, truths, va_ents)
    log(f"chosen {decision} -> " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()}))
    per_country = {}
    for c in sorted(set(country[e] for e in va_ents)):
        ce = [e for e in va_ents if country[e] == c]
        per_country[c] = report(preds, truths, ce)
        pc = per_country[c]
        log(f"  {c}: macro_f05={pc['macro_f05']:.4f} P={pc['pair_precision']:.3f} R={pc['pair_recall']:.3f} n={pc['entities']:,}")

    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_model(os.path.join(args.out_dir, "model.txt"), num_iteration=model.best_iteration)
    m1.save_model(os.path.join(args.out_dir, "stage1.txt"))
    meta = {"features": feats, "decision": decision, "threshold": best_t,
            "stage1": {"features": s1_feats, "cutoff": t1, "keep": args.stage1_keep,
                       "candidates_per_entity_val": cands_per_entity},
            "validation": rep, "per_country": per_country, "blocking_recall": recall,
            "decision_options": {str(k): v for k, v in options.items()},
            "threshold_grid": grid, "feature_importance_gain": [(f, float(g)) for f, g in imp]}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("top features: " + ", ".join(f for f, _ in imp[:10]))
    log(f"saved to {args.out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
