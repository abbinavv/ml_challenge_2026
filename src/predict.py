"""Step 4: score test candidates and write the two submission files.

Usage: python src/predict.py <test_cands.parquet> <model_dir> <out_dir> [threshold]

Writes <out_dir>/matching_results.tsv (scored) and <out_dir>/candidate_pairs.tsv
(exactly the pairs the model scored), with one row for every test S1 entity.
Also saves the scored pairs so other thresholds can be written without re-scoring.
"""

import json
import os
import sys
import time

sys.path.insert(0, "src")

import lightgbm as lgb
import polars as pl

from ber.decide import select, select_expected
from ber.features import (add_group_features, build_pairs, compute_stage1_features,
                          compute_stage2_features)
from ber.io import load_source, write_id_lists


def log(msg):
    print(msg, flush=True)


def score_all(cands, model, feats, s1, pool, stage1=None, chunk_entities=100000):
    """Features + probabilities for all blocking pairs, chunked by S1. With a
    stage-1 filter, pairs below its cut-off are dropped (they are not candidates)
    and the main model scores only the survivors."""
    ents = cands["s1_id"].unique().to_list()
    parts = []
    for i in range(0, len(ents), chunk_entities):
        t = time.time()
        sub = cands.filter(pl.col("s1_id").is_in(ents[i:i + chunk_entities]))
        # cascade: cheap features for all blocking candidates -> candidate filter ->
        # expensive features only for the survivors (the candidate set)
        pairs = compute_stage1_features(build_pairs(sub, s1, pool))
        if stage1 is not None:
            m1, f1, cut = stage1
            pairs = pairs.filter(pl.Series(m1.predict(pairs.select(f1).to_numpy()) >= cut))
        pairs = compute_stage2_features(pairs)
        prob = model.predict(pairs.select(feats).to_numpy())
        parts.append(pairs.select("s1_id", "cand_id").with_columns(pl.Series("prob", prob)))
        log(f"  scored {min(i + chunk_entities, len(ents)):,}/{len(ents):,} entities ({time.time()-t:.0f}s)")
    return pl.concat(parts)


def apply_decision(scored, decision):
    """Match lists from scored pairs using the decision rule chosen in training."""
    if decision["method"] == "expected":
        return select_expected(scored, floor=decision["floor"], empty_weight=decision["empty_weight"])
    return select(scored, decision["threshold"])


def write_outputs(scored, s1_ids, decision, out_dir):
    """Write matching_results.tsv (one-owner + decision rule) and candidate_pairs.tsv."""
    if not isinstance(decision, dict):
        decision = {"method": "threshold", "threshold": float(decision)}
    matches = apply_decision(scored, decision)
    cands = {s: ids for s, ids in scored.group_by("s1_id").agg("cand_id").iter_rows()}
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids, matches)
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, cands)
    n_pairs = sum(len(v) for v in matches.values())
    log(f"wrote {out_dir}: {len(s1_ids):,} rows, {len(matches):,} entities with matches, "
        f"{n_pairs:,} matched pairs ({n_pairs/len(s1_ids):.2f}/entity)")


def main(cands_path, model_dir, out_dir, threshold=None):
    t0 = time.time()
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    feats = meta["features"]
    decision = meta.get("decision", {"method": "threshold", "threshold": meta["threshold"]})
    if threshold is not None:  # explicit override -> plain threshold rule
        decision = {"method": "threshold", "threshold": float(threshold)}
    model = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
    stage1 = None
    if "stage1" in meta:
        stage1 = (lgb.Booster(model_file=os.path.join(model_dir, "stage1.txt")),
                  meta["stage1"]["features"], meta["stage1"]["cutoff"])

    s1 = load_source("test", 1)
    pool = pl.concat([load_source("test", 2), load_source("test", 3)])
    cands = add_group_features(pl.read_parquet(cands_path))
    log(f"test: {s1.height:,} S1, {cands.height:,} candidate pairs")

    scored = score_all(cands, model, feats, s1, pool, stage1=stage1)
    log(f"candidate set: {scored.height:,} pairs = {scored.height / s1.height:.2f} per S1 entity")
    os.makedirs(out_dir, exist_ok=True)
    scored.write_parquet(os.path.join(out_dir, "scored_pairs.parquet"))
    write_outputs(scored, s1["entity_id"].to_list(), decision, out_dir)
    log(f"done in {time.time()-t0:.0f}s (decision={decision})")


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("predict")
    main(*sys.argv[1:4], *(sys.argv[4:5]))
