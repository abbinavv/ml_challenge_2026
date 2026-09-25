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

from ber.decide import select
from ber.features import add_group_features, build_pairs, compute_features
from ber.io import load_source, write_id_lists


def log(msg):
    print(msg, flush=True)


def score_all(cands, model, feats, s1, pool, chunk_entities=100000):
    """Compute features and model probabilities for all candidate pairs, chunked by S1."""
    ents = cands["s1_id"].unique().to_list()
    parts = []
    for i in range(0, len(ents), chunk_entities):
        t = time.time()
        sub = cands.filter(pl.col("s1_id").is_in(ents[i:i + chunk_entities]))
        pairs = compute_features(build_pairs(sub, s1, pool))
        prob = model.predict(pairs.select(feats).to_numpy())
        parts.append(pairs.select("s1_id", "cand_id").with_columns(pl.Series("prob", prob)))
        log(f"  scored {min(i + chunk_entities, len(ents)):,}/{len(ents):,} entities ({time.time()-t:.0f}s)")
    return pl.concat(parts)


def write_outputs(scored, s1_ids, threshold, out_dir):
    """Write matching_results.tsv (after one-owner + threshold) and candidate_pairs.tsv."""
    matches = select(scored, threshold)
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
    threshold = meta["threshold"] if threshold is None else float(threshold)
    model = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))

    s1 = load_source("test", 1)
    pool = pl.concat([load_source("test", 2), load_source("test", 3)])
    cands = add_group_features(pl.read_parquet(cands_path))
    log(f"test: {s1.height:,} S1, {cands.height:,} candidate pairs")

    scored = score_all(cands, model, feats, s1, pool)
    os.makedirs(out_dir, exist_ok=True)
    scored.write_parquet(os.path.join(out_dir, "scored_pairs.parquet"))
    write_outputs(scored, s1["entity_id"].to_list(), threshold, out_dir)
    log(f"done in {time.time()-t0:.0f}s (threshold={threshold})")


if __name__ == "__main__":
    main(*sys.argv[1:4], *(sys.argv[4:5]))
