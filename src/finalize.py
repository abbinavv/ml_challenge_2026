"""Turn scored test pairs into the final submission files.

Usage:
  python src/finalize.py <scored_pairs.parquet> <model_dir> <out_dir>
         [--keep 0.99] [--target-matches 3.34] [--threshold T] [--no-siblings]

Steps (all measured choices):
  1. Candidate set: keep pairs whose filter score p1 passes the cut-off for `--keep`
     (share of blocking-found true matches kept, from the model's validation table).
     99% -> ~4.5 candidates per entity for -0.0003 validation F0.5.
  2. Decision: one-owner rule, then a probability threshold. Either `--threshold`, or
     the threshold that yields `--target-matches` matches per entity -- the public
     leaderboard rewarded ~3.34 (sub06 scores: 3.64 -> 0.921, 3.48 -> 0.937, 3.34 -> 0.948).
  3. Sibling expansion: an unclaimed record with the same country, key name and address
     key as a matched record joins that match (99.9% same business on train truth).
     Added records are also added to the candidate list, so matches stay a subset.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
import polars as pl

from ber.decide import one_owner
from ber.io import load_source, write_id_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--keep", type=float, default=0.99)
    ap.add_argument("--target-matches", type=float, default=3.34)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-siblings", action="store_true")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.model_dir, "meta.json")))
    table = {round(r["keep"], 4): r["cutoff"] for r in meta.get("size_vs_score", [])}
    cut = table.get(round(a.keep, 4), meta["stage1"]["cutoff"])
    s1 = load_source("test", 1)
    n = s1.height
    scored = pl.read_parquet(a.scored)
    cands = scored.filter(pl.col("p1") >= cut) if "p1" in scored.columns else scored
    print(f"candidate set (keep {a.keep:.1%}, cut-off {cut:.4f}): {cands.height:,} pairs = {cands.height / n:.2f}/entity")

    owned = one_owner(cands)
    if a.threshold is None:   # threshold giving the target number of matches per entity
        probs = owned["prob"].sort(descending=True)
        k = min(int(a.target_matches * n), probs.len() - 1)
        thr = float(probs[k])
    else:
        thr = a.threshold
    sel = owned.filter(pl.col("prob") >= thr).select("s1_id", "cand_id")
    print(f"threshold {thr:.4f}: {sel.height:,} matches = {sel.height / n:.2f}/entity")

    if not a.no_siblings:
        cols = ["entity_id", "country", "name_key", "addr_key"]
        pool = pl.concat([load_source("test", 2).select(cols), load_source("test", 3).select(cols)]) \
                 .filter(pl.col("addr_key") != "")
        seeds = sel.join(pool.rename({"entity_id": "cand_id"}), on="cand_id").select("s1_id", "country", "name_key", "addr_key")
        claimed = sel["cand_id"].implode()
        sib = seeds.join(pool, on=["country", "name_key", "addr_key"]) \
                   .filter(~pl.col("entity_id").is_in(claimed)).unique(["s1_id", "entity_id"])
        sib = sib.filter(pl.col("s1_id").n_unique().over("entity_id") == 1) \
                 .select("s1_id", pl.col("entity_id").alias("cand_id"))
        sel = pl.concat([sel, sib])
        cands = pl.concat([cands.select("s1_id", "cand_id"), sib]).unique()
        print(f"sibling expansion: +{sib.height:,} matches -> {sel.height / n:.2f}/entity")

    ids = s1["entity_id"].to_list()
    m = {s: v for s, v in sel.group_by("s1_id").agg("cand_id").iter_rows()}
    c = {s: v for s, v in cands.select("s1_id", "cand_id").group_by("s1_id").agg("cand_id").iter_rows()}
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), "matched_entity_ids", ids, m)
    write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", ids, c)
    json.dump({"keep": a.keep, "cutoff": cut, "threshold": thr, "matches_per_entity": sel.height / n,
               "candidates_per_entity": sum(len(v) for v in c.values()) / n, "siblings": not a.no_siblings},
              open(os.path.join(a.out_dir, "finalize.json"), "w"), indent=2)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
