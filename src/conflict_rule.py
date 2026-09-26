"""Decision rule that treats house-number conflicts separately.

Leaderboard evidence (sub06, same scores, different thresholds): threshold 0.45 ->
0.921, 0.725 -> 0.937, 0.90 -> 0.948. The pairs removed by the stricter threshold were
largely look-alike businesses on the same street with a DIFFERENT house number (43% of
the 0.725-0.90 band has conflicting numbers vs 2.9% of pairs above 0.97). This rule
keeps the normal threshold for pairs whose numbers agree (or where one side has no
number) and demands higher confidence when both sides have numbers and none match.

Usage: python src/conflict_rule.py <scored_pairs.parquet> <out_dir> <base_threshold> <conflict_threshold>
"""
import os
import sys

sys.path.insert(0, "src")
import polars as pl

from ber.decide import one_owner
from ber.io import load_source, write_id_lists


def conflict_select(scored, s1, pool, base, conflict):
    """One-owner rule, then per-pair threshold depending on house-number agreement."""
    q = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("addr_nums").alias("qn"))
    c = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("addr_nums").alias("cn"))
    df = one_owner(scored).join(q, on="s1_id").join(c, on="cand_id").with_columns(
        ((pl.col("qn") != "") & (pl.col("cn") != "")
         & (pl.col("qn").str.split(" ").list.set_intersection(pl.col("cn").str.split(" ")).list.len() == 0))
        .alias("conflict"))
    keep = df.filter(pl.when(pl.col("conflict")).then(pl.col("prob") >= conflict)
                     .otherwise(pl.col("prob") >= base))
    return {s: ids for s, ids in keep.group_by("s1_id").agg("cand_id").iter_rows()}, df


def main():
    path, out_dir, base, conflict = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    s1 = load_source("test", 1)
    pool = pl.concat([load_source("test", 2), load_source("test", 3)])
    scored = pl.read_parquet(path)
    matches, _ = conflict_select(scored, s1, pool, base, conflict)
    ids = s1["entity_id"].to_list()
    cands = {s: v for s, v in scored.group_by("s1_id").agg("cand_id").iter_rows()}
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", ids, matches)
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", ids, cands)
    n = sum(len(v) for v in matches.values())
    print(f"wrote {out_dir}: base>={base}, conflict>={conflict}: {n:,} matches ({n/len(ids):.2f}/entity)")


if __name__ == "__main__":
    main()
