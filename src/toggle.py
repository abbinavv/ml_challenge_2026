"""Make a leaderboard probe: a submission that differs from a base one by a chosen block
of pairs, so the score difference measures that block's true rate on test.

Usage: python src/toggle.py <base_dir> <out_dir> [--remove pairs.parquet] [--add pairs.parquet]

pairs.parquet: s1_id, cand_id. Added pairs are also added to the candidate list, so
matches stay a subset of candidates.
"""

import argparse
import os
import sys

sys.path.insert(0, "src")

import polars as pl

from ber.io import load_source, write_id_lists


def read_lists(path):
    d = pl.read_csv(path, separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8, "candidate_entity_ids": pl.Utf8})
    k, v = d.columns
    d = d.rename({k: "s1_id", v: "ids"})
    return d.drop_nulls("ids").with_columns(pl.col("ids").str.split(",")).explode("ids").rename({"ids": "cand_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--remove", default=None)
    ap.add_argument("--add", default=None)
    a = ap.parse_args()
    m = read_lists(os.path.join(a.base, "matching_results.tsv"))
    c = read_lists(os.path.join(a.base, "candidate_pairs.tsv"))
    n0 = m.height
    if a.remove:
        m = m.join(pl.read_parquet(a.remove).select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
    if a.add:
        extra = pl.read_parquet(a.add).select("s1_id", "cand_id")
        m = pl.concat([m, extra]).unique()
        c = pl.concat([c, extra]).unique()
    ids = load_source("test", 1)["entity_id"].to_list()
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), "matched_entity_ids", ids,
                   {s: v for s, v in m.group_by("s1_id").agg("cand_id").iter_rows()})
    write_id_lists(os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids", ids,
                   {s: v for s, v in c.group_by("s1_id").agg("cand_id").iter_rows()})
    print(f"{a.base} -> {a.out}: {n0:,} -> {m.height:,} matches")


if __name__ == "__main__":
    main()
