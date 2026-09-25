"""Step 2b: extra candidate-generation passes using exact keys (cheap hash joins).

The TF-IDF search misses three kinds of true matches (measured by error analysis):
  * a completely different trade name at the SAME address ('DREXGILD', same street)
  * website-style names ('pioneerproperty.com' for 'Pioneer Property')
  * records with an EMPTY address, where only the name can match
Each pass joins Source-1 to Source-2/3 records of the same country on an exact key.
Keys shared by many records carry little identity and would bloat the candidate set,
so keys above a frequency cap are skipped. New pairs are added to the TF-IDF
candidates (cos=0, rank=K+1) with flags saying which pass found them.

Usage: python src/add_key_candidates.py <train|test> <tfidf_cands.parquet> [max_key_freq]
Writes <tfidf_cands>_plus.parquet.
"""

import sys
import time

sys.path.insert(0, "src")

import polars as pl

from ber.io import load_source


def key_pass(s1, pool, key, flag, cap, pool_filter=None):
    """Exact-key join within country; skips keys held by more than `cap` pool records."""
    p = pool if pool_filter is None else pool.filter(pool_filter)
    p = p.filter(pl.col(key).str.len_chars() > 0)
    counts = p.group_by("country", key).len()
    p = p.join(counts.filter(pl.col("len") <= cap).select("country", key), on=["country", key])
    q = s1.filter(pl.col(key).str.len_chars() > 0)
    return q.select(pl.col("entity_id").alias("s1_id"), "country", key) \
            .join(p.select(pl.col("entity_id").alias("cand_id"), "country", key), on=["country", key]) \
            .select("s1_id", "cand_id").with_columns(pl.lit(1.0).alias(flag))


def main():
    split, path = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    t = time.time()
    cols = ["entity_id", "country", "name_compact", "name_key", "addr_key", "addr_norm"]
    s1 = load_source(split, 1).select(cols)
    pool = pl.concat([load_source(split, 2).select(cols), load_source(split, 3).select(cols)])
    # address key must identify a place: contains a number and at least 3 words
    addr_ok = pl.col("addr_key").str.contains(r"\d") & (pl.col("addr_key").str.count_matches(" ") >= 2)
    passes = [
        key_pass(s1.filter(addr_ok), pool, "addr_key", "via_addr_key", cap, pool_filter=addr_ok),
        key_pass(s1.filter(pl.col("name_compact").str.len_chars() >= 6), pool, "name_compact", "via_compact", cap),
        key_pass(s1, pool, "name_key", "via_empty_addr", cap, pool_filter=pl.col("addr_norm") == ""),
    ]
    extra = pl.concat(passes, how="diagonal").group_by("s1_id", "cand_id").agg(
        [pl.col(f).max() for f in ("via_addr_key", "via_compact", "via_empty_addr")]).fill_null(0.0)
    cands = pl.read_parquet(path)
    k = int(cands["rank"].max())
    merged = cands.join(extra, on=["s1_id", "cand_id"], how="full", coalesce=True).with_columns(
        pl.col("cos").fill_null(0.0), pl.col("rank").fill_null(k + 1).cast(pl.Int16),
        *[pl.col(f).fill_null(0.0) for f in ("via_addr_key", "via_compact", "via_empty_addr")])
    out = path.replace(".parquet", "_plus.parquet")
    merged.write_parquet(out)
    new = merged.height - cands.height
    print(f"{split}: tfidf pairs {cands.height:,} + new from key passes {new:,} "
          f"({new / s1.height:.2f}/S1) -> {merged.height:,} ({time.time()-t:.0f}s) -> {out}", flush=True)
    for f in ("via_addr_key", "via_compact", "via_empty_addr"):
        print(f"  {f}: {int(extra[f].sum()):,} pairs", flush=True)
    if split == "train":   # blocking recall before/after
        from ber.io import load_ground_truth
        gt, _ = load_ground_truth()
        n = gt.height
        before = gt.join(cands.select("s1_id", pl.col("cand_id").alias("match_id")), on=["s1_id", "match_id"]).height
        after = gt.join(merged.select("s1_id", pl.col("cand_id").alias("match_id")), on=["s1_id", "match_id"]).height
        print(f"  blocking recall: {before / n:.2%} -> {after / n:.2%}", flush=True)


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("add_key_candidates")
    main()
