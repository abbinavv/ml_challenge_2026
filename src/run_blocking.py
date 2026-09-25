"""Step 2: generate top-K candidates for every S1 record of a split.

Usage: python src/run_blocking.py <train|test> [K] [n_sample_s1]
(n_sample_s1: block only a random sample of S1 queries; the pool is always complete)
Writes cache/<split>_cands_k<K>.parquet with columns s1_id, cand_id, cos, rank.
"""
import sys, time
sys.path.insert(0, "src")
import polars as pl
from ber.io import load_source, CACHE_DIR
from ber.blocking import block_all

if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("run_blocking")
    split = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t = time.time()
    s1 = load_source(split, 1)
    if len(sys.argv) > 3:
        s1 = s1.sample(int(sys.argv[3]), seed=42)
    pool = pl.concat([load_source(split, 2), load_source(split, 3)])
    cands = block_all(s1, pool, top_k=k, log=lambda m: print(m, flush=True))
    tag = f"_s{sys.argv[3]}" if len(sys.argv) > 3 else ""
    out = f"{CACHE_DIR}/{split}_cands_k{k}{tag}.parquet"
    cands.write_parquet(out)
    print(f"{split}: {cands.height:,} candidate pairs for {s1.height:,} S1 "
          f"({cands.height/s1.height:.1f}/S1) in {time.time()-t:.0f}s -> {out}", flush=True)
