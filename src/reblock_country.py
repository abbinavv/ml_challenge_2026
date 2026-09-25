"""Re-run blocking for ONE country with different settings and splice it into an
existing candidate file (other countries' candidates are kept unchanged).

Why: the speed shortcut (ignore words found in more than max_df of a country's
pool) drops generic words for the US and India, but France's data is concentrated
in a few cities, so it drops city names ('pessac' 4.8%, 'nouvelle aquitaine' 11%)
and blocking can no longer separate same-named businesses in different places.

Usage: python src/reblock_country.py <split> <country> <cands.parquet> <K> <max_df|none> <out.parquet>
"""
import sys, time
sys.path.insert(0, "src")
import polars as pl
from ber.io import load_source
from ber.blocking import block_country


def main():
    split, country, path, k, mdf, out = sys.argv[1:7]
    max_df = 1.0 if mdf == "none" else float(mdf)
    t = time.time()
    q = load_source(split, 1).filter(pl.col("country") == country)
    p = pl.concat([load_source(split, 2), load_source(split, 3)]).filter(pl.col("country") == country)
    new = block_country(q, p, top_k=int(k), max_df=max_df, log=lambda m: print(m, flush=True))
    old = pl.read_parquet(path)
    keep = old.filter(~pl.col("s1_id").is_in(q["entity_id"].implode()))
    cols = ["s1_id", "cand_id", "cos", "rank"]
    merged = pl.concat([keep.select(cols), new.select(cols)])
    merged.write_parquet(out)
    print(f"{country}: {new.height:,} new pairs replace {old.height - keep.height:,}; total {merged.height:,} "
          f"({time.time()-t:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("reblock_country")
    main()
