"""Step 1: parse and normalise all 6 source files into the Parquet cache."""
import sys, time
sys.path.insert(0, "src")
from ber.io import build_source_cache

if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("build_cache")
    for split in ("test", "train"):
        for source in (1, 2, 3):
            t = time.time()
            df = build_source_cache(split, source)
            print(f"{split}_s{source}: {df.height:,} rows in {time.time()-t:.0f}s", flush=True)
