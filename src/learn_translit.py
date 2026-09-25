"""Learn a word dictionary for business names written in Indian scripts.

`anyascii` transliterates native-script words phonetically ("प्राइवेट" -> "praivet"),
which is far from the Latin spelling ("private"). The training ground truth contains
tens of thousands of matched pairs where the Source-2/3 name is in an Indian script
and the Source-1 name is in Latin script. When both names have the same number of
words they are almost always word-for-word renderings, so aligning them position by
position gives (transliterated word -> Latin word) counts. A mapping is kept when it
is frequent and dominant.

Uses only the provided training data (no external resources).
Usage: python src/learn_translit.py  -> writes cache/translit.json
"""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")

import polars as pl

from ber.io import CACHE_DIR, load_ground_truth, load_source

MIN_COUNT = 3       # seen at least this many times
MIN_SHARE = 0.5     # the top Latin word must account for this share of alignments


def main():
    s1 = load_source("train", 1).select("entity_id", "name_norm")
    pool = pl.concat([load_source("train", 2), load_source("train", 3)]) \
        .filter(pl.col("non_latin")).select("entity_id", "name_norm")
    gt, _ = load_ground_truth()
    pairs = gt.join(pool.rename({"entity_id": "match_id", "name_norm": "src"}), on="match_id") \
              .join(s1.rename({"entity_id": "s1_id", "name_norm": "dst"}), on="s1_id")
    counts = defaultdict(Counter)
    aligned = 0
    for src, dst in pairs.select("src", "dst").iter_rows():
        a, b = src.split(), dst.split()
        if len(a) != len(b) or not a:
            continue
        aligned += 1
        for x, y in zip(a, b):
            counts[x][y] += 1
    table = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        total = sum(c.values())
        if n >= MIN_COUNT and n / total >= MIN_SHARE and x != y:
            table[x] = y
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "translit.json"), "w") as f:
        json.dump(table, f, ensure_ascii=False, sort_keys=True)
    print(f"native-script pairs={pairs.height:,} aligned={aligned:,} "
          f"source words={len(counts):,} kept mappings={len(table):,}")
    for x in list(table)[:15]:
        print(f"  {x} -> {table[x]}")


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("learn_translit")
    main()
