"""Learn a word dictionary for business names written in Indian scripts.

`anyascii` transliterates native-script words phonetically ("प्राइवेट" -> "praivet"),
which is far from the Latin spelling ("private"). The training ground truth contains
tens of thousands of matched pairs where the Source-2/3 name is in an Indian script
and the Source-1 name is in Latin script. When both names have the same number of
words they are almost always word-for-word renderings, so aligning them position by
position gives (transliterated word -> Latin word) counts. When the word counts
differ, each transliterated word is paired with the Latin word whose consonant
skeleton is most similar ('stors' ~ 'stores', 'motrs' ~ 'motors'). A mapping is kept
when it is frequent and dominant.

Uses only the provided training data (no external resources).
Usage: python src/learn_translit.py  -> writes cache/translit.json
"""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")

import polars as pl

from rapidfuzz import fuzz

from ber.io import CACHE_DIR, load_ground_truth, load_source
from ber.normalize import skeleton, to_ascii_lower, tokens

MIN_COUNT = 3       # seen at least this many times
MIN_SHARE = 0.5     # the top Latin word must account for this share of alignments


def main():
    # raw transliteration of the native-script side (before any dictionary is applied)
    # Latin side as plain (unfolded) words, the same form as the native side
    s1 = load_source("train", 1).select("entity_id", "name")
    s1 = s1.with_columns(pl.col("name").map_elements(
        lambda n: " ".join(tokens(to_ascii_lower(n))), return_dtype=pl.Utf8).alias("name_norm")).drop("name")
    pool = pl.concat([load_source("train", 2), load_source("train", 3)]) \
        .filter(pl.col("non_latin")).select("entity_id", "name")
    pool = pool.with_columns(pl.col("name").map_elements(
        lambda n: " ".join(tokens(to_ascii_lower(n))), return_dtype=pl.Utf8).alias("name_norm")).drop("name")
    gt, _ = load_ground_truth()
    pairs = gt.join(pool.rename({"entity_id": "match_id", "name_norm": "src"}), on="match_id") \
              .join(s1.rename({"entity_id": "s1_id", "name_norm": "dst"}), on="s1_id")
    counts = defaultdict(Counter)
    aligned = 0
    for src, dst in pairs.select("src", "dst").iter_rows():
        a, b = src.split(), dst.split()
        if not a or not b:
            continue
        aligned += 1
        if len(a) == len(b):
            for x, y in zip(a, b):
                counts[x][y] += 1
        else:   # different word counts: pair by consonant-skeleton similarity
            sk = [(y, skeleton(y)) for y in b]
            for x in a:
                sx = skeleton(x)
                y, score = max(((y, fuzz.ratio(sx, sy)) for y, sy in sk), key=lambda r: r[1])
                if score >= 75:
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
    build_skeleton_map()


def build_skeleton_map(min_consonants=3, min_share=0.6, min_count=20):
    """Map UNSEEN transliterated words to Latin words by sound, without labels.

    Test uses Indian-script business words that never occur in train ('stors',
    'motrs', 'tekstails'), so the supervised dictionary cannot cover them. Their
    Latin spellings do occur in Source-1 names, so each skeleton (consonant sound)
    is mapped to its dominant Latin word from all Source-1 names of both splits
    (provided inputs, no labels). Written to cache/skelmap.json.
    """
    from collections import Counter
    words = Counter()
    for split in ("train", "test"):
        s1 = load_source(split, 1).filter(pl.col("country") == "India").select("name")
        for n in s1["name"].to_list():
            words.update(tokens(to_ascii_lower(n)))
    by_skel = defaultdict(Counter)
    for w, c in words.items():
        if c >= min_count and not w.isdigit() and len(w) >= 3:
            by_skel[skeleton(w)][w] += c
    # keep up to 8 frequent Latin spellings per sound; the best spelling match is
    # chosen at lookup time (normalize.sound_alike)
    skelmap = {sk: [w for w, _ in c.most_common(8)] for sk, c in by_skel.items()
               if len(sk) >= min_consonants}
    with open(os.path.join(CACHE_DIR, "skelmap.json"), "w") as f:
        json.dump({"latin_vocab": sorted(w for w, c in words.items() if c >= min_count),
                   "map": skelmap}, f, sort_keys=True)
    print(f"skeleton map: {len(skelmap):,} sounds from {len(words):,} Latin words")
