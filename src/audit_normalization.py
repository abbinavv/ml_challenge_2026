"""Audit the text normalisation (key name / address key, as currently coded) on KNOWN true matches (train ground truth).

For a random sample of true (S1, S2/S3) pairs, compares the normalised fields word
by word and counts what still differs, so cleaning fixes can be prioritised by how
often a problem really occurs. Memory-light: reads only the sampled records.

Usage: python src/audit_normalization.py [n_entities]
"""
import random
import sys
from collections import Counter

sys.path.insert(0, "src")
import polars as pl

from ber.io import DATA_DIR, cache_path
from ber.normalize import normalize_address, normalize_name

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
random.seed(11)

# stream the ground truth, keep a random sample of entities with matches
pairs = []
with open(f"{DATA_DIR}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        if random.random() < N / 2_100_000:
            s1, ids = line.rstrip("\n").split("\t")
            pairs += [(s1, m) for m in ids.split(",") if m]
need = {p for pr in pairs for p in pr}
cols = ["entity_id", "name", "address", "country", "name_norm", "name_core", "name_compact",
        "addr_norm", "addr_nums", "non_latin"]
rec = pl.concat([pl.scan_parquet(cache_path("train", s)).select(cols)
                 .filter(pl.col("entity_id").is_in(list(need))).collect() for s in (1, 2, 3)])
# recompute the normalised fields with the CURRENT normalize.py (not the cache)
R = {}
for r in rec.iter_rows():
    nm = normalize_name(r[1]); ad = normalize_address(r[2], r[3])
    # layout: id, name, address, country, norm, core(key), compact, addr_key, nums
    R[r[0]] = (r[0], r[1], r[2], r[3], nm[0], nm[3], nm[2], ad[2], ad[1])

name_extra, name_missing = Counter(), Counter()
addr_extra, addr_missing = Counter(), Counter()
stats = Counter()
examples = {"name_core_diff": [], "nums_diff": [], "compact_diff": []}
for s1, m in pairs:
    a, b = R[s1], R[m]
    c = a[3]
    stats[f"pairs_{c}"] += 1
    qa, qb = set(a[5].split()), set(b[5].split())
    if a[5] == b[5]:
        stats[f"core_equal_{c}"] += 1
    if a[6] == b[6]:
        stats[f"compact_equal_{c}"] += 1
    elif len(examples["compact_diff"]) < 40:
        examples["compact_diff"].append((c, a[1], b[1], a[6], b[6]))
    for t in qb - qa:
        name_extra[t] += 1
    for t in qa - qb:
        name_missing[t] += 1
    ta, tb = set(a[7].split()), set(b[7].split())
    for t in tb - ta:
        addr_extra[t] += 1
    for t in ta - tb:
        addr_missing[t] += 1
    na, nb = set(a[8].split()), set(b[8].split())
    if na and nb and not (na & nb):
        stats[f"nums_conflict_{c}"] += 1
        if len(examples["nums_diff"]) < 25:
            examples["nums_diff"].append((c, a[2], b[2]))
    if not nb:
        stats[f"cand_no_nums_{c}"] += 1

print(f"sampled true pairs: {len(pairs):,}")
for c in ("US", "India"):
    n = stats[f"pairs_{c}"]
    print(f"  {c}: pairs={n:,}  core name equal={stats[f'core_equal_{c}']/n:.1%}  "
          f"compact equal={stats[f'compact_equal_{c}']/n:.1%}  "
          f"numbers conflict={stats[f'nums_conflict_{c}']/n:.1%}  candidate has no numbers={stats[f'cand_no_nums_{c}']/n:.1%}")
def show(title, cnt, k=45):
    print(f"\n{title}:")
    print("  " + ", ".join(f"{t}({n})" for t, n in cnt.most_common(k)))
show("NAME words in S2/S3 but not S1 (noise not removed / variants)", name_extra)
show("NAME words in S1 but not S2/S3 (dropped / abbreviated / misspelt)", name_missing)
show("ADDRESS words in S2/S3 but not S1", addr_extra)
show("ADDRESS words in S1 but not S2/S3", addr_missing)
print("\nCOMPACT NAME differences (country | S1 raw | S2/S3 raw | S1 compact | S2/S3 compact):")
for e in examples["compact_diff"]:
    print("  ", " | ".join(e))
print("\nNUMBER conflicts (country | S1 address | S2/S3 address):")
for e in examples["nums_diff"]:
    print("  ", " | ".join(e))
