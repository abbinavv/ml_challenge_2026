"""Before/after check for a normalisation change, on KNOWN true matches.

Old fields come from the Parquet cache; new fields are recomputed from the raw
text with the current src/ber/normalize.py. Reports, per country, how often true
pairs agree and how similar they are. Memory-light (sampled records only).
Usage: python src/compare_normalization.py [n_entities]
"""
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, "src")
import polars as pl
from rapidfuzz import fuzz

from ber.io import DATA_DIR, cache_path
from ber.normalize import normalize_address, normalize_name

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
random.seed(11)
pairs = []
with open(f"{DATA_DIR}/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        if random.random() < N / 2_100_000:
            s1, ids = line.rstrip("\n").split("\t")
            pairs += [(s1, m) for m in ids.split(",") if m]
need = list({p for pr in pairs for p in pr})
cols = ["entity_id", "name", "address", "country", "name_core", "name_compact", "addr_norm", "addr_nums"]
rec = pl.concat([pl.scan_parquet(cache_path("train", s)).select(cols)
                 .filter(pl.col("entity_id").is_in(need)).collect() for s in (1, 2, 3)])
R = {r[0]: r for r in rec.iter_rows()}
NEW = {k: (normalize_name(r[1]), normalize_address(r[2], r[3])) for k, r in R.items()}

m = defaultdict(list)
for s1, x in pairs:
    a, b = R[s1], R[x]
    c = a[3]
    (na, aa), (nb, ab) = NEW[s1], NEW[x]
    m[(c, "core_eq_old")].append(a[4] == b[4])
    m[(c, "core_eq_new")].append(na[1] == nb[1])
    m[(c, "compact_eq_old")].append(a[5] == b[5])
    m[(c, "compact_eq_new")].append(na[2] == nb[2])
    m[(c, "key_eq_new")].append(na[3] == nb[3])
    m[(c, "name_sim_old")].append(fuzz.token_set_ratio(a[4], b[4]))
    m[(c, "name_sim_new")].append(max(fuzz.token_set_ratio(na[1], nb[1]), fuzz.token_set_ratio(na[3], nb[3])))
    m[(c, "addr_sim_old")].append(fuzz.token_set_ratio(a[6], b[6]))
    m[(c, "addr_sim_new")].append(fuzz.token_set_ratio(aa[0], ab[0]))
    m[(c, "addrkey_sim_new")].append(fuzz.token_set_ratio(aa[2], ab[2]))
    old_na, old_nb = set(a[7].split()), set(b[7].split())
    new_na, new_nb = set(aa[1].split()), set(ab[1].split())
    m[(c, "nums_share_old")].append(bool(old_na & old_nb))
    m[(c, "nums_share_new")].append(bool(new_na & new_nb))

def pct(v): return f"{100 * sum(v) / len(v):5.1f}%"
def q(v, p): return sorted(v)[int(len(v) * p)]
print(f"true pairs: {len(pairs):,}")
for c in ("US", "India"):
    print(f"\n{c} ({len(m[(c, 'core_eq_old')]):,} pairs)       old     ->  new")
    print(f"  core name identical     {pct(m[(c,'core_eq_old')])}  ->  {pct(m[(c,'core_eq_new')])}   (key name identical: {pct(m[(c,'key_eq_new')])})")
    print(f"  compact name identical  {pct(m[(c,'compact_eq_old')])}  ->  {pct(m[(c,'compact_eq_new')])}")
    for k in ("name_sim", "addr_sim"):
        o, n = m[(c, k + "_old")], m[(c, k + "_new")]
        print(f"  {k:9} p10/p25/median {q(o,.1):3.0f}/{q(o,.25):3.0f}/{q(o,.5):3.0f}  ->  {q(n,.1):3.0f}/{q(n,.25):3.0f}/{q(n,.5):3.0f}   mean {st.mean(o):.1f} -> {st.mean(n):.1f}")
    k = m[(c, "addrkey_sim_new")]
    print(f"  address key sim (new)   p10/p25/median {q(k,.1):3.0f}/{q(k,.25):3.0f}/{q(k,.5):3.0f}   mean {st.mean(k):.1f}")
    print(f"  share a number          {pct(m[(c,'nums_share_old')])}  ->  {pct(m[(c,'nums_share_new')])}")
