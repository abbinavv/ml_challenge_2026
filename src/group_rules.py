"""Label-free group rules: which kinds of pairs are true on TEST.

Usage: python src/group_rules.py <val_scored.parquet> <test_scored_pairs.parquet> <out_dir>

Every pair (owned, prob >= 0.3) gets a name-change kind (identical, typo, reorder,
drop_word, add_filler, filler_for_word, add_vocab, swap_vocab, swap_other, brand,
initialism) and a house-number relation (same, overlap, no_num, blank, or conflict:<change kind>,
see decide.number_change_kind).
True variants are generated the same way in train and test, so per group the true
pairs per S1 measured on validation predict the true pairs per S1 on test; test
true rate = that / test pairs per S1. Writes <out_dir>/test_pair_tags.parquet and
<out_dir>/group_rules.parquet: 'rescue' groups (score bands below 0.97, test rate >= 0.85)
and 'veto' groups (per band, test rate <= 0.5; a group almost absent from validation
    but frequent on test counts as decoys), for US and India. Rescue groups need
>= 100 validation pairs; all groups >= 500 test pairs.
"""

import sys
sys.path.insert(0, "src")
import polars as pl
from rapidfuzz import fuzz
from ber.io import load_source
from ber.calibrate import owned
from ber.decide import FILLER_WORDS, name_vocabulary, number_change_kind
from ber.normalize import fold
FILL = {fold(w) for w in FILLER_WORDS}
def build_vocab():
    names = {}
    for split in ("train","test"):
        s1 = load_source(split,1).select("country","name_core")
        for c, g in s1.group_by("country"): names.setdefault(c[0], []).extend(g["name_core"].to_list())
    return name_vocabulary(names)
VOC = build_vocab()
def seen(w, other): return any(w == y or w.startswith(y) or y.startswith(w) or fuzz.ratio(w, y) >= 75 for y in other)
def name_kind(country, c1, c2, k1, k2):
    a, b = (c1 or "").split(), (c2 or "").split()
    if not a or not b: return "empty"
    if "".join(a) == "".join(b): return "identical"
    if sorted(a) == sorted(b): return "reorder"
    if not (set(a) & set(b)) and not any(seen(w, a) for w in b):
        return "initialism" if len(b) == 1 and len(b[0]) <= 4 else "brand"
    ga = [w for w in b if len(w) >= 3 and not w.isdigit() and not seen(w, a)]
    la = [w for w in a if len(w) >= 3 and not w.isdigit() and not seen(w, b)]
    voc = VOC.get(country, set())
    gv = [w for w in ga if w not in FILL and w in voc]
    if not ga and not la: return "typo" if k1 != k2 else "keyeq_typo"
    if gv and la: return "swap_vocab"
    if gv: return "add_vocab"
    if ga and all(w in FILL for w in ga) and not la: return "add_filler"
    if ga and all(w in FILL for w in ga): return "filler_for_word"
    if la and not ga: return "drop_word"
    return "swap_other"
def nums(n1, n2, e2):
    if e2: return "blank"
    a, b = set((n1 or "").split()), set((n2 or "").split())
    if not a or not b: return "no_num"
    if a == b: return "same"
    return "overlap" if a & b else "conflict"
def tag(pairs, split):
    s1 = load_source(split,1).select(pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("c1"), pl.col("name_key").alias("k1"), pl.col("addr_nums").alias("n1"), pl.col("addr_norm").alias("a1"))
    pool = pl.concat([load_source(split,2),load_source(split,3)]).select(pl.col("entity_id").alias("cand_id"), pl.col("name_core").alias("c2"), pl.col("name_key").alias("k2"),
                                  pl.col("addr_nums").alias("n2"), (pl.col("addr_norm").fill_null("")=="").alias("e2"), pl.col("addr_norm").alias("a2"))
    d = pairs.join(s1, on="s1_id").join(pool, on="cand_id")
    return d.with_columns(
        pl.struct("country","c1","c2","k1","k2").map_elements(lambda r: name_kind(r["country"], r["c1"], r["c2"], r["k1"], r["k2"]), return_dtype=pl.Utf8).alias("nk"),
        pl.struct("n1","n2","e2","a1","a2").map_elements(lambda r: (lambda c: "conflict:" + number_change_kind(r["a1"], r["a2"]) if c == "conflict" else c)(nums(r["n1"], r["n2"], r["e2"])), return_dtype=pl.Utf8).alias("nc"))
BANDS = [(0.3, 0.6), (0.6, 0.8664), (0.8664, 0.97), (0.97, 1.01)]


def main():
    import os
    val_p, test_p, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    v = tag(owned(pl.read_parquet(val_p, columns=["s1_id","cand_id","prob","label"])).filter(pl.col("prob")>=BANDS[0][0]), "train")
    t = tag(owned(pl.read_parquet(test_p, columns=["s1_id","cand_id","prob"])).filter(pl.col("prob")>=BANDS[0][0]), "test")
    t.select("s1_id","cand_id","nk","nc").write_parquet(os.path.join(out_dir, "test_pair_tags.parquet"))
    rules = []
    n_val = {"US": 90123, "India": 59877}
    n_test = dict(load_source("test",1).group_by("country").len().iter_rows())
    import json
    json.dump(BANDS, open(os.path.join(out_dir, "bands.json"), "w"))
    for band, (lo, hi) in enumerate(BANDS):
        vv = v.filter((pl.col("prob")>=lo)&(pl.col("prob")<hi)); tt = t.filter((pl.col("prob")>=lo)&(pl.col("prob")<hi))
        a = vv.group_by("country","nc","nk").agg(pl.len().alias("v_n"), pl.col("label").mean().round(3).alias("val_rate"), pl.col("label").sum().alias("vt"))
        a = a.with_columns((pl.col("vt")/pl.col("country").replace_strict(n_val, return_dtype=pl.Float64)).alias("tps"))
        b = tt.group_by("country","nc","nk").agg(pl.len().alias("t_n")).with_columns((pl.col("t_n")/pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)).alias("tt"))
        j = b.join(a, on=["country","nc","nk"], how="left").with_columns((pl.col("tps")/pl.col("tt")).round(3).alias("test_rate"))
        j = j.filter((pl.col("country")!="France") & (pl.col("t_n")>=500))
        if hi <= 0.97:
            rules.append(j.filter((pl.col("v_n")>=100) & (pl.col("test_rate")>=0.85)).with_columns(pl.lit("rescue").alias("action"), pl.lit(band).alias("band")))
        rules.append(j.filter(pl.col("test_rate").fill_null(0.0)<=0.5).with_columns(pl.lit("veto").alias("action"), pl.lit(band).alias("band")))
    r = pl.concat([x.with_columns(pl.col(pl.Float64).cast(pl.Float64)) for x in rules], how="diagonal")
    r = r.select("band","country","nc","nk","action","v_n","val_rate","t_n","test_rate")
    # France has no labels. Its true variants follow the same generator (identical names,
    # typos, dropped / reordered words, filler swaps such as 'Prepa Plomberie SAS' ->
    # 'Prepa SAS Services'), so it borrows the US rescue rules for those kinds. Kinds that
    # can be ANOTHER business at the same address (brand, swapped real word) are not
    # borrowed: France has twice as many co-located businesses as train.
    safe_nk = ["identical", "typo", "drop_word", "reorder", "filler_for_word", "initialism"]
    safe_nc = ["same", "blank", "no_num", "conflict:digit_added_or_lost", "conflict:one_digit_sub_big", "conflict:zeros"]
    fr = r.filter((pl.col("country") == "US") & (pl.col("action") == "rescue") & pl.col("nk").is_in(safe_nk) & pl.col("nc").is_in(safe_nc))
    r = pl.concat([r, fr.with_columns(pl.lit("France").alias("country"))])
    r.write_parquet(os.path.join(out_dir, "group_rules.parquet"))
    with pl.Config(tbl_rows=80, tbl_width_chars=200): print(r.sort("action","band","t_n", descending=[False,False,True]))
if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("group_rules")
    main()
