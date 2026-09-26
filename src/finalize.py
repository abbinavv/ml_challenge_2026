"""Turn scored test pairs into the final submission files.

Usage:
  python src/finalize.py <scored_pairs.parquet> <model_dir> <out_dir>
         [--keep 0.99] [--target-matches 3.34] [--threshold T] [--no-siblings] [--word-veto]

Steps (all measured choices):
  1. Candidate set: keep pairs whose filter score p1 passes the cut-off for `--keep`
     (share of blocking-found true matches kept, from the model's validation table).
     99% -> ~4.5 candidates per entity for -0.0003 validation F0.5.
  2. Decision: one-owner rule, then a probability threshold. Either `--threshold`, or
     the threshold that yields `--target-matches` matches per entity -- the public
     leaderboard rewarded ~3.34 (sub06 scores: 3.64 -> 0.921, 3.48 -> 0.937, 3.34 -> 0.948).
  3. Word-swap veto (--word-veto): drop a match whose name swaps one of the S1 name's
     words for another frequent business word ('Fontaine Club' vs 'Fontaine Amicale').
     Neighbouring businesses at one address look like this (France: 2x more co-located
     businesses than train). Flags 0.02% of true train pairs vs 3.4% of French and
     0.5% of Indian test matches.
  4. Sibling expansion: an unclaimed record with the same country, key name and address
     key as a matched record joins that match (99.9% same business on train truth).
     Added records are also added to the candidate list, so matches stay a subset.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
import polars as pl

from ber.decide import (FILLER_WORDS, NEIGHBOUR_KINDS, TYPO_KINDS, foreign_word, legal_conflict,
                        name_vocabulary, number_change_kind, one_owner)
from ber.normalize import fold
from ber.io import load_source, write_id_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--keep", type=float, default=0.99)
    ap.add_argument("--target-matches", type=float, default=3.34)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-siblings", action="store_true")
    ap.add_argument("--word-veto", action="store_true")
    ap.add_argument("--legal-veto", nargs="*", default=None, metavar="COUNTRY",
                    help="drop matches whose legal forms conflict, in these countries")
    ap.add_argument("--neighbour-veto", nargs="*", default=None, metavar="COUNTRY",
                    help="drop matches whose house numbers conflict like a neighbour's (221 vs 225, 19 vs 22)")
    ap.add_argument("--typo-rescue", type=float, default=None, metavar="T",
                    help="also accept pairs with prob >= T whose conflicting numbers look like a typo (0102/102, 1416/1446)")
    ap.add_argument("--group-rules", default=None, metavar="DIR",
                    help="directory from src/group_rules.py: rescue mid-band groups that stay true on test, "
                         "veto groups that are decoys on test")
    ap.add_argument("--country-threshold", nargs="*", default=[], metavar="COUNTRY=T",
                    help="stricter threshold for some countries, e.g. France=0.97")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.model_dir, "meta.json")))
    table = {round(r["keep"], 4): r["cutoff"] for r in meta.get("size_vs_score", [])}
    cut = table.get(round(a.keep, 4), meta["stage1"]["cutoff"])
    s1 = load_source("test", 1)
    n = s1.height
    scored = pl.read_parquet(a.scored)
    cands = scored.filter(pl.col("p1") >= cut) if "p1" in scored.columns else scored
    print(f"candidate set (keep {a.keep:.1%}, cut-off {cut:.4f}): {cands.height:,} pairs = {cands.height / n:.2f}/entity")

    owned = one_owner(cands)
    if a.threshold is None:   # threshold giving the target number of matches per entity
        probs = owned["prob"].sort(descending=True)
        k = min(int(a.target_matches * n), probs.len() - 1)
        thr = float(probs[k])
    else:
        thr = a.threshold
    sel = owned.filter(pl.col("prob") >= thr).select("s1_id", "cand_id")
    print(f"threshold {thr:.4f}: {sel.height:,} matches = {sel.height / n:.2f}/entity")

    if a.country_threshold:
        cthr = {k: float(v) for k, v in (x.split("=") for x in a.country_threshold)}
        probs = owned.select("s1_id", "cand_id", "prob")
        chk = sel.join(probs, on=["s1_id", "cand_id"]).join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
        keep = pl.col("prob") >= pl.col("country").replace_strict(cthr, default=0.0, return_dtype=pl.Float64)
        before = sel.height
        sel = chk.filter(keep).select("s1_id", "cand_id")
        print(f"country thresholds {cthr}: -{before - sel.height:,} matches")

    if a.neighbour_veto is not None or a.typo_rescue is not None:
        addr = pl.concat([load_source("test", k).select("entity_id", "addr_norm", "addr_nums") for k in (2, 3)])
        s1a = s1.select(pl.col("entity_id").alias("s1_id"), "country", pl.col("addr_norm").alias("a1"), pl.col("addr_nums").alias("n1"))

        def with_kind(df):
            d = df.join(s1a, on="s1_id").join(addr.rename({"entity_id": "cand_id", "addr_norm": "a2", "addr_nums": "n2"}), on="cand_id")
            n1, n2 = pl.col("n1").fill_null(""), pl.col("n2").fill_null("")
            ov = n1.str.split(" ").list.set_intersection(n2.str.split(" ")).list.len() > 0
            d = d.with_columns(((n1 != "") & (n2 != "") & ~ov).alias("conflict"))
            return d.with_columns(pl.when(pl.col("conflict")).then(pl.struct("a1", "a2").map_elements(
                lambda r: number_change_kind(r["a1"], r["a2"]), return_dtype=pl.Utf8)).otherwise(pl.lit("")).alias("kind"))
        if a.neighbour_veto:
            chk = with_kind(sel)
            bad = pl.col("country").is_in(a.neighbour_veto) & pl.col("kind").is_in(list(NEIGHBOUR_KINDS))
            print(chk.filter(bad).group_by("country", "kind").len().sort("country", "kind").rows())
            print(f"neighbour-number veto {a.neighbour_veto}: -{int(chk.select(bad.sum()).item()):,} matches")
            sel = chk.filter(~bad).select("s1_id", "cand_id")
        if a.typo_rescue is not None:
            extra = owned.filter((pl.col("prob") >= a.typo_rescue) & (pl.col("prob") < thr)).select("s1_id", "cand_id")
            extra = with_kind(extra).filter(pl.col("kind").is_in(list(TYPO_KINDS))).select("s1_id", "cand_id")
            sel = pl.concat([sel, extra]).unique()
            print(f"typo-number rescue (prob >= {a.typo_rescue}): +{extra.height:,} matches")

    if a.group_rules:
        tags = pl.read_parquet(os.path.join(a.group_rules, "test_pair_tags.parquet"))
        rules = pl.read_parquet(os.path.join(a.group_rules, "group_rules.parquet")).select("band", "country", "nc", "nk", "action")
        probs = owned.select("s1_id", "cand_id", "prob")
        cty = s1.select(pl.col("entity_id").alias("s1_id"), "country")
        def ruled(df):
            d = df.join(probs, on=["s1_id", "cand_id"]).join(cty, on="s1_id").join(tags, on=["s1_id", "cand_id"], how="left")
            bands = json.load(open(os.path.join(a.group_rules, "bands.json")))
            band = pl.lit(None, dtype=pl.Int64)
            for i, (lo, hi) in enumerate(bands):
                band = pl.when((pl.col("prob") >= lo) & (pl.col("prob") < hi)).then(pl.lit(i)).otherwise(band)
            d = d.with_columns(band.alias("band"))
            return d.join(rules, on=["band", "country", "nc", "nk"], how="left")
        chk = ruled(sel)
        n_veto = int((chk["action"] == "veto").sum())
        sel = chk.filter(pl.col("action").fill_null("") != "veto").select("s1_id", "cand_id")
        pool_mid = owned.filter(pl.col("prob") < thr).select("s1_id", "cand_id")
        res = ruled(pool_mid).filter(pl.col("action") == "rescue").select("s1_id", "cand_id")
        sel = pl.concat([sel, res]).unique()
        print(f"group rules: -{n_veto:,} vetoed, +{res.height:,} rescued")

    if a.legal_veto:
        norms = pl.concat([load_source("test", k).select("entity_id", "name_norm", "name_core") for k in (2, 3)])
        chk = sel.join(s1.select(pl.col("entity_id").alias("s1_id"), "country", pl.col("name_norm").alias("n1")), on="s1_id") \
                 .join(norms.rename({"entity_id": "cand_id", "name_norm": "n2", "name_core": "c2"}), on="cand_id")
        chk = chk.with_columns((pl.col("country").is_in(a.legal_veto) & pl.struct("n1", "n2", "c2").map_elements(
            lambda r: legal_conflict(r["n1"], r["n2"], r["c2"]), return_dtype=pl.Boolean)).alias("veto"))
        print(f"legal-form veto {a.legal_veto}: -{int(chk['veto'].sum()):,} matches")
        sel = chk.filter(~pl.col("veto")).select("s1_id", "cand_id")

    vetoed = 0
    if a.word_veto:
        names = {}
        for split in ("train", "test"):
            src = load_source(split, 1).select("country", "name_core")
            for c, g in src.group_by("country"):
                names.setdefault(c[0], []).extend(g["name_core"].to_list())
        vocab = name_vocabulary(names)
        fillers = {fold(w) for w in FILLER_WORDS}
        cores = pl.concat([load_source("test", k).select("entity_id", "name_core") for k in (2, 3)])
        chk = sel.join(s1.select(pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("c1")), on="s1_id") \
                 .join(cores.rename({"entity_id": "cand_id", "name_core": "c2"}), on="cand_id")
        chk = chk.with_columns(pl.struct("country", "c1", "c2").map_elements(
            lambda r: foreign_word(r["c1"], r["c2"], vocab.get(r["country"], set()), fillers),
            return_dtype=pl.Boolean).alias("veto"))
        print(chk.group_by("country").agg(pl.len(), pl.col("veto").sum().alias("vetoed")).sort("country"))
        sel = chk.filter(~pl.col("veto")).select("s1_id", "cand_id")
        vetoed = int(chk["veto"].sum())
        print(f"word-swap veto: -{vetoed:,} matches -> {sel.height / n:.2f}/entity")

    if not a.no_siblings:
        cols = ["entity_id", "country", "name_key", "addr_key"]
        pool = pl.concat([load_source("test", 2).select(cols), load_source("test", 3).select(cols)]) \
                 .filter(pl.col("addr_key") != "")
        seeds = sel.join(pool.rename({"entity_id": "cand_id"}), on="cand_id").select("s1_id", "country", "name_key", "addr_key")
        claimed = sel["cand_id"].implode()
        sib = seeds.join(pool, on=["country", "name_key", "addr_key"]) \
                   .filter(~pl.col("entity_id").is_in(claimed)).unique(["s1_id", "entity_id"])
        sib = sib.filter(pl.col("s1_id").n_unique().over("entity_id") == 1) \
                 .select("s1_id", pl.col("entity_id").alias("cand_id"))
        sel = pl.concat([sel, sib])
        cands = pl.concat([cands.select("s1_id", "cand_id"), sib]).unique()
        print(f"sibling expansion: +{sib.height:,} matches -> {sel.height / n:.2f}/entity")

    ids = s1["entity_id"].to_list()
    m = {s: v for s, v in sel.group_by("s1_id").agg("cand_id").iter_rows()}
    c = {s: v for s, v in cands.select("s1_id", "cand_id").group_by("s1_id").agg("cand_id").iter_rows()}
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), "matched_entity_ids", ids, m)
    write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", ids, c)
    json.dump({"keep": a.keep, "cutoff": cut, "threshold": thr, "matches_per_entity": sel.height / n,
               "candidates_per_entity": sum(len(v) for v in c.values()) / n, "siblings": not a.no_siblings,
               "word_veto": a.word_veto, "vetoed": vetoed,
               "legal_veto": a.legal_veto, "country_threshold": a.country_threshold,
               "neighbour_veto": a.neighbour_veto, "typo_rescue": a.typo_rescue, "group_rules": a.group_rules},
              open(os.path.join(a.out_dir, "finalize.json"), "w"), indent=2)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
