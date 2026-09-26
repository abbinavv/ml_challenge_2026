"""Rival features: how well a candidate fits this S1 compared with the other S1
entities that compete for it.

Train truth: 98% of the 'wrong' same-name candidates with a blank address are true
matches of ANOTHER S1 with the same name (branches of one brand at different
addresses). Pair features alone cannot tell which namesake owns such a record; the
comparison with the rivals can ('Younkin Plus' fits 'Younkin Plús' better than
'Younkin Plus PLLC').
"""

import polars as pl
from rapidfuzz import fuzz

RIVAL_FEATURES = ["n_rivals", "n_namesakes", "raw_ratio", "raw_margin", "raw_rank",
                  "norm_margin", "legal_tok_eq", "cand_blank", "cos_rank_among"]


_LEGAL_TAIL = ["llc", "inc", "corp", "ltd", "limited", "pvt", "private", "pllc", "pc", "lp", "llp",
               "co", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "pa"]


def _legal_expr(col):
    """Sorted legal-suffix tokens of a raw name, as one string (vectorised)."""
    toks = pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[.]", "").str.replace_all(r"[^a-z0-9]+", " ").str.split(" ")
    return toks.list.eval(pl.element().filter(pl.element().is_in(_LEGAL_TAIL))).list.sort().list.join(" ")


def rival_features(pairs, all_cands, s1, pool, n_chunks=1):
    """pairs: (s1_id, cand_id, ...) to describe. all_cands: the FULL blocking table of
    the split (s1_id, cand_id, cos) -- every S1 listing a candidate is its rival.
    s1/pool: source tables (entity_id, name, name_norm, name_key, addr_norm).
    Rivals are grouped by candidate, so chunking by candidate is exact."""
    from rapidfuzz import process
    s1c = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("name").fill_null("").str.to_lowercase().alias("n1"),
                    pl.col("name_norm").fill_null("").alias("m1"), pl.col("name_key").alias("k1"),
                    _legal_expr("name").alias("l1"))
    pc = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("name").fill_null("").str.to_lowercase().alias("n2"),
                     pl.col("name_norm").fill_null("").alias("m2"), pl.col("name_key").alias("k2"),
                     _legal_expr("name").alias("l2"),
                     (pl.col("addr_norm").fill_null("") == "").alias("cand_blank"))
    cands = pairs.select("cand_id").unique().with_columns((pl.col("cand_id").hash() % n_chunks).alias("_c"))
    parts = []
    for k in range(n_chunks):
        cc = cands.filter(pl.col("_c") == k).select("cand_id")
        comp = all_cands.select("s1_id", "cand_id", "cos").join(cc, on="cand_id").join(s1c, on="s1_id").join(pc, on="cand_id")
        comp = comp.with_columns(
            pl.Series("raw_ratio", process.cpdist(comp["n1"].to_list(), comp["n2"].to_list(), scorer=fuzz.ratio, workers=-1), dtype=pl.Float64),
            pl.Series("norm_ratio", process.cpdist(comp["m1"].to_list(), comp["m2"].to_list(), scorer=fuzz.ratio, workers=-1), dtype=pl.Float64),
            (pl.col("l1") == pl.col("l2")).cast(pl.Float64).alias("legal_tok_eq"),
            (pl.col("k1") == pl.col("k2")).alias("namesake"))
        comp = comp.with_columns(
            pl.len().over("cand_id").alias("n_rivals"),
            pl.col("namesake").sum().over("cand_id").alias("n_namesakes"),
            pl.col("raw_ratio").rank("min", descending=True).over("cand_id").alias("raw_rank"),
            pl.col("cos").rank("min", descending=True).over("cand_id").alias("cos_rank_among"))
        def margin(col):
            top2 = pl.col(col).top_k(2).over("cand_id", mapping_strategy="join")
            best, second = top2.list.get(0), top2.list.get(1, null_on_oob=True).fill_null(-100.0)
            return pl.when(pl.col(col) == best).then(pl.col(col) - second).otherwise(pl.col(col) - best)
        comp = comp.with_columns(margin("raw_ratio").alias("raw_margin"), margin("norm_ratio").alias("norm_margin"))
        parts.append(comp.select("s1_id", "cand_id", *RIVAL_FEATURES))
    comp = pl.concat(parts)
    out = pairs.join(comp, on=["s1_id", "cand_id"], how="left")
    return out.with_columns([pl.col(f).cast(pl.Float64) for f in RIVAL_FEATURES])


CONTEXT_FEATURES = ["prob_rank", "n_hi", "same_nums_hi", "same_nums_lo", "same_name_hi", "same_name_lo", "prob_gap_top"]


def context_features(scored, pool):
    """Features from the S1's own scored candidate list: does this candidate agree
    (house numbers, name) with the records the model is sure about, or with the ones
    it rejects (a look-alike business down the street has its own consistent number)?
    scored: s1_id, cand_id, prob. pool: entity_id, addr_nums, name_norm."""
    p = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("addr_nums").fill_null("").alias("nn"),
                    pl.col("name_norm").fill_null("").alias("mm"))
    d = scored.select("s1_id", "cand_id", "prob").join(p, on="cand_id", how="left")
    hi, lo = (pl.col("prob") >= 0.9), (pl.col("prob") < 0.3)
    d = d.with_columns(
        pl.col("prob").rank("ordinal", descending=True).over("s1_id").alias("prob_rank"),
        hi.sum().over("s1_id").alias("n_hi"),
        ((pl.col("nn") != "") & hi).sum().over("s1_id", "nn").alias("_nh"),
        ((pl.col("nn") != "") & lo).sum().over("s1_id", "nn").alias("_nl"),
        hi.sum().over("s1_id", "mm").alias("_mh"),
        lo.sum().over("s1_id", "mm").alias("_ml"),
        (pl.col("prob").max().over("s1_id") - pl.col("prob")).alias("prob_gap_top"))
    # exclude the candidate itself from its own counts
    d = d.with_columns(
        (pl.col("_nh") - (hi & (pl.col("nn") != "")).cast(pl.UInt32)).alias("same_nums_hi"),
        (pl.col("_nl") - (lo & (pl.col("nn") != "")).cast(pl.UInt32)).alias("same_nums_lo"),
        (pl.col("_mh") - hi.cast(pl.UInt32)).alias("same_name_hi"),
        (pl.col("_ml") - lo.cast(pl.UInt32)).alias("same_name_lo"))
    out = scored.join(d.select("s1_id", "cand_id", *CONTEXT_FEATURES), on=["s1_id", "cand_id"], how="left")
    return out.with_columns([pl.col(f).cast(pl.Float64) for f in CONTEXT_FEATURES])
