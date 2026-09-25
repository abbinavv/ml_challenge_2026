"""Pair features for the matching model.

All features are country-agnostic similarities (no country one-hot), so the model
transfers to countries absent from training such as France.

Groups:
  * blocking   : cosine score, rank, gap to the query's best candidate
  * competition: how strongly other S1 queries claim the same candidate (soft
                 version of the one-owner rule observed in train)
  * name       : fuzzy scores on the normalised, core and compact name forms
  * address    : token similarity and house/unit number agreement
  * record     : source (S2/S3), native-script name, missing address
"""

from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FIELDS = ["name_norm", "name_core", "name_compact", "addr_norm", "addr_nums", "non_latin"]

FEATURES = [
    "cos", "rank", "cos_gap", "cos_ratio", "n_cands",
    "cand_n_lists", "cand_best_cos", "cos_minus_cand_best", "is_cand_best",
    "name_tset", "name_tsort", "name_ratio", "name_partial", "compact_jw",
    "compact_eq", "name_jacc", "core_len_diff",
    "addr_tset", "addr_ratio", "nums_jacc", "nums_conflict", "nums_q", "nums_c",
    "cand_addr_empty", "cand_is_s3", "cand_non_latin",
]


def _jacc(a, b):
    """Jaccard similarity of two space-separated token strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _string_feats(rows):
    """Worker: fuzzy string features for a chunk of pairs (query fields, cand fields)."""
    out = np.zeros((len(rows), 13), dtype=np.float32)
    for i, (qn, qc, qk, qa, qnum, cn, cc, ck, ca, cnum) in enumerate(rows):
        qs, cs = set(qnum.split()), set(cnum.split())
        out[i] = (
            fuzz.token_set_ratio(qc, cc),
            fuzz.token_sort_ratio(qc, cc),
            fuzz.ratio(qn, cn),
            fuzz.partial_ratio(qk, ck),
            JaroWinkler.normalized_similarity(qk, ck) * 100,
            float(qk == ck and qk != ""),
            _jacc(qc, cc),
            abs(len(qc) - len(cc)),
            fuzz.token_set_ratio(qa, ca),
            fuzz.ratio(qa, ca),
            len(qs & cs) / len(qs | cs) if (qs or cs) else 0.0,
            float(bool(qs) and bool(cs) and not (qs & cs)),
            len(qs),
        )
    return out


def build_pairs(cands, s1, pool):
    """Attach query and candidate fields to candidate pairs (one row per pair)."""
    q = s1.select(["entity_id"] + FIELDS).rename({f: f"q_{f}" for f in FIELDS} | {"entity_id": "s1_id"})
    c = pool.select(["entity_id"] + FIELDS).rename({f: f"c_{f}" for f in FIELDS} | {"entity_id": "cand_id"})
    return cands.join(q, on="s1_id", how="left").join(c, on="cand_id", how="left")


def add_group_features(cands):
    """Blocking + competition features. Must run on the FULL candidate table of a
    split (all S1 queries), before any sampling, so train and test see the same
    distribution for the per-candidate competition counts."""
    return cands.with_columns(
        pl.col("cos").max().over("s1_id").alias("_top"),
        pl.len().over("s1_id").alias("n_cands"),
        pl.len().over("cand_id").alias("cand_n_lists"),
        pl.col("cos").max().over("cand_id").alias("cand_best_cos"),
    ).with_columns(
        (pl.col("_top") - pl.col("cos")).alias("cos_gap"),
        (pl.col("cos") / pl.col("_top")).alias("cos_ratio"),
        (pl.col("cos") - pl.col("cand_best_cos")).alias("cos_minus_cand_best"),
        (pl.col("cos") >= pl.col("cand_best_cos")).cast(pl.Float32).alias("is_cand_best"),
    ).drop("_top")


def compute_features(pairs, workers=10, chunk=50000):
    """Add the per-pair FEATURES to a frame from build_pairs() (group features
    from add_group_features() must already be present)."""
    pairs = pairs.with_columns(
        (pl.col("c_addr_norm") == "").cast(pl.Float32).alias("cand_addr_empty"),
        pl.col("cand_id").str.starts_with("S3-").cast(pl.Float32).alias("cand_is_s3"),
        pl.col("c_non_latin").cast(pl.Float32).alias("cand_non_latin"),
        (pl.col("c_addr_nums").str.split(" ").list.len()
         * (pl.col("c_addr_nums") != "")).cast(pl.Float32).alias("nums_c"),
    )

    # fuzzy string features (parallel)
    cols = ["q_name_norm", "q_name_core", "q_name_compact", "q_addr_norm", "q_addr_nums",
            "c_name_norm", "c_name_core", "c_name_compact", "c_addr_norm", "c_addr_nums"]
    rows = list(zip(*[pairs[c].fill_null("").to_list() for c in cols]))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers) as pool:
        mats = pool.map(_string_feats, chunks)
    m = np.vstack(mats) if mats else np.zeros((0, 13), np.float32)
    names = ["name_tset", "name_tsort", "name_ratio", "name_partial", "compact_jw",
             "compact_eq", "name_jacc", "core_len_diff", "addr_tset", "addr_ratio",
             "nums_jacc", "nums_conflict", "nums_q"]
    pairs = pairs.with_columns([pl.Series(n, m[:, j]) for j, n in enumerate(names)])
    return pairs.with_columns([pl.col(f).cast(pl.Float32) for f in FEATURES])
