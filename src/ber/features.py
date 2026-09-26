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
  * cluster    : agreement with the entity's other candidates. A business's true
                 records resemble each other, so a candidate similar to the entity's
                 strongest candidate, or sharing its exact name/numbers with other
                 candidates in the list, is more likely a true match
  * rarity     : how many Source-1 entities share this exact name / address. A rare
                 name or unique address is strong evidence (trade names, empty
                 addresses); a candidate whose name/address exactly equals ANOTHER
                 entity's is probably that entity's record (look-alike decoys)
  * form       : legal-form conflict (Private Limited vs LLP), content words added or
                 missing beyond generic descriptors ('Vijay Hospitality Steel')
"""

from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FIELDS = ["name_norm", "name_core", "name_compact", "name_key", "name_alt",
          "addr_norm", "addr_nums", "addr_key", "non_latin"]

FEATURES = [
    "cos", "rank", "cos_gap", "cos_ratio", "n_cands",
    "cand_n_lists", "cand_best_cos", "cos_minus_cand_best", "is_cand_best",
    "name_tset", "name_tsort", "name_ratio", "name_partial", "compact_jw",
    "compact_eq", "name_jacc", "core_len_diff",
    "addr_tset", "addr_ratio", "nums_jacc", "nums_conflict", "nums_q", "nums_c",
    "cand_addr_empty", "cand_is_s3", "cand_non_latin",
    "top_name_tset", "top_addr_tset", "top_nums_eq", "n_same_compact", "n_same_nums",
    "key_tset", "key_eq", "alt_tset", "acronym", "addr_key_tset", "nums_fuzzy",
    "legal_conflict", "legal_both", "key_extra", "key_missing", "skel_tset",
    "q_key_freq", "c_key_s1_freq", "c_key_other_s1", "q_addr_freq", "c_addr_s1_freq", "c_addr_other_s1",
    "via_addr_key", "via_compact", "via_empty_addr",
]

_STRING_NAMES = ["name_tset", "name_tsort", "name_ratio", "name_partial", "compact_jw",
                 "compact_eq", "name_jacc", "core_len_diff", "addr_tset", "addr_ratio",
                 "nums_jacc", "nums_conflict", "nums_q",
                 "key_tset", "key_eq", "alt_tset", "acronym", "addr_key_tset", "nums_fuzzy",
                 "legal_conflict", "legal_both", "key_extra", "key_missing", "skel_tset"]

# Features that need blocking over ALL S1 queries of a split (see add_group_features).
COMPETITION_FEATURES = {"cand_n_lists", "cand_best_cos", "cos_minus_cand_best", "is_cand_best"}


def _jacc(a, b):
    """Jaccard similarity of two space-separated token strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _near_numbers(qs, cs):
    """True if two different numbers look like the same number with a typo:
    a digit dropped/added at either end ('731' vs '31', '11850' vs '1185') or
    off by at most 2 ('14637' vs '14638'). Only numbers of 2+ digits count."""
    for a in qs:
        if len(a) < 2:
            continue
        for b in cs:
            if len(b) < 2 or a == b:
                continue
            if a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a):
                return True
            if abs(int(a) - int(b)) <= 2:
                return True
            if len(a) == len(b) >= 3 and sum(x != y for x, y in zip(a, b)) == 1:
                return True     # one digit mistyped ('10834' vs '60834')
    return False


from .normalize import fold, skeleton


def _skel(name):
    """Space-joined consonant skeletons of a name's words (sound-alike form)."""
    return " ".join(skeleton(t) for t in name.split())

# Legal-form families (letter-folded like the name tokens). Two names whose forms
# fall in different families ('Private Limited' vs 'LLP') are a warning sign.
_LEGAL_FAMILY = {fold(w): fam for fam, words in {
    "pvt": ["private", "pvt"], "ltd": ["limited", "ltd"], "llp": ["llp"],
    "inc": ["inc", "incorporated"], "llc": ["llc"], "pllc": ["pllc"], "corp": ["corp", "corporation"],
    "co": ["co", "company"], "sarl": ["sarl"], "sas": ["sas", "sasu"], "eurl": ["eurl"], "sa": ["sa"],
}.items() for w in words}


def _legal(norm):
    """Legal-form families present in a normalised name."""
    return {_LEGAL_FAMILY[t] for t in norm.split() if t in _LEGAL_FAMILY}


def _unmatched_words(a, b):
    """Words of key-name `a` with no exact or typo-level (ratio >= 80) match in `b`."""
    bs = b.split()
    return sum(1 for t in a.split() if t not in bs and not any(fuzz.ratio(t, u) >= 80 for u in bs))


def _acronym(q_core, c_compact):
    """True if one name is the initials of the other ('best agro' vs 'ba')."""
    words = q_core.split()
    return len(words) >= 2 and "".join(w[0] for w in words) == c_compact


def _string_feats(rows):
    """Worker: fuzzy string features for a chunk of pairs (query fields, cand fields)."""
    out = np.zeros((len(rows), len(_STRING_NAMES)), dtype=np.float32)
    for i, (qn, qc, qk, qkey, qa, qnum, qakey, cn, cc, ck, ckey, calt, ca, cnum, cakey) in enumerate(rows):
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
            fuzz.token_set_ratio(qkey, ckey),
            float(qkey == ckey and qkey != ""),
            fuzz.token_set_ratio(qc, calt) if calt else 0.0,
            float(_acronym(qc, ck) or _acronym(cc, qk)),
            fuzz.token_set_ratio(qakey, cakey),
            float(not (qs & cs) and _near_numbers(qs, cs)),
            float(bool(lq := _legal(qn)) and bool(lc := _legal(cn)) and not (lq & lc)),
            float(bool(_legal(qn)) and bool(_legal(cn))),
            _unmatched_words(ckey, qkey),
            _unmatched_words(qkey, ckey),
            fuzz.token_set_ratio(_skel(qc), _skel(cc)),
        )
    return out


_FREQ_CACHE = {}


def _s1_freqs(s1):
    """Counts of each (country, key name) and (country, address key) among ALL S1."""
    k = id(s1)
    if k not in _FREQ_CACHE:
        _FREQ_CACHE.clear()
        _FREQ_CACHE[k] = (
            s1.group_by("country", "name_key").len().rename({"len": "_kf"}),
            s1.filter(pl.col("addr_key") != "").group_by("country", "addr_key").len().rename({"len": "_af"}),
        )
    return _FREQ_CACHE[k]


def add_rarity_features(pairs, s1):
    """How many S1 entities share the query's / the candidate's exact key name and
    address key; 'other' excludes the query itself."""
    kf, af = _s1_freqs(s1)
    pairs = pairs.join(kf.rename({"name_key": "q_name_key", "_kf": "q_key_freq"}), on=["country", "q_name_key"], how="left") \
                 .join(kf.rename({"name_key": "c_name_key", "_kf": "c_key_s1_freq"}), on=["country", "c_name_key"], how="left") \
                 .join(af.rename({"addr_key": "q_addr_key", "_af": "q_addr_freq"}), on=["country", "q_addr_key"], how="left") \
                 .join(af.rename({"addr_key": "c_addr_key", "_af": "c_addr_s1_freq"}), on=["country", "c_addr_key"], how="left")
    pairs = pairs.with_columns([pl.col(c).fill_null(0) for c in ("q_key_freq", "c_key_s1_freq", "q_addr_freq", "c_addr_s1_freq")])
    return pairs.with_columns(
        (pl.col("c_key_s1_freq") - (pl.col("q_name_key") == pl.col("c_name_key")).cast(pl.Int64)).clip(0).alias("c_key_other_s1"),
        (pl.col("c_addr_s1_freq") - ((pl.col("q_addr_key") == pl.col("c_addr_key")) & (pl.col("c_addr_key") != "")).cast(pl.Int64)).clip(0).alias("c_addr_other_s1"),
    )


def build_pairs(cands, s1, pool):
    """Attach query and candidate fields to candidate pairs (one row per pair)."""
    q = s1.select(["entity_id", "country"] + FIELDS).rename({f: f"q_{f}" for f in FIELDS} | {"entity_id": "s1_id"})
    c = pool.select(["entity_id"] + FIELDS).rename({f: f"c_{f}" for f in FIELDS} | {"entity_id": "cand_id"})
    pairs = cands.join(q, on="s1_id", how="left").join(c, on="cand_id", how="left")
    return add_rarity_features(pairs, s1)


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


# The five string features the candidate filter (stage B) needs; everything else in
# _STRING_NAMES is only computed for the pairs that survive the filter.
_CHEAP_NAMES = ["name_tset", "compact_eq", "addr_tset", "nums_jacc", "key_tset"]
_STRING_COLS = ["q_name_norm", "q_name_core", "q_name_compact", "q_name_key", "q_addr_norm",
                "q_addr_nums", "q_addr_key",
                "c_name_norm", "c_name_core", "c_name_compact", "c_name_key", "c_name_alt",
                "c_addr_norm", "c_addr_nums", "c_addr_key"]


def _cheap_feats(rows):
    """Worker: the cheap string features used by the candidate filter (same formulas
    as the corresponding entries of _string_feats)."""
    out = np.zeros((len(rows), len(_CHEAP_NAMES)), dtype=np.float32)
    for i, (qn, qc, qk, qkey, qa, qnum, qakey, cn, cc, ck, ckey, calt, ca, cnum, cakey) in enumerate(rows):
        qs, cs = set(qnum.split()), set(cnum.split())
        out[i] = (
            fuzz.token_set_ratio(qc, cc),
            float(qk == ck and qk != ""),
            fuzz.token_set_ratio(qa, ca),
            len(qs & cs) / len(qs | cs) if (qs or cs) else 0.0,
            fuzz.token_set_ratio(qkey, ckey),
        )
    return out


def _parallel(fn, pairs, n_out, workers, chunk):
    """Run a row-wise feature worker over the string columns in parallel."""
    rows = list(zip(*[pairs[c].fill_null("").to_list() for c in _STRING_COLS]))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers) as pool:
        mats = pool.map(fn, chunks)
    return np.vstack(mats) if mats else np.zeros((0, n_out), np.float32)


def _cast(pairs):
    return pairs.with_columns([pl.col(f).cast(pl.Float32) for f in FEATURES if f in pairs.columns])


def compute_stage1_features(pairs, workers=10, chunk=50000):
    """Cheap features for EVERY blocking candidate: record flags, the five cheap string
    similarities and the cluster features (which describe the entity's full candidate
    list, so they must see all candidates). Enough for the candidate filter."""
    pairs = pairs.with_columns(
        (pl.col("c_addr_norm") == "").cast(pl.Float32).alias("cand_addr_empty"),
        pl.col("cand_id").str.starts_with("S3-").cast(pl.Float32).alias("cand_is_s3"),
        pl.col("c_non_latin").cast(pl.Float32).alias("cand_non_latin"),
        (pl.col("c_addr_nums").str.split(" ").list.len()
         * (pl.col("c_addr_nums") != "")).cast(pl.Float32).alias("nums_c"),
    )
    m = _parallel(_cheap_feats, pairs, len(_CHEAP_NAMES), workers, chunk)
    pairs = pairs.with_columns([pl.Series(n, m[:, j]) for j, n in enumerate(_CHEAP_NAMES)])
    pairs = add_cluster_features(pairs, workers=workers)
    for flag in ("via_addr_key", "via_compact", "via_empty_addr"):   # set by extra blocking passes
        if flag not in pairs.columns:
            pairs = pairs.with_columns(pl.lit(0.0).alias(flag))
    return _cast(pairs)


def compute_stage2_features(pairs, workers=10, chunk=50000):
    """The remaining (expensive) string features, for the candidates that survived
    the filter only."""
    m = _parallel(_string_feats, pairs, len(_STRING_NAMES), workers, chunk)
    pairs = pairs.with_columns([pl.Series(n, m[:, j]) for j, n in enumerate(_STRING_NAMES)
                                if n not in _CHEAP_NAMES])
    return _cast(pairs)


def compute_features(pairs, workers=10, chunk=50000):
    """All FEATURES for every pair (stage 1 + stage 2). Used for training, where both
    the filter and the main model are learned; identical values to the cascade."""
    return compute_stage2_features(compute_stage1_features(pairs, workers, chunk), workers, chunk)


def _top_feats(rows):
    """Worker: candidate-vs-strongest-candidate similarities for a chunk of pairs."""
    out = np.zeros((len(rows), 3), dtype=np.float32)
    for i, (cn, ca, cnum, tn, ta, tnum) in enumerate(rows):
        out[i] = (fuzz.token_set_ratio(cn, tn), fuzz.token_set_ratio(ca, ta),
                  float(cnum != "" and cnum == tnum))
    return out


def add_cluster_features(pairs, workers=10, chunk=100000):
    """Add the cluster-support features. `pairs` must contain whole entities
    (all candidates of each S1), as produced by chunking on s1_id."""
    top = pairs.filter(pl.col("rank") == pl.col("rank").min().over("s1_id")) \
        .unique("s1_id", keep="first") \
        .select("s1_id", pl.col("c_name_core").alias("t_name"),
                pl.col("c_addr_norm").alias("t_addr"), pl.col("c_addr_nums").alias("t_nums"))
    pairs = pairs.join(top, on="s1_id", how="left")
    cols = ["c_name_core", "c_addr_norm", "c_addr_nums", "t_name", "t_addr", "t_nums"]
    rows = list(zip(*[pairs[c].fill_null("").to_list() for c in cols]))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers) as pool:
        mats = pool.map(_top_feats, chunks)
    m = np.vstack(mats) if mats else np.zeros((0, 3), np.float32)
    pairs = pairs.with_columns(
        pl.Series("top_name_tset", m[:, 0]), pl.Series("top_addr_tset", m[:, 1]),
        pl.Series("top_nums_eq", m[:, 2]),
        (pl.len().over(["s1_id", "c_name_compact"]) - 1).cast(pl.Float32).alias("n_same_compact"),
        pl.when(pl.col("c_addr_nums") != "")
          .then(pl.len().over(["s1_id", "c_addr_nums"]) - 1).otherwise(0)
          .cast(pl.Float32).alias("n_same_nums"),
    ).drop("t_name", "t_addr", "t_nums")
    return pairs
