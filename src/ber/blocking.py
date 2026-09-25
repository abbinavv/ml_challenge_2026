"""Candidate generation (blocking).

For each country separately (matches never cross countries in train), every
Source-1 record is compared with every Source-2/3 record of the same country via
word-token TF-IDF over "core name + normalised address", and the top-K most
similar records become its candidates. Combined name+address text is essential:
name-only blocking keeps just 72% (US) of true pairs at K=50 because many decoys
share a name but not an address.

Word tokens instead of character 3-grams: same recall (US 97.8% vs 98.0% at K=20)
at ~7x the speed. Words in more than `max_df` of a country's pool (street, road,
state codes...) are dropped; at 0.02 this costs ~0.5 (US) / ~1.2 (India) points of
recall at K=20 for another ~3.5x speed-up (measured on 5K labelled queries).

The search is a sparse matrix product done in chunks of queries, so memory stays
bounded; each country is independent, so unseen countries (France) work the same way.
"""

import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def blocking_text(df):
    """The string each record is indexed by: core name + normalised address."""
    return (df["name_core"] + " " + df["addr_norm"]).to_list()


def block_country(queries, pool, top_k=20, max_df=0.02, chunk=50000, threads=10, log=print):
    """Top-K candidates for every query record within one country.

    queries, pool : polars DataFrames (cached source schema) of one country
    Returns a DataFrame (s1_id, cand_id, cos, rank) with up to top_k rows per query.
    """
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=max_df,
                          dtype=np.float32, sublinear_tf=True)
    pool_m = vec.fit_transform(blocking_text(pool))
    pool_t = pool_m.T.tocsr()
    del pool_m
    q_m = vec.transform(blocking_text(queries))
    log(f"    index built: pool={pool.height:,} queries={queries.height:,} "
        f"vocab={len(vec.vocabulary_):,} ({time.time() - t0:.0f}s)")

    # Collect integer positions only; IDs are attached once at the end as Arrow
    # strings. (Holding NumPy string arrays of IDs for ~66M pairs took ~6 GB.)
    out_q, out_p, out_cos, out_rank = [], [], [], []
    t1 = time.time()
    for start in range(0, q_m.shape[0], chunk):
        res = sp_matmul_topn(q_m[start:start + chunk], pool_t, top_n=top_k,
                             threshold=0.0, n_threads=threads, sort=True)
        counts = np.diff(res.indptr)
        out_q.append((np.repeat(np.arange(res.shape[0]), counts) + start).astype(np.int32))
        out_p.append(res.indices.astype(np.int32))
        out_cos.append(res.data.astype(np.float32))
        out_rank.append(np.concatenate([np.arange(1, c + 1, dtype=np.int16) for c in counts])
                        if counts.sum() else np.array([], np.int16))
        del res
        done = min(start + chunk, q_m.shape[0])
        rate = done / max(time.time() - t1, 1e-9)
        log(f"    {done:,}/{q_m.shape[0]:,} queries  ({rate:,.0f}/s, "
            f"eta {(q_m.shape[0] - done) / rate:.0f}s)")
    del pool_t, q_m
    q_idx = pl.Series(np.concatenate(out_q)) if out_q else pl.Series([], dtype=pl.Int32)
    p_idx = pl.Series(np.concatenate(out_p)) if out_p else pl.Series([], dtype=pl.Int32)
    return pl.DataFrame({
        "s1_id": queries["entity_id"].gather(q_idx),
        "cand_id": pool["entity_id"].gather(p_idx),
        "cos": np.concatenate(out_cos) if out_cos else np.array([], np.float32),
        "rank": np.concatenate(out_rank) if out_rank else np.array([], np.int16),
    })


def block_all(s1, pool, top_k=20, log=print, **kw):
    """Run blocking for every country present in the queries (open set of labels)."""
    parts = []
    for country in sorted(s1["country"].unique().to_list()):
        q = s1.filter(pl.col("country") == country)
        p = pool.filter(pl.col("country") == country)
        log(f"  [{country}] queries={q.height:,} pool={p.height:,}")
        if p.height == 0 or q.height == 0:
            continue
        parts.append(block_country(q, p, top_k=top_k, log=log, **kw))
    return pl.concat(parts) if parts else pl.DataFrame(
        schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8, "cos": pl.Float32, "rank": pl.Int16})
