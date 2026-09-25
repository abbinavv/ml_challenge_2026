"""Reading the challenge TSVs, caching normalised records, writing submission files.

Raw files are parsed by splitting on TAB only (no CSV quote handling), so the
483 test names that contain CSV-style quotes are read exactly as written, the
same way in every file.
"""

import os
from multiprocessing import Pool

import polars as pl

from .normalize import normalize_address, normalize_name

DATA_DIR = os.environ.get("BER_DATA", "data/dataset")
CACHE_DIR = os.environ.get("BER_CACHE", "cache")

SPLITS = ("train", "test")


def source_path(split, source):
    """Path of a raw source file, e.g. source_path('test', 2)."""
    return os.path.join(DATA_DIR, split, f"{split}_source{source}.tsv")


def read_raw(path):
    """Read a 4-column source TSV into lists (ids, names, addresses, countries)."""
    ids, names, addrs, countries = [], [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            ids.append(eid)
            names.append(name)
            addrs.append(addr)
            countries.append(country)
    return ids, names, addrs, countries


def _normalize_chunk(rows):
    """Worker: normalise a chunk of (name, address, country) tuples."""
    out = []
    for name, addr, country in rows:
        norm, core, compact, key, alt = normalize_name(name)
        a_norm, a_nums, a_key = normalize_address(addr, country)
        non_latin = any(ord(ch) > 0x24F for ch in name)
        out.append((norm, core, compact, key, alt, a_norm, a_nums, a_key, non_latin))
    return out


def normalize_records(names, addrs, countries, workers=10, chunk=20000):
    """Normalise all records in parallel; returns a polars DataFrame of derived fields."""
    rows = list(zip(names, addrs, countries))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_normalize_chunk, chunks)
    flat = [r for p in parts for r in p]
    return pl.DataFrame(
        flat,
        schema=["name_norm", "name_core", "name_compact", "name_key", "name_alt",
                "addr_norm", "addr_nums", "addr_key", "non_latin"],
        orient="row",
    )


def cache_path(split, source):
    """Parquet cache path for one source file."""
    return os.path.join(CACHE_DIR, f"{split}_s{source}.parquet")


def build_source_cache(split, source):
    """Parse + normalise one source file and write it to the Parquet cache."""
    ids, names, addrs, countries = read_raw(source_path(split, source))
    derived = normalize_records(names, addrs, countries)
    df = pl.DataFrame({"entity_id": ids, "name": names, "address": addrs, "country": countries})
    df = pl.concat([df, derived], how="horizontal")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.write_parquet(cache_path(split, source))
    return df


def load_source(split, source):
    """Load one cached source (builds the cache on first use)."""
    path = cache_path(split, source)
    if not os.path.exists(path):
        return build_source_cache(split, source)
    return pl.read_parquet(path)


def load_ground_truth():
    """Train ground truth as a long table of (s1_id, match_id) pairs, plus all S1 ids."""
    path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    s1_ids, pairs_s1, pairs_m = [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, ids = line.rstrip("\n").split("\t")
            s1_ids.append(s1)
            for m in ids.split(","):
                if m:
                    pairs_s1.append(s1)
                    pairs_m.append(m)
    return pl.DataFrame({"s1_id": pairs_s1, "match_id": pairs_m}), s1_ids


def write_id_lists(path, header_col, s1_ids, lists):
    """Write a submission-format TSV.

    One row per S1 id in `s1_ids` (order preserved). `lists` maps s1 id -> iterable of
    S2/S3 ids. Rows with no ids still contain the TAB. IDs are joined with ',' and no
    spaces; de-duplicated; LF line endings; UTF-8.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1 in s1_ids:
            ids = lists.get(s1, ())
            seen = list(dict.fromkeys(i for i in ids if i.startswith(("S2-", "S3-"))))
            f.write(f"{s1}\t{','.join(seen)}\n")
