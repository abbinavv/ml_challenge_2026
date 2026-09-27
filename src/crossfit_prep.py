"""Prepare the cross-fitted inputs of the second stage (after aws/ec2_crossfit.sh).

Usage: python src/crossfit_prep.py
Reads  cache/crossfit/cf{0,1}/oof_cf{0,1}.parquet   (out-of-sample train scores, each half-model
                                                     scored the other half) and
       cache/crossfit/cf{0,1}/test_scored_pairs.parquet
Writes cache/crossfit/oof_sample.parquet            898K train entities: the 150K v8 validation
                                                     entities + 750K random others (memory: 16 GB)
       cache/crossfit/oof_sample_counts.json        entities per country in that sample
       cache/crossfit/test_avg_scored.parquet       test scores averaged over the two half-models
       cache/crossfit/test_tags_cf.parquet          test pair tags (owner pairs, prob >= 0.3)
       cache/crossfit/test_tags_cf_low.parquet      test pair tags (owner pairs, prob >= 0.02)
"""

import importlib.util
import json
import sys

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from ber.calibrate import owned
from ber.io import load_source


def main():
    c = pl.read_parquet("cache/train_cands_k30_plus.parquet", columns=["s1_id"]).unique()
    e = np.array(sorted(c["s1_id"].to_list())); np.random.default_rng(7).shuffle(e)
    va = set(e[2000000:2150000].tolist())
    rest = [x for x in e.tolist() if x not in va]
    extra = np.random.default_rng(11).choice(rest, size=750000, replace=False).tolist()
    keep = pl.DataFrame({"s1_id": list(va) + extra})
    cols = ["s1_id", "cand_id", "prob", "p1", "label"]
    v = pl.concat([pl.read_parquet(f"cache/crossfit/cf{h}/oof_cf{h}.parquet", columns=cols) for h in (0, 1)]).join(keep, on="s1_id")
    v.write_parquet("cache/crossfit/oof_sample.parquet")
    s1c = load_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    json.dump(dict(keep.join(s1c, on="s1_id").group_by("country").len().iter_rows()), open("cache/crossfit/oof_sample_counts.json", "w"))
    t = pl.concat([pl.read_parquet(f"cache/crossfit/cf{h}/test_scored_pairs.parquet", columns=["s1_id", "cand_id", "prob", "p1"]) for h in (0, 1)]) \
          .group_by("s1_id", "cand_id").agg(pl.col("prob").mean(), pl.col("p1").mean())
    t.write_parquet("cache/crossfit/test_avg_scored.parquet")
    spec = importlib.util.spec_from_file_location("gr", "src/group_rules.py"); gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    for floor, name in ((0.3, "test_tags_cf"), (0.02, "test_tags_cf_low")):
        to = owned(t).filter(pl.col("prob") >= floor)
        gr.tag(to.select("s1_id", "cand_id", "prob"), "test").select("s1_id", "cand_id", "nk", "nc").write_parquet(f"cache/crossfit/{name}.parquet")
    print("sample entities", v["s1_id"].n_unique(), "pairs", v.height, "| test pairs", t.height)


if __name__ == "__main__":
    from ber.guard import exclusive
    exclusive("crossfit_prep")
    main()
