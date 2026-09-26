"""Predict the leaderboard score of a decision policy (US + India part), from labels.

Validation entities (held out from the first-stage model) carry true labels. Their
negatives are re-weighted to test levels per group (src/prior_stack.py weights), which
emulates test: per entity, TP = selected true pairs, FP = weighted selected negatives,
T = all true matches of the entity (including ones blocking never found), and
    F0.5 = 1.25 TP / (0.25 T + TP + FP);   an entity with no true match scores exp(-FP).
Averaged over ALL validation entities (the leaderboard's macro average). The emulated
values are anchored to real leaderboard scores of two policies (sub08, sub22):
    LB(X) = LB(sub22) + k * (E(X) - E(sub22)),  k = (LB22 - LB08) / (E22 - E08).
"""

import json
import sys

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from ber.io import load_ground_truth

LB08, LB22 = 0.956511, 0.963841


def main():
    v = pl.read_parquet("cache/test_pc3_v8_val_oof.parquet")
    c = pl.read_parquet("cache/train_cands_k30_plus.parquet", columns=["s1_id"]).unique()
    e = np.array(sorted(c["s1_id"].to_list())); np.random.default_rng(7).shuffle(e); va = e[2000000:2150000].tolist()
    gt, _ = load_ground_truth()
    T = gt.filter(pl.col("s1_id").is_in(va)).group_by("s1_id").len().rename({"len": "T"})
    ents = pl.DataFrame({"s1_id": va}).join(T, on="s1_id", how="left").fill_null(0)
    rules = pl.read_parquet("cache/rules_v8e/group_rules.parquet").select("band", "country", "nc", "nk", "action")
    bands = json.load(open("cache/rules_v8e/bands.json"))
    b = pl.lit(None, dtype=pl.Int64)
    for i, (lo, hi) in enumerate(bands):
        b = pl.when((pl.col("prob") >= lo) & (pl.col("prob") < hi)).then(pl.lit(i)).otherwise(b)
    v = v.with_columns(b.alias("band").cast(pl.Int32)).join(rules.with_columns(pl.col("band").cast(pl.Int32)),
                                                            on=["band", "country", "nc", "nk"], how="left") \
         .with_columns(pl.col("action").fill_null(""), pl.col("nc").fill_null(""))
    neigh = pl.col("nc").is_in(["conflict:one_digit_sub_small", "conflict:near_value", "conflict:transposed", "conflict:na"]) & (pl.col("country") == "US")
    typo = pl.col("nc").is_in(["conflict:digit_added_or_lost", "conflict:one_digit_sub_big", "conflict:zeros"]) & (pl.col("prob") >= 0.8664)
    policies = {
        "sub08": pl.col("prob") >= 0.8664,
        "sub15": pl.col("prob") >= 0.97,
        "sub22": (((pl.col("prob") >= 0.97) & ~neigh) | typo | (pl.col("action") == "rescue")) & (pl.col("action") != "veto"),
        "sub26": pl.col("oof") >= 0.85,
    }
    E = {}
    for name, rule in policies.items():
        sel = v.filter(rule)
        per = sel.group_by("s1_id").agg((pl.col("label") == 1).sum().alias("tp"), ((1 - pl.col("label")) * pl.col("w")).sum().alias("fp"))
        d = ents.join(per, on="s1_id", how="left").fill_null(0)
        f = pl.when(pl.col("T") > 0).then(1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("tp") + pl.col("fp"))) \
              .otherwise((-pl.col("fp")).exp())
        E[name] = float(d.select(f.mean()).item())
    k = (LB22 - LB08) / (E["sub22"] - E["sub08"])
    print(f"anchor: emulated sub08 {E['sub08']:.4f}, sub22 {E['sub22']:.4f} -> scale k = {k:.3f}")
    for name in policies:
        pred = LB22 + k * (E[name] - E["sub22"])
        print(f"{name}: emulated {E[name]:.4f}  -> predicted leaderboard {pred:.4f}" + ("   (real " + str({"sub08": LB08, "sub22": LB22}[name]) + ")" if name in ("sub08", "sub22") else ""))


if __name__ == "__main__":
    main()
