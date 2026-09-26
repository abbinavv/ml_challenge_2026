"""Leaderboard prediction for US/India decision changes, anchored on sub22 -> sub26.

sub22 and sub26 have identical French answers, so their leaderboard difference
(0.963841 -> 0.967148) is caused only by US/India decisions. The label-based simulation
(validation entities, negatives re-weighted to test levels per group and score band,
src/prior_stack.py weights) gives E(policy); the scale k = dLB / dE from that pair
converts any other US/India policy into a predicted leaderboard score.
"""

import json
import sys

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from ber.io import load_ground_truth

LB22, LB26 = 0.963841, 0.967148


def main():
    w = pl.read_parquet("cache/test_pc3_v8_val_oof.parquet")          # evaluation weights + pc3 OOF
    o2 = pl.read_parquet("cache/test_pc2_v8_val_oof.parquet").select("s1_id", "cand_id", pl.col("oof").alias("oof2"))
    v = w.join(o2, on=["s1_id", "cand_id"], how="left")
    c = pl.read_parquet("cache/train_cands_k30_plus.parquet", columns=["s1_id"]).unique()
    e = np.array(sorted(c["s1_id"].to_list())); np.random.default_rng(7).shuffle(e); va = e[2000000:2150000].tolist()
    gt, _ = load_ground_truth()
    ents = pl.DataFrame({"s1_id": va}).join(gt.filter(pl.col("s1_id").is_in(va)).group_by("s1_id").len().rename({"len": "T"}), on="s1_id", how="left").fill_null(0)
    rules = pl.read_parquet("cache/rules_v8e/group_rules.parquet").select("band", "country", "nc", "nk", "action")
    bands = json.load(open("cache/rules_v8e/bands.json"))
    b = pl.lit(None, dtype=pl.Int64)
    for i, (lo, hi) in enumerate(bands):
        b = pl.when((pl.col("prob") >= lo) & (pl.col("prob") < hi)).then(pl.lit(i)).otherwise(b)
    v = v.drop("band").with_columns(b.alias("band").cast(pl.Int32)).join(rules.with_columns(pl.col("band").cast(pl.Int32)), on=["band", "country", "nc", "nk"], how="left") \
         .with_columns(pl.col("action").fill_null(""), pl.col("nc").fill_null(""))
    neigh = pl.col("nc").is_in(["conflict:one_digit_sub_small", "conflict:near_value", "conflict:transposed", "conflict:na"]) & (pl.col("country") == "US")
    typo = pl.col("nc").is_in(["conflict:digit_added_or_lost", "conflict:one_digit_sub_big", "conflict:zeros"]) & (pl.col("prob") >= 0.8664)
    sub22 = (((pl.col("prob") >= 0.97) & ~neigh) | typo | (pl.col("action") == "rescue")) & (pl.col("action") != "veto")

    def emu(rule):
        per = v.filter(rule).group_by("s1_id").agg((pl.col("label") == 1).sum().alias("tp"), ((1 - pl.col("label")) * pl.col("w")).sum().alias("fp"))
        d = ents.join(per, on="s1_id", how="left").fill_null(0)
        return float(d.select(pl.when(pl.col("T") > 0).then(1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("tp") + pl.col("fp")))
                              .otherwise((-pl.col("fp")).exp()).mean()).item())
    E22, E26 = emu(sub22), emu(pl.col("oof2") >= 0.85)
    k = (LB26 - LB22) / (E26 - E22)
    print(f"anchor: E(sub22) {E22:.4f}, E(sub26) {E26:.4f} -> k = {k:.3f}")
    cands = {"sub22 rules": sub22, "sub26 (pc2 >= 0.85)": pl.col("oof2") >= 0.85}
    for t in (0.5, 0.6, 0.7, 0.78, 0.85, 0.9):
        cands[f"pc2 >= {t}"] = pl.col("oof2") >= t
        cands[f"pc3 >= {t}"] = pl.col("oof") >= t
    for name, rule in cands.items():
        E = emu(rule)
        print(f"{name:22s} E {E:.4f} -> predicted leaderboard {LB26 + k * (E - E26):.4f}")


if __name__ == "__main__":
    main()
