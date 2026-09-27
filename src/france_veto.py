"""Drop French matches that the rescue rules added but the calibrated model rejects.

Usage: python src/france_veto.py <in_dir> <out_dir> <france_pc.parquet> [--gamma 2.55] [--min 0.78]

France has no labels, so its matches below first-stage 0.97 come from rescue rules with rates
borrowed from the US. Each is re-scored by the prior-corrected model encoded as US
(prior_stack.py --apply-to=France:US) and calibrated with the curve measured on the public
leaderboard (sub35: pairs the model rated 0.516 were ~18.5% true -> true = prob ** 2.55).
Those below the F0.5 break-even (~0.78) are removed.
"""

import argparse
import os
import sys

sys.path.insert(0, "src")

import polars as pl

from ber.calibrate import owned
from ber.io import load_source
from toggle import read_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir"); ap.add_argument("france_pc")
    ap.add_argument("--gamma", type=float, default=2.55); ap.add_argument("--min", type=float, default=0.78)
    ap.add_argument("--scored", default="output/v8_raw/scored_pairs.parquet")
    a = ap.parse_args()
    fr = load_source("test", 1).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1_id"))
    m = read_lists(os.path.join(a.in_dir, "matching_results.tsv")).join(fr, on="s1_id")
    v8 = owned(pl.read_parquet(a.scored, columns=["s1_id", "cand_id", "prob"]))
    pcf = pl.read_parquet(a.france_pc).select("s1_id", "cand_id", "pc")
    x = m.join(v8, on=["s1_id", "cand_id"], how="left").join(pcf, on=["s1_id", "cand_id"], how="left")
    drop = x.filter((pl.col("prob") < 0.97) & ((pl.col("pc").fill_null(0.5) ** a.gamma) < a.min)).select("s1_id", "cand_id")
    tmp = os.path.join(a.out_dir + "_drop.parquet")
    drop.write_parquet(tmp)
    import subprocess
    subprocess.run([sys.executable, "src/toggle.py", a.in_dir, a.out_dir, "--remove", tmp], check=True)
    os.remove(tmp)
    print(f"France: {m.height:,} matches, {drop.height:,} rescued matches below calibrated {a.min} removed")


if __name__ == "__main__":
    main()
