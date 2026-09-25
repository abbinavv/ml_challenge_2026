"""Error analysis on validation entities: which true matches do we lose, and why?

Usage: python src/error_analysis.py <train_cands.parquet> <model_dir> [n_entities]
Shows (1) true matches that blocking never proposed and (2) true matches the model
rejected, as raw name/address side by side, plus false merges.
"""
import json, os, random, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from ber.decide import one_owner
from ber.features import add_group_features
from ber.io import load_ground_truth, load_source
from train_model import featurize

cands_path, model_dir = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
meta = json.load(open(os.path.join(model_dir, "meta.json")))
feats = meta["features"]
m1 = lgb.Booster(model_file=os.path.join(model_dir, "stage1.txt"))
m2 = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
cands = add_group_features(pl.read_parquet(cands_path))
s1 = load_source("train", 1); pool = pl.concat([load_source("train", 2), load_source("train", 3)])
gt, _ = load_ground_truth()
ents = sorted(cands["s1_id"].unique().to_list()); random.seed(5); ents = random.sample(ents, n)
df = featurize(cands.filter(pl.col("s1_id").is_in(ents)), s1, pool, feats)
lab = gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(1).alias("label"))
df = df.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
df = df.with_columns(pl.Series("p1", m1.predict(df.select(meta["stage1"]["features"]).to_numpy())))
df = df.with_columns(pl.Series("prob", m2.predict(df.select(feats).to_numpy())))
kept = df.filter(pl.col("p1") >= meta["stage1"]["cutoff"])
chosen = one_owner(kept).filter(pl.col("prob") >= meta["decision"].get("threshold", 0.7))
raw = pl.concat([s1, pool]).select("entity_id", "name", "address", "country")
R = {r[0]: r[1:] for r in raw.filter(pl.col("entity_id").is_in(
    list(set(gt.filter(pl.col("s1_id").is_in(ents))["match_id"]) | set(ents) | set(df["cand_id"])))).iter_rows()}
truth = gt.filter(pl.col("s1_id").is_in(ents))
tp = set(zip(truth["s1_id"], truth["match_id"]))
blocked = set(zip(df["s1_id"], df["cand_id"])); filt = set(zip(kept["s1_id"], kept["cand_id"]))
got = set(zip(chosen["s1_id"], chosen["cand_id"]))
miss_block = [p for p in tp if p not in blocked]
miss_filter = [p for p in tp if p in blocked and p not in filt]
miss_model = [p for p in tp if p in filt and p not in got]
false_pos = [p for p in got if p not in tp]
print(f"true pairs {len(tp):,}: lost by blocking {len(miss_block)/len(tp):.1%}, by filter "
      f"{len(miss_filter)/len(tp):.1%}, by model {len(miss_model)/len(tp):.1%}; false merges {len(false_pos):,}")
def show(title, ps, k=14):
    print(f"\n=== {title} ({len(ps):,}) ===")
    for a, b in random.sample(ps, min(k, len(ps))):
        print(f"  [{R[a][2]}] S1: {R[a][0]!r} | {R[a][1]!r}\n        {b[:2]}: {R[b][0]!r} | {R[b][1]!r}")
show("MISSED BY BLOCKING", miss_block)
show("REJECTED BY MODEL", miss_model)
show("FALSE MERGES", false_pos, 10)
