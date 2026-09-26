# Business Entity Resolution — Amazon ML Challenge 2026 (team Dronaut)

For every Source-1 business record, find the Source-2/3 records that describe the same
real-world business, using only name, address and country. Scored by macro F0.5 per
Source-1 entity (singletons included).

Pipeline: **normalise → block (candidate generation) → LightGBM pair model (two-stage
cascade) → label-free test calibration → decision rules → submission files.**

Final submission: public leaderboard **0.967148** (27 Sep 2026).
`candidate_pairs.tsv` holds **5.30 candidates per Source-1 entity**.

## Environment

- Python 3.12; pinned runtime dependencies in `requirements.txt` (full environment: `requirements.lock.txt`)
- Local steps ran on an Apple M5 (10 cores, 16 GB RAM), one heavy step at a time
  (`src/ber/guard.py` enforces this). Model training ran on an AWS EC2 r7i.2xlarge
  (8 vCPU, 64 GB) with `aws/ec2_train.sh`; it also runs locally given ~48 GB of RAM.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Data

```bash
ln -s /path/to/student_resource data      # expects data/dataset/{train,test}/*.tsv
```

`BER_DATA` overrides the dataset path and `BER_CACHE` the cache folder.

## Reproduce end to end

Run from the repository root, one step at a time. `PY=.venv/bin/python`.

```bash
# 1. Parse + normalise the six source files; learn the Indian-script word dictionary
#    and the sound-alike map from the training data; re-normalise with them
$PY src/build_cache.py
$PY src/learn_translit.py
$PY src/build_cache.py

# 2. Blocking: per-country word TF-IDF, top-30 per Source-1 record (train and test);
#    France re-blocked without the common-word cut-off; exact-key passes added
$PY src/run_blocking.py train 30
$PY src/run_blocking.py test 30
$PY src/reblock_country.py test France cache/test_cands_k30.parquet 30 none cache/test_cands_k30_fr.parquet
$PY src/add_key_candidates.py train cache/train_cands_k30.parquet 20        # -> cache/train_cands_k30_plus.parquet
$PY src/add_key_candidates.py test cache/test_cands_k30_fr.parquet 20       # -> cache/test_cands_k30_fr_plus.parquet

# 3. Pair model v8: candidate filter + main LightGBM on 2.0M train entities, 150K held out
#    (EC2: aws/ec2_train.sh <bucket> v8 2000000 150000 --max-rounds 4000 --conflict-neg-weight 3)
$PY src/train_model.py cache/train_cands_k30_plus.parquet artifacts/v8 --full \
    --n-train 2000000 --n-val 150000 --max-rounds 4000 --conflict-neg-weight 3

# 4. Score test (cascade) and re-score the 150K held-out entities (labels kept)
$PY src/predict.py cache/test_cands_k30_fr_plus.parquet artifacts/v8 output/v8_raw
$PY src/score_validation.py cache/train_cands_k30_plus.parquet artifacts/v8 cache/val_v8_scored.parquet

# 5. Label-free test calibration: group rules from validation-vs-test pair densities
$PY src/group_rules.py cache/val_v8_scored.parquet output/v8_raw/scored_pairs.parquet cache/rules_v8e

# 6. Prior-corrected second stage for US/India (validation labels, test-level negatives).
#    --street: the validation-side tags split far/no-number groups by street agreement,
#    exactly as for the submitted file (test tags come from step 5, without that split).
$PY src/prior_stack.py cache/val_v8_scored.parquet output/v8_raw/scored_pairs.parquet \
    cache/rules_v8e/test_pair_tags.parquet cache/test_pc2_v8.parquet --street

# 7. Final decisions and both submission files
$PY src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/final \
    --keep 0.99 --threshold 0.97 --word-veto --neighbour-veto US France \
    --typo-rescue 0.8664 --group-rules cache/rules_v8e \
    --calibrated cache/test_pc2_v8.parquet --calibrated-min 0.85

# 8. Official checker
python3 data/utils/validate_submission.py \
    --matching output/final/matching_results.tsv \
    --candidate output/final/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids
```

Steps 5-7 were re-run from the saved step-4 outputs and reproduce the submitted files
exactly (5,695,612 matches, 9,186,087 candidate pairs, zero differences).

## Code map

| File | Role |
|---|---|
| `src/ber/normalize.py` | Transliteration, learned Indian-script dictionary, sound-alike map, legal forms, compact / key names, address canonicalisation, regions |
| `src/ber/io.py` | Tab-only TSV parsing, Parquet cache, ground truth, submission writer |
| `src/ber/blocking.py` | Per-country word TF-IDF top-K search over name + address |
| `src/ber/features.py` | 57 country-agnostic pair features (stage-1 cheap / stage-2 expensive) |
| `src/ber/decide.py` | One-owner rule, word-swap veto, legal-form conflict, house-number change kinds |
| `src/ber/calibrate.py` | Pair-density calibration helpers (validation vs test) |
| `src/ber/domain.py` | Cheap pair descriptors shared by the calibration steps |
| `src/ber/metrics.py` | Official macro F0.5 (singleton rule included) |
| `src/learn_translit.py` | Learns the Indian-script word dictionary and sound-alike map |
| `src/run_blocking.py`, `src/reblock_country.py`, `src/add_key_candidates.py` | Candidate generation |
| `src/train_model.py` | Entity-level split, candidate filter + main LightGBM, decision report |
| `src/predict.py` | Cascade scoring of test candidates |
| `src/score_validation.py` | Re-scores held-out entities with features and labels |
| `src/group_rules.py` | Label-free rescue / veto rules per (country, score band, number relation, name change) |
| `src/prior_stack.py` | Prior-corrected second stage (test-level negative weights) |
| `src/finalize.py` | Candidate set, one-owner, threshold, rules, calibrated decision, sibling expansion, files |
| `aws/ec2_train.sh`, `aws/ec2_crossfit.sh` | EC2 training / cross-fitting runs |

## Rules compliance

- **No external data**: only the provided files. Dictionaries are learned from the
  training data; test inputs are used without labels (blocking, pair densities).
- **Model licence**: LightGBM (MIT); no pretrained models.
- **Open set of countries**: blocking runs per country, features are country-agnostic;
  France (test-only) gets no country-specific model.

`submissions/LOG.md` records every leaderboard upload with its exact command.
