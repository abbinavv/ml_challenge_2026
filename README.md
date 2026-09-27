# Business Entity Resolution — Amazon ML Challenge 2026 (team Dronaut)

For every Source-1 business record, find the Source-2/3 records that describe the same
real-world business, using only name, address and country. Scored by macro F0.5 per
Source-1 entity (singletons included).

Pipeline: **normalise → block (candidate generation) → LightGBM pair model (two-stage
cascade) → label-free test calibration → decision rules → submission files.**

Final submission: this pipeline (sub41), public leaderboard **0.973202** (27 Sep 2026).
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

# 6. Cross-fitted first stage (EC2, ~2.5 h): two half-models, each scores the other half of
#    train and the test set -> cache/crossfit/cf{0,1}/ ; then the second-stage inputs
bash aws/ec2_crossfit.sh <bucket>          # results copied to cache/crossfit/cf0, cf1
$PY src/crossfit_prep.py

# 7. Prior-corrected second stage for US/India on 898K cross-fitted entities: fine pair
#    features + rival, list-context and record-to-record cohesion features, first-stage
#    scores down to 0.02, 127 leaves x 800 rounds. --street: validation-side tags split
#    far/no-number groups by street agreement (as for every submitted second stage).
$PY src/prior_stack.py cache/crossfit/oof_sample.parquet cache/crossfit/test_avg_scored.parquet \
    cache/crossfit/test_tags_cf_low.parquet cache/test_pc_final.parquet \
    --street --context --cohesion --floor=0.02 --gbm=127,800 --n-val-json=cache/crossfit/oof_sample_counts.json
#    France (no labels) scored by the base second stage encoded as US, for step 8b
$PY src/prior_stack.py cache/crossfit/oof_sample.parquet cache/crossfit/test_avg_scored.parquet \
    cache/crossfit/test_tags_cf.parquet cache/test_pc_cf.parquet \
    --street --apply-to=France:US --n-val-json=cache/crossfit/oof_sample_counts.json

# 8. Final decisions and both submission files. The cut-off comes from the leaderboard: model
#    probabilities are over-confident on test (sub35, sub37: true rate ~ prob^5-8 after sub40), so 0.94 is
#    the F0.5 optimum; 8b drops French rule-rescued matches below that calibration and French
#    matches the model rates below 0.9.
$PY src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/final_base \
    --keep 0.99 --threshold 0.97 --word-veto --neighbour-veto US France \
    --typo-rescue 0.8664 --group-rules cache/rules_v8e \
    --calibrated cache/test_pc_final.parquet --calibrated-min 0.94
$PY src/france_veto.py output/final_base output/final cache/test_pc_cf_France.parquet --main-min 0.9

# 8c. Package: output/ + code/ + documentation in the organisers' zip layout
$PY src/make_package.py output/final Dronaut

# 9. Official checker
python3 data/utils/validate_submission.py \
    --matching output/final/matching_results.tsv \
    --candidate output/final/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids
```

The main second stage (step 7, first command) and steps 8-8c are deterministic (LightGBM
deterministic mode, fixed tie-breaks): repeated runs give identical files. The France scoring
run (step 7, second command) was made before the determinism fix; a re-run can differ in a
handful of French decisions.

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
| `src/ber/rivals.py` | Rival (other S1s listing a candidate), list-context and record-to-record cohesion features |
| `src/ber/metrics.py` | Official macro F0.5 (singleton rule included) |
| `src/learn_translit.py` | Learns the Indian-script word dictionary and sound-alike map |
| `src/run_blocking.py`, `src/reblock_country.py`, `src/add_key_candidates.py` | Candidate generation |
| `src/train_model.py` | Entity-level split, candidate filter + main LightGBM, decision report |
| `src/predict.py` | Cascade scoring of test candidates |
| `src/score_validation.py` | Re-scores held-out entities with features and labels |
| `src/group_rules.py` | Label-free rescue / veto rules per (country, score band, number relation, name change) |
| `src/crossfit_prep.py` | Samples the cross-fitted train scores, averages the test scores, tags test pairs |
| `src/prior_stack.py` | Prior-corrected second stage (test-level negative weights; rival, context and cohesion features) |
| `src/france_veto.py` | Drops French rule-rescued matches the calibrated model rejects |
| `src/make_package.py` | Builds the submission zip (runs the official checker, scans for credentials) |
| `src/finalize.py` | Candidate set, one-owner, threshold, rules, calibrated decision, sibling expansion, files |
| `aws/ec2_train.sh`, `aws/ec2_crossfit.sh` | EC2 training / cross-fitting runs |

## Rules compliance

- **No external data**: only the provided files. Dictionaries are learned from the
  training data; test inputs are used without labels (blocking, pair densities).
- **Model licence**: LightGBM (MIT); no pretrained models.
- **Open set of countries**: blocking runs per country, features are country-agnostic;
  France (test-only) gets no country-specific model.

`submissions/LOG.md` records every leaderboard upload with its exact command.
