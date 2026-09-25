# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business record, find the Source-2/3 records that describe the same
real-world business, using only name, address and country. Scored by macro F0.5
(per Source-1 entity, precision-weighted).

Pipeline: **normalise → block (candidate generation) → pair features → LightGBM →
one-owner rule + F0.5-tuned decision → submission files.**

## Environment

- Python 3.12, dependencies pinned in `requirements.lock.txt`
- Tested on Apple M5 (10 cores, 16 GB RAM). Peak memory stays within 16 GB when the
  steps below are run **one at a time**.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

## Data

Put the official `student_resource/` bundle anywhere and point `data` at it:

```bash
ln -s /path/to/student_resource data      # expects data/dataset/{train,test}/*.tsv
```

(`BER_DATA` can override the dataset path; `BER_CACHE` the cache folder.)

## Reproduce end to end

Run from the repository root, one step at a time:

```bash
PY=.venv/bin/python

# 1. Parse + normalise all six source files into the Parquet cache (~1 min)
$PY src/build_cache.py

# 2. Learn the Indian-script word dictionary from train pairs, then re-normalise
#    so names written in Indian scripts use it (~2 min)
$PY src/learn_translit.py
$PY src/build_cache.py

# 3. Blocking: top-30 candidates for every S1 record, train and test (~35 min each)
$PY src/run_blocking.py train 30
$PY src/run_blocking.py test 30

# 4. Train the pair model and choose the decision rule on validation (~10 min)
$PY src/train_model.py cache/train_cands_k30.parquet artifacts/v2 --full

# 5. Score test and write both submission files (~10 min)
$PY src/predict.py cache/test_cands_k30.parquet artifacts/v2 output/final

# 6. Validate with the official checker
python3 data/utils/validate_submission.py \
    --matching output/final/matching_results.tsv \
    --candidate output/final/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids
```

Optional safety net for long runs on 16 GB machines: `sh src/memguard.sh 12 &`
stops the pipeline if free memory falls below 12%.

## Code map

| File | Role |
|---|---|
| `src/ber/normalize.py` | Transliteration, legal-form removal, compact names, address abbreviations, state codes, learned Indian-script dictionary |
| `src/ber/io.py` | Raw TSV parsing (tab-split only), Parquet cache, ground truth, submission writer |
| `src/ber/blocking.py` | Per-country word TF-IDF search over name + address, top-K candidates |
| `src/ber/features.py` | 31 country-agnostic pair features: blocking, competition, name, address, record, cluster support |
| `src/ber/decide.py` | One-owner rule, global threshold, per-entity expected-F0.5 selection |
| `src/ber/metrics.py` | Official macro F0.5 (singleton rule included), self-tested on the worked example |
| `src/learn_translit.py` | Learns the Indian-script word dictionary from training pairs |
| `src/train_model.py` | Entity-level split, LightGBM training, decision-rule selection, per-country report |
| `src/predict.py` | Scores test candidates, writes `matching_results.tsv` and `candidate_pairs.tsv` |
| `src/eval_country_transfer.py` | Leave-one-country-out check (proxy for the unseen country, France) |

## Rules compliance

- **No external data**: only the provided files; the Indian-script dictionary is
  learned from the training ground truth.
- **Model licence**: LightGBM (MIT); no pretrained models.
- **Open set of countries**: blocking runs per country and features have no country
  one-hot, so France (test-only) is handled like any other country.

See `context.md` (verified data facts), `masterplan.md` (plan and design review) and
`submissions/LOG.md` (every leaderboard upload).
