# Submission log

Every leaderboard upload, with the exact code version (git tag) and settings that produced it.
Validation = 60,000 held-out train entities (entity-level split), official macro F0.5.

| # | Date (IST) | Git tag | Model | Blocking | Threshold | Validation F0.5 | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 25 Sep | sub-01 | v1: LightGBM, 22 pair features | word TF-IDF name+address, top-20, max_df 0.02 | 0.65 | 0.9402 (P 98.4% / R 87.2%) | _pending_ | Baseline. 113,487 empty rows (6.6%); 3.06 matches/entity |
| 2 | 25 Sep | sub-02 | same as #1 | same as #1 | 0.80 | 0.9386 | _pending_ | Calibration: more conservative, tests whether test has more decoys than train. 121,821 empty rows; 2.98 matches/entity |
| 3 | 25 Sep | sub-03 | v4: two-stage (candidate filter + LightGBM), 56 features incl. competition, cluster, rarity, legal-form | word TF-IDF name+address top-30 + exact-key passes (address, compact name, empty-address names); new cleaning v3 + Indian-script dictionary; filter keeps 99.5% | per-entity expected F0.5 (empty weight 2.0) | 0.9735 (P 99.4% / R 94.2%; India 0.968, US 0.977) | _pending_ | 6.13 candidates/entity (was 20); 3.46 matches/entity; 90,714 empty rows |
| 4 | 25 Sep | sub-04 | v5: same as v4, trained on 450K entities (was 250K) | same candidates as sub03 | global threshold 0.70 | 0.9749 (P 99.4% / R 94.5%; India 0.970, US 0.978) | **0.935552** | address fix: dots no longer glue numbers to words ('No.301' -> 301; 10-14% of India addresses); 6.02 candidates/entity; 3.47 matches/entity |

| 5 | 26 Sep | sub-05 | v5 (unchanged) | France re-blocked WITHOUT the common-word shortcut (it dropped city names: 'pessac' 4.8%, 'nouvelle aquitaine' 11% of the France pool), merged with the old France candidates; US/India unchanged | global threshold 0.70 | n/a (France has no labels) | _pending_ | controlled experiment: only France changes (15.5% of French entities' answers differ); 6.06 candidates/entity |
| 6 | 26 Sep | sub-06 | v6: cleaning fixes (France regions/departements, key-name collisions, sound-alike mapping of Indian-script words unseen in train, skeleton similarity feature), 450K entities | full re-blocking with current cleaning; France without the common-word shortcut; exact-key passes | global threshold 0.725 | 0.9746 (P 99.4% / R 94.3%) | **0.936795** | vs sub05: 18% of French, 11% of Indian, 5% of US entities changed; 6.19 candidates/entity |
## How each file was produced

```bash
python src/build_cache.py
python src/run_blocking.py train 20 200000
python src/train_model.py cache/train_cands_k20_s200000.parquet artifacts/v1
python src/run_blocking.py test 20
python src/predict.py cache/test_cands_k20.parquet artifacts/v1 output/sub01          # threshold from model (0.65)
python src/predict.py cache/test_cands_k20.parquet artifacts/v1 output/sub02 0.80     # (sub02 was written from sub01's saved scores)
```

sub03:
```bash
python src/build_cache.py && python src/learn_translit.py && python src/build_cache.py
python src/run_blocking.py train 30 && python src/run_blocking.py test 30
python src/add_key_candidates.py train cache/train_cands_k30.parquet 20
python src/add_key_candidates.py test cache/test_cands_k30.parquet 20
python src/train_model.py cache/train_cands_k30_plus.parquet artifacts/v4 --full --n-train 250000 --n-val 80000
python src/predict.py cache/test_cands_k30_plus.parquet artifacts/v4 output/sub03
```
