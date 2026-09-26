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

## Leaderboard experiments (26 Sep)

Same sub06 model scores, only the decision changed, to learn the direction of the
validation-vs-leaderboard gap:

| Upload | Decision | Matches/entity | Public LB |
|---|---|---|---|
| exp06_loose | threshold 0.45 | 3.64 | 0.920969 |
| sub06 | threshold 0.725 | 3.48 | 0.936795 |
| exp06_strict | threshold 0.90 | 3.34 | **0.948075** |

Conclusion: precision problem on test. The 0.725-0.90 band is 43% house-number
conflicts (same name + same street, different number: look-alike neighbours), vs 2.9%
of pairs above 0.97. Next: conflict-aware rule (src/conflict_rule.py) -- exp06_conflict97
(conflicts need >= 0.97, others 0.725; 3.36/entity) and exp06_conflict995 (>= 0.995; 3.31/entity).

## sub07 (26 Sep) -- v8 model trained on AWS EC2

Model v8: trained on r7i.2xlarge (aws/ec2_train.sh) with house-number-conflict negatives
weighted x3 (`--conflict-neg-weight 3`), 3573 rounds. Validation F0.5 0.9749
(P 0.9945, R 0.9442, singletons 0.9707). Test scored on EC2, finalized locally:

```
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub07 --keep 0.99 --target-matches 3.34
```

- candidate set: keep 99% (filter cut-off 0.0370) -> 5.30 candidates/entity
- threshold 0.8664 -> 3.34 matches/entity; sibling expansion +25,885 -> 3.35/entity
- official validator (--check-ids): PASS
- Public LB: (pending)

## sub08 (26 Sep) -- sub07 + word-swap veto

Finding: France (15% of test, no training labels) has 2x the co-located businesses of
train (13% of S1 share an address vs 6%; 2.36 pool records per address vs 1.1-1.5).
Test India/US match train on these statistics and on the mix of accepted pair types.
Implied from the leaderboard: France scores ~0.80 while India/US hold ~0.975.
French decoys = a neighbouring business at the same address with one business word
swapped ('Fontaine Club' vs 'Fontaine Amicale'). True variants only swap in filler words
(center/services/partners/groupe/...). The veto (src/ber/decide.py foreign_word) flags
0.02% of true train pairs vs 3.3% of French, 0.5% of Indian, 0.12% of US test matches.

```
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub08 --keep 0.99 --target-matches 3.34 --word-veto
```

- vetoed 44,515 matches (France 28,472 / India 13,215 / US 2,828) -> 3.33 matches/entity after siblings
- 2,246 entities become empty (all sampled were decoy-only: likely singletons, now scored 1.0)
- official validator (--check-ids): PASS
- Public LB: **0.956511** (best; +0.0084 over exp06_strict 0.948075 = v8 retrain + word-swap veto)
- Tried and rejected: "conflicting number shared by another candidate" rule (flags 65% of
  true conflicting pairs on train; not discriminative).

## Prepared 26 Sep for the 27 Sep uploads (one change each vs sub08, all PASS --check-ids)

| File | Change vs sub08 | Tests |
|---|---|---|
| sub09 | + French legal-form veto (-4,839: 'Maison Event SARL' vs 'SNC') | are form changes decoys in France? |
| sub10 | + France threshold 0.97 (-38,334) | are France's borderline pairs mostly wrong? |
| sub11 | global target 3.25 matches/entity (thr 0.9419) | has the stricter-is-better trend peaked? |
| sub12 | v6 scores instead of v8 (same finalize) | did the v8 retrain help on test? |

```
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub09 --keep 0.99 --word-veto --target-matches 3.34 --legal-veto France
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub10 --keep 0.99 --word-veto --target-matches 3.34 --country-threshold France=0.97
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub11 --keep 0.99 --word-veto --target-matches 3.25
python src/finalize.py output/sub06/scored_pairs.parquet artifacts/v6 output/sub12 --keep 0.99 --word-veto --target-matches 3.34
```
Checked and ruled out on 26 Sep: test singleton rate (5.7% vs 5.6% train), Indian-script
match rate (17.9% vs 18.0%), street agreement, S2/S3 split, conflicting-number-cluster rule.

## Evening 26 Sep: validation error map and rival-feature stacker (sub13, NOT recommended)

v8 validation re-scored locally (src/score_validation.py, reproduces 0.9749). Of 518,632
true pairs, lost: blocking 10,342 (2.0%), filter 2,541, one-owner 1,053, threshold 14,986 (2.9%).
Oracles (validation F0.5): no false positives 0.9800; + all shortlisted misses 0.9866;
perfect on the shortlist 0.9917 (blocking recall 97.5% caps it); perfect 1.0.
43% of threshold misses are records with a BLANK address; 98% of the "false" blank-address
same-name candidates are true matches of ANOTHER S1 with the same name (branches), i.e. an
assignment problem between namesakes. Raw-name fit picks the owner 31% of the time (random 12%).
No row-order / id leakage (correlations ~0.001).

Stacker (src/ber/rivals.py + src/stack_rivals.py): v8 prob + rival features (fit vs other
S1s listing the candidate) + list-context features. Half/half validation:
  @3.3 matches/entity: 0.9744 -> 0.9772;  @3.2 matches/entity: 0.9696 -> 0.9694 (no gain).
The gain exists only at looser operating points; the leaderboard rewards the strict one
(test threshold 0.87 vs 0.675 on validation). sub13 (stacked scores, sub08 settings) built
and PASS, but not recommended: ~80K pairs reshuffled, many near-number look-alikes.

## Label-free test calibration (26 Sep evening) -> sub15 recommended for the 27 Sep 00:00 upload

True variants are generated the same way in train and test; test adds decoys. So per score
band (or pair-type group) test precision ~= validation true pairs per S1 / test pairs per S1
(src/ber/calibrate.py). Top bands match exactly (India 2.53 vs 2.53 pairs/S1 above 0.999,
US 2.67 vs 2.68), mid bands are ~2x denser on test:
  v8 0.87-0.93: val precision 91-93% -> test ~41-55%;  0.93-0.97: 95-97% -> ~57-67%;
  0.97-0.99: 98-99% -> ~76-84%.  By pair type (US/India, prob 0.5-0.97): same name +
  conflicting number 18%, India shared number + different name 34%, ...
The absolute score estimate is NOT reliable on test (sub08 estimated 0.929 vs LB 0.9565), but
both analyses and the leaderboard trend say the 0.87 cut-off is too loose in every country.

| File | Change vs sub08 |
|---|---|
| sub15 | global threshold 0.97 (3.18 matches/entity) -- recommended first on 27 Sep |
| sub11 | global threshold 0.9419 (3.24/entity) |
| sub10 | France-only threshold 0.97 |

## sub16 (26 Sep evening): per-pair test calibration (domain classifier)

src/domain_adjust.py: classifier separating test pairs from validation pairs (US + India,
first-stage prob >= 0.3, pair-level features only; entity-level context features were
removed after they flagged obvious true pairs of decoy-rich businesses). Density ratio
r = D/(1-D); test-calibrated prob = prob / max(r, 1). Mean calibrated precision by band:
0.87-0.93: US 0.38 / India 0.52; 0.93-0.97: 0.53 / 0.66; 0.97-0.99: 0.76 / 0.80;
>0.999: 0.96 (share with r > 2: 0%). France keeps v8 prob with threshold 0.97.

```
python src/domain_adjust.py cache/val_v8_domain.parquet cache/test_v8_domain.parquet output/v8_raw/scored_pairs.parquet output/v8d/scored_pairs.parquet
python src/finalize.py output/v8d/scored_pairs.parquet artifacts/v8 output/sub16 --keep 0.99 --threshold 0.77 --country-threshold France=0.97 --word-veto
```
3.18 matches/entity; vs sub15 (same count): ~110K pairs swapped each way. PASS.
Upload order for 27 Sep: sub15 (global 0.97) first; sub16 second if sub15 confirms that
stricter is better.

## sub17 (26 Sep evening): house-number change types (label-free, measured)

For pairs whose house numbers conflict, the TYPE of change separates typos from
neighbours (validation truth vs test density, src/ber/decide.py number_change_kind):
  band 0.8664-0.97: digit added/lost val 97-98% -> test 95-97%; one digit changed, big jump
  98-100% -> ~100%; one digit changed within 9: 88-91% -> 11-26%; value within 20: 84-89% -> 9-10%.
  band >= 0.97 (US): one digit within 9: 99% -> 49%; value within 20: 99.8% -> 58%; na 93% -> 31%.
  India >= 0.97: 76-77% (break-even, left alone). France: no labels; all French decoys seen
  were neighbours (19 vs 22, 139 vs 142), so the US rule is applied.
```
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub17 --keep 0.99 --threshold 0.97 --word-veto --neighbour-veto US France --typo-rescue 0.8664
```
-28,030 neighbour matches (US 24.3K, France 3.7K), +13,830 typo matches; 3.17/entity. PASS.

## sub18 (26 Sep evening): label-free group rules (src/group_rules.py)

Per (country, score band, house-number relation, name-change kind): test true rate =
validation true pairs per S1 / test pairs per S1. Rescue mid-band (0.8664-0.97) groups with
test rate >= 0.85 (17 groups, e.g. India same number + trade-name 5,188 pairs ~1.00, US same
number + name typo 4,910 ~1.00); veto groups with test rate <= 0.5 (e.g. India high band,
overlapping number + business word added 2,314 pairs 0.006, swapped 3,307 pairs 0.025).
```
python src/group_rules.py cache/val_v8_scored.parquet output/v8_raw/scored_pairs.parquet cache/rules_v8
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub18 --keep 0.99 --threshold 0.97 --word-veto --neighbour-veto US France --typo-rescue 0.8664 --group-rules cache/rules_v8
```
-6,635 vetoed, +45,917 rescued; 3.19 matches/entity. PASS.

## sub19 (26 Sep evening): group rules over all score bands (recall)

src/group_rules.py now tags conflicting-number pairs with their change kind and covers
bands [0.3,0.6), [0.6,0.8664), [0.8664,0.97), [0.97,1]. Band-shift check: for clean groups
the test pairs fall in the same bands as validation TRUE pairs (US same+identical: 99.98% vs
99.99% in the top band; US blank+identical: 18/11/4/67% vs 21/11/3/65%), so band-specific
rates hold. Band-free test rates: blank address + identical name US 0.89 / India 0.95
(validation 0.75-0.78: test has fewer same-name branches competing for blank records);
neighbour numbers 0.10-0.44; business word added/swapped 0.00-0.11.
```
python src/group_rules.py cache/val_v8_scored.parquet output/v8_raw/scored_pairs.parquet cache/rules_v8b
python src/finalize.py output/v8_raw/scored_pairs.parquet artifacts/v8 output/sub19 --keep 0.99 --threshold 0.97 --word-veto --neighbour-veto US France --typo-rescue 0.8664 --group-rules cache/rules_v8b
```
-6,635 vetoed, +100,750 rescued; 3.22 matches/entity. PASS.
