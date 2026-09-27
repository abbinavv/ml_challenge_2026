# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Dronaut
**Team Members:** Nouman Shafique, Ishan Bag, Abhinav Raj
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve business records with blocking (per-country word TF-IDF plus exact-key passes,
then a learned candidate filter that keeps 5.3 candidates per Source-1 entity), a two-stage
LightGBM pair model on 57 country-agnostic features, and a decision stage calibrated to the
**test** distribution without test labels. The key finding: test contains about twice as
many look-alike decoys as train (neighbouring businesses: a house number a few doors away,
one business word swapped, an extra business word). A model tuned on train validation is
over-confident on them. We measure this shift from pair densities (true variants are
generated the same way in train and test, so any excess of a pair type on test is decoys),
turn it into decision rules and a prior-corrected second-stage model with record-to-record
"cohesion" evidence, and raised the public leaderboard from 0.9356 to **0.9722**.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full data (train: 2.2M Source-1 entities, 10.3M Source-2/3 records; test:
1.73M / 9.97M):

| Finding | Value | Consequence |
|---|---|---|
| Matches per Source-1 entity | mean 3.46, max 11 | Recall on multi-match entities matters |
| Singletons | 5.6% (train and, by our estimate, test) | Worth 1.0 each only if predicted empty |
| Cross-country matches | 0 | Block within country |
| Source-2/3 records matched to >1 entity | 0 | One-owner rule |
| Source-2/3 records per Source-1 entity | train 4.7, test 5.8 | Test has ~2x the decoys |
| Test-only country | France, 15% of test entities | Country-agnostic features; France calibrated separately |
| Same-name Source-1 entities (branches) | 53-61% share a key name with another | Blank-address records are ambiguous between namesakes |
| France co-location | 13% of French entities share an address (train 6%) | More neighbouring-business decoys |
| Native-script names (India) | ~18% of Indian Source-2/3 records, 9 scripts | Learned transliteration dictionary |

Noise on known matches: word shuffles, typos (incl. OCR-style l/i, rn/m), injected accents,
legal-suffix changes, generic filler words replacing a descriptor ('Palma Mining' ->
'PALMA LLC CENTER'), DBA / trade names, initialisms, and addresses with reordered, missing or
abbreviated parts, '<NULL>' tokens and leading-zero / digit typos in house numbers.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate filter + gradient-boosted pair classifier +
label-free test calibration of the decision.

**Core Innovation:** *Density-ratio calibration.* True variants of a business are generated
by the same process in train and test; test only adds decoys. For any pair type g (score band
x house-number relation x name-change kind),

    test true rate(g) = [true pairs per Source-1 entity on validation](g) / [pairs per Source-1 entity on test](g)

Types that contain no decoys come out identical on both sides (India: 2.53 vs 2.53 pairs per
entity above score 0.999), which validates the assumption. The resulting rates drive rescue /
veto rules and the weights of a prior-corrected second-stage model. No test label is used.

---

## 3. Candidate Generation (Blocking)

- **Normalisation:** anyascii transliteration plus a word dictionary learned from ~50K
  native-script training matches (positional and consonant-skeleton alignment), a sound-alike
  map for unseen Indian-script words (built from Source-1 names only), legal-form removal,
  compact and key names, address abbreviation canonicalisation, state / region codes (incl.
  French regions and departements), splitting of glued house numbers ('No.301' -> 301).
- **Blocking keys used:** per-country word TF-IDF over name + address (top-30 neighbours,
  common-word cut-off max_df 0.02; France without the cut-off, since it removed city names),
  plus exact-key passes: same address key, same compact name, same name for records with an
  empty address (keys shared by at most 20 records).
- **Candidate filter:** a small LightGBM on cheap features keeps 99% of the true matches that
  blocking found (cut-off measured on held-out entities).
- **Candidate pairs generated:** 9,186,608 in `candidate_pairs.tsv` = **5.30 per Source-1
  entity** (blocking before the filter: ~31 per entity).
- **How you ensured true matches were not lost:** blocking recall measured on held-out train
  entities: 98.0% of true pairs reach the blocking candidates, 97.5% survive the filter. The
  remaining misses were analysed (mostly generic names with truncated addresses, or trade
  names with no shared word); extra key-based passes recovered too few of them (< 0.2 points)
  to justify the larger candidate set.

---

## 4. Matching Model

**Features used (57, country-agnostic):**
- Name features: token-set / partial / Jaro-Winkler ratios on normalised, core and compact
  names; consonant-skeleton similarity; key-name equality; words added / missing; legal-form
  family conflict; initialism detection.
- Address features: token-set ratio, house-number Jaccard / conflict / count of shared
  numbers, address-key similarity, empty-address flags.
- Other: blocking cosine and rank, gap to the best candidate, how many other Source-1 entities
  list the candidate (competition), key-name rarity, cluster support (how many of the entity's
  candidates agree), which blocking pass found the pair.

**Model type:** LightGBM, two stages. Stage 1 (cheap features) = candidate filter; stage 2
(all features) = main model, 127 leaves, 3,573 boosting rounds with early stopping, trained on
2.0M train entities (entity-level split, 150K held out) on AWS EC2; negatives whose house
numbers conflict weighted x3.

**Decision stage (test-calibrated):**
1. One-owner rule (each Source-2/3 record goes to its highest-scoring Source-1 entity).
2. Base threshold 0.97 on the stage-2 probability. The public leaderboard rose each time the
   threshold was raised (3.64 -> 3.48 -> 3.34 matches per entity: 0.921 -> 0.937 -> 0.948),
   and the density calibration showed why: pairs scored 0.87-0.97 are 91-97% true on
   validation but only ~40-66% true on test.
3. House-number change types: a digit added or lost, or one digit changed with a big value
   jump, stays ~95-100% true on test (typo) and is rescued down to 0.8664; one digit changed
   within 9, a value within 20, or transposed digits fall to 10-50% true on test (neighbouring
   business) and are rejected in the US and France.
4. Group rules (`src/group_rules.py`): per (country, score band, house-number relation incl.
   overlap subset / mixed, name-change kind) the test true rate above; groups >= 0.82 are
   rescued, groups <= 0.74 rejected (around the F0.5 break-even F*/1.25 ~ 0.78).
5. Word-swap veto: a candidate that swaps one of the Source-1 name's words for another
   frequent business word ('Fontaine Club' vs 'Fontaine Amicale') is rejected (0.02% of true
   train pairs flagged vs 3.3% of French test matches).
6. Prior-corrected second stage for US and India (`src/prior_stack.py`): a LightGBM trained on
   898K train entities whose first-stage scores are out-of-sample (cross-fitting on EC2: two
   half-models, each scoring the other half), with labels, fine pair features, change kinds,
   rival features (how the candidate fits competing Source-1 entities), list context, and
   record-to-record cohesion (how its name / address / house number agree with the entity's
   other confident vs rejected candidates). Negatives are re-weighted per group to test
   levels (test negatives per entity = test pairs per entity - validation true pairs per
   entity), so it learns P_test(true | pair). It scores pairs down to first-stage 0.02 (1.5%
   of true matches scored below 0.3); 127 leaves x 800 rounds. The cut-off was set from the
   leaderboard itself: lowering it to 0.39 (sub35) cost 0.0097, showing the model's mid-range
   probabilities are over-confident on test (pairs rated 0.52 were ~18.5% true; true rate ~
   prob^2.55), so the final cut-off keeps pairs whose calibrated rate clears the F0.5
   break-even. France (no labels) keeps steps 1-5, borrowing only US rescue rules for kinds
   that cannot be another business at the same address; French rescued matches whose
   calibrated rate (same model encoded as US) is below break-even are dropped.
7. Sibling expansion: an unclaimed record with the same country, key name and address key as
   a matched record joins that match (99.9% same business on train truth).

**Threshold selection method:** base threshold from the leaderboard trend and the density
calibration; rule bars from the F0.5 break-even; the second-stage cut-off from macro F0.5 on
held-out entities re-weighted to test levels.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** 0.9749 for the v8 model on 150K held-out train entities
  (precision 0.9945, recall 0.9442; India 0.9705, US 0.9779). Under test-level negative
  weights (out-of-fold, pairs scored >= 0.3), the second stage reaches precision 0.9975, recall
  0.9914 at P >= 0.78 (log-loss 0.1069 -> 0.0215), vs 0.9929 / 0.9648 for the rule-only decision.
  Ceiling: only 95.8% of true matches reach the scored pairs (blocking ~2%, candidate filter
  0.5%, one-owner 0.2%, first-stage < 0.3 ~1.5%); scoring down to 0.02 recovers part of the last.
- **Public leaderboard progression:**

| File | Change | Public F0.5 |
|---|---|---|
| sub04 | v5 model, 450K entities, threshold 0.70 | 0.935552 |
| sub06 | cleaning v3, full re-blocking, threshold 0.725 | 0.936795 |
| exp06 | same scores, threshold 0.45 / 0.90 | 0.920969 / 0.948075 |
| sub08 | v8 (2.0M entities, EC2) + word-swap veto | 0.956511 |
| sub22 | test-calibrated decision rules (steps 2-4) | 0.963841 |
| sub26 | + prior-corrected second stage (150K entities) | 0.967148 |
| **sub33** | **+ cross-fitted 898K entities, rival / context / cohesion features, floor 0.02** | **0.9681** |
| sub34 | deterministic re-run of sub33 | 0.968043 |
| sub35 | cut-off 0.39 (probe) | 0.958349 |
| sub37 | cut-off 0.85 + French calibrated veto | 0.970977 |
| sub38 | larger second stage (127 leaves x 800 rounds), count-matched cut-off | 0.971088 |
| **sub40 (final)** | **cut-off 0.92 (curve refit on sub37: true ~ prob^3.19), French low-confidence veto** | **0.972202** |

- **Common false positives (wrong merges):** neighbouring businesses built to look alike: the
  same name a few house numbers away (221 vs 225, 19 vs 22), one business word swapped or
  added at the same address ('Emmanuel Fetes SA' vs 'Emmanuel Amis SA', 'Shiv & Sons' vs
  'Shiv & Sons Industries'), legal-form changes in France ('Maison Event SARL' vs 'SNC'), and
  addresses sharing some numbers but not others (3/289 vs 3/290).
- **Common false negatives (missed matches):** records with a blank address whose name is
  shared by several Source-1 branches (only 25-43% of such candidates are the entity's own on
  validation); genuine variants whose house number contains a far-off typo (indistinguishable
  from a relocated business); and ~2% of true matches that blocking never proposes (generic
  names with truncated addresses, trade names with no shared word).

---

## 6. Conclusion

A standard blocking + gradient-boosting pipeline reaches 0.975 on train validation but loses
precision on test, where look-alike decoys are twice as frequent. Measuring that shift from
pair densities, without labels, and correcting the decision for it (type-specific rules and a
prior-corrected second stage with cohesion evidence) was worth +0.037 on the public leaderboard
(0.9356 -> 0.9722)
while keeping 5.3 candidates per entity. The main lesson: when test differs from train,
validate the decision rule against the target distribution, not only the model.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains all source (`src/`), `README.md` (exact commands,
data -> blocking -> matching -> output) and `requirements.txt` (pinned). Entry points in order:
`build_cache.py`, `learn_translit.py`, `run_blocking.py`, `reblock_country.py`,
`add_key_candidates.py`, `train_model.py`, `predict.py`, `score_validation.py`,
`group_rules.py`, `crossfit_prep.py`, `prior_stack.py`, `finalize.py`. The second stage and
the final step are deterministic (identical output on repeated runs). Note: the submitted second stage was trained with the
street-agreement split enabled on the validation side only (`prior_stack.py --street`); the
README reproduces that configuration as submitted. The packaged output is sub40 (README steps
7-8c).

### B. Additional Results

Test-calibrated true rates (examples; validation -> test):

| Pair type (US unless noted) | Validation | Test |
|---|---|---|
| Score 0.87-0.93, all types | 91-93% | 41-55% |
| Conflicting house number, one digit changed within 9 | 85% | 18% |
| Conflicting house number, value within 20 | 80% | 11% |
| Conflicting house number, digit added or lost | 98% | 94-96% |
| India, overlapping numbers + business word added (score >= 0.97) | n/a (1 pair) | 0.6% |
| Blank address, identical name (band-free) | 75-78% | 89-95% |

Tried and rejected (measured): number-cluster rule (flags 65% of true conflicting pairs),
dropping all conflicting numbers (-0.016 on validation), street-agreement split (no
separation), rival-feature stacker (gain only at loose cut-offs), label-free domain classifier
used alone (worse than group rules on the group estimate).
