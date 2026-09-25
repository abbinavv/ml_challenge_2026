# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME — to fill]
**Team Members:** [to fill]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve business records with a four-stage pipeline: data cleaning driven by an audit of
known true matches, per-country word-level TF-IDF blocking on name + address, a LightGBM
pair classifier on 37 country-agnostic similarity features, and a decision stage that
applies a one-owner rule and a threshold tuned directly for macro F0.5. Key innovations are
a word dictionary for Indian-script names learned from the training pairs (name similarity
of native-script true pairs: median 50 -> 100), cleaning rules chosen by measuring what still
differs between known matches, and "cluster" and "competition" features that exploit the
structure of the data (each Source-2/3 record belongs to at most one Source-1 entity).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full training data (2.2M Source-1 entities, 10.3M Source-2/3 records):

| Finding | Value | Consequence |
|---|---|---|
| Matches per Source-1 entity | mean 3.46, max 11 | Most of the score is recall on multi-match entities |
| Singletons (no match) | 5.6% | Worth 1.0 each only if predicted empty |
| Cross-country matches | 0 | Block within country |
| Source-2/3 records matched to >1 entity | 0 | One-owner rule |
| Source-2/3 records matching nothing (decoys) | 26% | Source of false merges |
| Decoys sharing an exact core name with some entity | 22% | Decoys are look-alikes: address evidence is essential |
| Test records per Source-1 entity vs train | 5.5-5.8 vs 4.7 | Test likely has more decoys: precision matters even more |
| Test-only country | France, 15% of test | Features must be country-agnostic |
| Native-script names (India) | ~12% of India records, 9 scripts | Transliteration alone gives median similarity 50 |

Noise observed on known matches: word shuffles, typos, injected accents, legal-suffix swaps,
website-style names (`0nelogistics.com`), "X trading as Y" names, honorifics (Mr, Smt, Sri),
OCR-style swaps (`5ervices`, `lnc`, `INIMITA8LE`), duplicated words; addresses with reordered
parts, abbreviations, missing parts, native-script state names, renamed cities
(Bombay/Mumbai), and house-number typos (`731` vs `31`, `14637` vs `14638`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (with rule-based decision layer)
**Core Innovation:** Audit-driven cleaning + learned Indian-script dictionary + structure-aware
features (one-owner competition and within-entity cluster agreement).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** for each country separately, word-token TF-IDF over
  "core name + normalised address"; each Source-1 record takes its top-30 most similar
  Source-2/3 records by cosine similarity (sparse matrix product, chunked). Words present in
  more than 2% of a country's pool are dropped from the index for speed.
- **Why this design (measured on labelled queries):**

  | Blocking variant | Recall@20 (US) | Speed |
  |---|---|---|
  | Name-only char 3-gram | 65.5% | slow |
  | Name + address char 3-gram | 98.0% | ~12 h for test |
  | **Name + address word tokens** | **97.8%** | ~35 min for test |

  Name-only blocking fails because many decoys share a business name; adding the address
  is essential.
- **Candidate pairs generated:** [to fill: test pairs, ~30 per Source-1 entity]
- **How we ensured true matches were not lost:** recall measured on held-out training
  entities after every change; combined name+address text; Indian-script dictionary so
  native-script names meet their Latin form; top-30 instead of top-20. Measured blocking
  recall: [to fill].

---

## 4. Matching Model

**Features used (37, none country-specific):**
- **Name:** token-set / token-sort / plain ratio, partial ratio, Jaro-Winkler on compact
  form, compact equality, word Jaccard, length difference, key-name (generic words removed)
  similarity and equality, alternate-name ("trading as") similarity, acronym match
- **Address:** token-set and plain ratio on normalised address, address-key similarity,
  house-number Jaccard, number conflict, near-miss numbers (digit dropped / off by <= 2),
  number counts, missing address
- **Blocking / competition:** cosine score, rank, gap and ratio to the entity's best
  candidate, candidate count; how many entities list the candidate and whether this entity
  is its strongest claimant (soft one-owner rule)
- **Cluster support:** similarity to the entity's strongest candidate, number of other
  candidates sharing the same compact name or the same house numbers
- **Record:** source (S2/S3), native-script name

**Model type:** LightGBM binary classifier (MIT licence), trained on candidate pairs of
~300K entities, early stopping on a disjoint validation sample; no pretrained models.

**Threshold selection method:** one-owner rule (each Source-2/3 record kept only for the
entity that scores it highest), then the decision rule with the best macro F0.5 on
validation among (a) a global probability threshold and (b) per-entity expected-F0.5
selection. Entities with nothing above the bar get an empty list (singleton-safe).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [to fill: final validation score, per country]

| Version | Change | Validation macro F0.5 |
|---|---|---|
| v1 | baseline: word blocking top-20 + 22 features | 0.9402 |
| v2 (smoke) | + Indian-script dictionary, cluster features | 0.9446 |
| v3 (smoke) | + audit-driven cleaning, key/alt/acronym/near-number features | 0.9483 |
| v3 full | + full-set competition features, top-30 blocking | [to fill] |

- **Common false positives (wrong merges):** [to fill after error analysis] — expected:
  look-alike decoys at the same address with a similar name.
- **Common false negatives (missed matches):** [to fill] — expected: blocking misses for
  India, heavily truncated addresses, names reduced to initials.

---

## 6. Conclusion

[to fill — 2-3 sentences]

---

## Appendix

### A. Code Artefacts

Code in `code/business_entity_resolution/`; see its `README.md` for the exact end-to-end
commands (data -> blocking -> matching -> output).

| File | Role |
|---|---|
| `src/ber/normalize.py` | Cleaning rules (names, addresses, Indian-script dictionary lookup) |
| `src/learn_translit.py` | Learns the Indian-script word dictionary from training pairs |
| `src/ber/blocking.py`, `src/run_blocking.py` | Per-country TF-IDF blocking |
| `src/ber/features.py` | Pair features |
| `src/train_model.py` | Training, validation, decision-rule selection |
| `src/ber/decide.py` | One-owner rule, threshold, expected-F0.5 selection |
| `src/predict.py` | Test scoring and submission files |
| `src/ber/metrics.py` | Official macro F0.5 |

### B. Additional Results

Cleaning audit on 71,878 known true pairs:

| | US before -> after | India before -> after |
|---|---|---|
| Identical core names | 54.4% -> 62.2% | 60.7% -> 70.5% |
| Identical key names | — -> 66.2% | — -> 76.9% |
| Name similarity, worst 10% | 78 -> 88 | 74 -> 92 |
