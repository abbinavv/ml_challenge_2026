# Masterplan — Amazon ML Challenge 2026: Business Entity Resolution

Companion to [context.md](context.md) (verified facts). This file says **what we must deliver,
how we will build it, where it runs, and when**. Section 5 (pipeline) is a **proposal awaiting
approval** — nothing is implemented yet.

**Deadline: 27 Sep 2026, 23:59 IST.** Leaderboard uploads: 5/day, 15 total.

---

## 1. Goal and success criteria

| # | Criterion | Target |
|---|---|---|
| G1 | Valid, SCORED leaderboard submission | First one **tonight (25 Sep)** |
| G2 | Macro F0.5 on our own validation split | As high as possible; report in write-up |
| G3 | Blocking recall ceiling (share of true pairs in candidates) | >= 97% with a bounded candidate count |
| G4 | Works for France (unseen country) | No country-specific model parts; verified with a leave-one-country-out test |
| G5 | Final zip reproduces both output files from raw data | One command, pinned environment |
| G6 | Rank goal | Top 50 (interviews) |

---

## 2. Requirements checklist

### 2.1 Hard rules — rejection or disqualification if broken

- [ ] **R1** `matching_results.tsv` is tab-separated with the exact header `source1_entity_id<TAB>matched_entity_ids`
- [ ] **R2** Exactly one row for **each of the 1,732,544** test S1 entities — no missing rows, no duplicate rows
- [ ] **R3** Only `S2-`/`S3-` IDs that exist in the test files; no `S1-` IDs; no duplicates within a list
- [ ] **R4** Rows with no matches still contain the tab (`S1-xxx\t`)
- [ ] **R5** IDs joined with `,` and **no spaces**; no quoting
- [ ] **R6** UTF-8, LF line endings (`\n`, never `\r\n`)
- [ ] **R7** Country is treated as an open set: no hard-coding, filtering or one-hot encoding to {US, India}; France rows are fully processed
- [ ] **R8** **No external data**: no APIs, geocoding, business registries, web lookups or internet enrichment. Only the provided files, plus general-purpose libraries and hand-written rules
- [ ] **R9** Any pretrained model: **MIT or Apache-2.0 licence and <= 8B parameters** (applied to every pretrained model used, not just the last one)
- [ ] **R10** `validate_submission.py --check-ids` prints PASS before **every** upload

### 2.2 Deliverables

- [ ] **D1** `output/matching_results.tsv` — final matches (same file as the leaderboard upload)
- [ ] **D2** `output/candidate_pairs.tsv` — the exact candidate set the model scores (the last filtering stage); every match must be in it
- [ ] **D3** `code/business_entity_resolution/src/` — all source code, with **comments describing each function**
- [ ] **D4** `code/business_entity_resolution/README.md` — exact end-to-end run instructions (data -> blocking -> matching -> output)
- [ ] **D5** `code/business_entity_resolution/requirements.txt` — pinned versions (from `requirements.lock.txt`)
- [ ] **D6** Filled-in `Documentation_template.md` — approach, blocking, features, model, threshold method, validation F0.5, error analysis. ~2-page main body, detail in the appendix
- [ ] **D7** Zip named `<team_name>_submission.zip` with the structure above
- [ ] **D8** Log of every leaderboard submission (file, date, validation score, public score, notes) — `submissions/LOG.md`

### 2.3 Requirements that come from the data

- [ ] **Q1** Scale: ~10M S2/S3 records per split -> memory-aware, chunked processing; cache parsed data as Parquet
- [ ] **Q2** Parse raw lines with an explicit tab split / `quoting=QUOTE_NONE`, the **same way everywhere** (483 test rows have CSV-style quotes)
- [ ] **Q3** Block within country (0 cross-country matches in train)
- [ ] **Q4** Use the one-owner rule: each S2/S3 record goes to at most one S1
- [ ] **Q5** Handle Indian-script names (~12% of India S2/S3): transliterate to Latin (`anyascii`) before comparing
- [ ] **Q6** Handle website-style names (`baycenter.com`, `0nelogistics.com`): strip www/.com, map 0->o, compare with the name's spaces removed
- [ ] **Q7** Normalise legal suffixes (US, India **and French**: SARL, SAS, SASU, EURL, EI, SA), address abbreviations, and number formats (`B-00200` -> `b200`)
- [ ] **Q8** Robust to shuffled words, typos, injected accents, junk prefixes/suffixes, and addresses with parts reordered, altered or missing (~3% empty)
- [ ] **Q9** Precision first: F0.5 plus 26% distractor records -> the decision threshold is tuned for macro F0.5, not accuracy or F1
- [ ] **Q10** Singletons are only 5.6%, but each one is worth a full 1.0 -> never force a match when nothing is convincing

---

## 3. Compute recommendation: **Mac + AWS (hybrid)**, not AWS-only

**Recommendation:** develop and iterate on this Mac; run the **full-data jobs** (all 2.2M train /
1.7M test entities) on **one large-memory AWS EC2 instance**. Keep the Mac as the primary
workspace and AWS as the heavy-lifting machine.

| Option | Verdict | Why |
|---|---|---|
| **Mac only** | Possible but risky | 16 GB RAM. Full-data blocking and feature building would need careful chunking everywhere; one memory spike costs hours we don't have |
| **AWS only** | Not recommended | Every edit-run cycle goes through a remote machine; setup and data upload cost time; idle hours burn credits. The Mac's M5 is fast for development |
| **Mac + AWS** | **Recommended** | Fast iteration locally on country-stratified samples (5-10%); full runs on 128 GB RAM where memory is never the problem. Same Python 3.12 + lock file on both, so code moves unchanged |

**Division of work**

| Mac (16 GB, M5) | AWS EC2 (e.g. r7i.4xlarge: 16 vCPU / 128 GB) |
|---|---|
| Exploring the data, writing normalisation rules | Full train blocking + feature building |
| Blocking and feature development on samples | Model training on the full pair set |
| Model experiments on samples | Full test inference -> submission files |
| France rule development | Final reproducible run for the zip |
| Validator runs, packaging, documentation | |

**AWS specifics**
- Use **plain EC2**, not SageMaker notebooks. The SageMaker free tier is `ml.t3.medium` (4 GB) —
  useless here — and SageMaker instances cost more than the same EC2 size.
- `r7i.4xlarge` (128 GB) costs roughly **$1/hour** on-demand in us-east-1 (check the console for the
  exact price). ~15-25 hours of use = **~$15-50**, well inside the $200 credits.
- **Check today, before we need it:**
  1. **vCPU quota** — new accounts often have a low limit for on-demand instances (Service
     Quotas -> "Running On-Demand Standard instances"). Raising it can take hours.
  2. **Free vs Paid plan** — "Free plan" accounts restrict some instance types. Upgrading to the
     Paid plan keeps the credits.
  3. Same region (us-east-1) for every team member; set a billing alert.
- **Stop the instance whenever it is idle.** Move data with `aws s3 cp` (2.4 GB, one time).

---

## 4. How we validate (so we can trust scores without spending uploads)

1. **Holdout by S1 entity:** split the train S1 entities into folds. Blocking always searches
   **all** train S2/S3 records, including the 26% that match nothing, so the setup mirrors test.
2. **Cross-fitting (final runs):** train on fold A and predict fold B, and vice versa, then apply the
   one-owner rule across **all** S1 entities at once — exactly how the test run behaves.
3. **France stand-in:** leave-one-country-out — train on US only and score on India (and the reverse).
   The drop in score tells us how well features transfer to an unseen country.
4. **Metrics tracked every run:** blocking recall ceiling, candidates per S1, macro F0.5 (overall
   / per country / singletons only), precision, recall.
5. The scorer is our own code implementing the official formula exactly (singleton rule included),
   tested against the worked example (predicted 3, true 2 -> 0.714).

---

## 5. Proposed pipeline — **needs your approval**

```
raw TSV --> [0 Load + normalise] --> [1 Blocking] --> candidate_pairs.tsv
                                          |
                                          v
                               [2 Pair features] --> [3 LightGBM] --> [4 Decide + one-owner] --> matching_results.tsv
```

**Stage 0 — Load and normalise** (cached to Parquet)
- Raw tab-split parsing. For each record build: a normalised name (accents removed, lower case,
  Indian scripts transliterated, junk removed); a **core name** (legal suffixes and generic words
  removed); a **compact name** (no spaces or punctuation, `.com`/`www` removed, 0->o); address
  tokens with abbreviations expanded; address numbers with leading zeros removed; street words;
  city/state words; postcode if present.

**Stage 1 — Blocking** (within each country; several passes, merged, capped at top-K per S1)
- a) Character 3-gram TF-IDF on the core name -> nearest neighbours by cosine (`sparse_dot_topn`)
- b) TF-IDF on address numbers + street words -> nearest neighbours
- c) Exact rare keys: (house number + street word), compact name
- d) Optional: multilingual sentence embeddings for names in Indian scripts
  (`paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0) searched with FAISS
- Tuned on validation: **recall >= 97% with ~30-50 candidates per S1**. The output is written as
  `candidate_pairs.tsv`.

**Stage 2 — Pair features** (country-agnostic, so France works)
- Name: several `rapidfuzz` scores (token-set, partial, Jaro-Winkler), word overlap, compact-name
  match, match against a website-style name, script flags
- Address: house number exact / conflicting / missing, street-word overlap, city and state overlap,
  address missing
- Context: the blocking similarity scores, the candidate's **rank** within its S1's list, gap to the
  best candidate, source (S2 or S3)
- **No country one-hot** (R7); country only affects which normalisation rules apply

**Stage 3 — Model**
- LightGBM binary classifier (MIT licence) on the labelled candidate pairs. Class balance and
  features checked per country.

**Stage 4 — Decision**
- **One-owner rule:** each S2/S3 record is kept only for the S1 entity that scores it highest.
- **Per-entity selection:** sort an entity's candidates by probability and keep the set that
  maximises *expected* F0.5 (or a single threshold tuned on validation — both compared).
- An empty list when nothing clears the bar (protects singletons).

**Stage 5 — Output**
- Write both TSVs to the exact format (R1-R6) -> run the validator with `--check-ids` -> log the submission.

**Alternatives considered**
- *Rules only (no ML):* fast to build but can't weigh dozens of noisy signals; kept only as the
  tonight baseline.
- *Fine-tuned transformer pair model (e.g. a small cross-encoder):* possibly more accurate on hard
  pairs, but slow on ~50M test pairs within 2 days; possible later refinement on the uncertain pairs
  only.

---

## 6. Design review findings (Fri 25 Sep, measured on the real data)

Each finding below was measured, and each one changes the plan.

| # | Finding (measured) | Change to the plan |
|---|---|---|
| F1 | Test has **~24% more S2/S3 records per S1** than train in every country (5.5-5.8 vs 4.7). If true matches stay at 3.46 per entity, decoys nearly **double** (1.2 -> ~2.3 per S1) | Validation **simulates test density**: drop ~19% of train S1 entities but keep their S2/S3 records, so they become extra decoys. Thresholds are tuned in that harder setting. Two uploads are reserved to calibrate the threshold against the public leaderboard |
| F2 | **22% of decoys share an exact core name with some S1** (true matches: 56%) — decoys are look-alikes | Precision depends on address evidence plus **cluster-support features** (how similar a candidate is to the entity's other strong candidates) |
| F3 | **Name-only blocking fails** (top-50 recall: US 72%, India 59%); **name+address combined top-10 = US 96.8%** | Core blocking pass = combined name+address char 3-gram TF-IDF. Name-only and address-only passes are secondary, for the union only |
| F4 | **India recall stops at ~93%** (union of 3 passes, top-20 each); India is 47% of test | Priority fix: learn a word dictionary from ~50K labelled Indian-script name pairs in train, add a phonetic fallback, map state names written in Indian scripts. Target >= 96% |
| F5 | Indian-script names: median similarity only **58** after `anyascii` transliteration | Same fix as F4; also give those records features that lean on the address |
| F6 | Only **1.2%** of true pairs have a weak name AND no shared address number | A recall ceiling of ~99% is achievable; the rest is blocking and model quality |
| F7 | Blocking cost: **~12 min per country for 10K queries**, mostly building the index | Time the full test run in Step 2 before committing; AWS or pass-pruning if it's too slow |

---

## 6b. What is actually built (updated Fri 25 Sep evening)

| Stage | Implementation | Measured |
|---|---|---|
| Normalise | `src/ber/normalize.py`: transliteration, legal-form removal, compact names, address abbreviations, US/India state codes (incl. native-script spellings), **learned Indian-script word dictionary** (`src/learn_translit.py`, 517 words from 551K train pairs) | Indian-script name similarity: median 50 -> **100**; dictionary covers 72% of test native-script words (train: 73%) |
| Block | `src/ber/blocking.py`: **word-token** TF-IDF on core name + address per country, drop words in >2% of pool, top-K | Char 3-grams were 12 h for test; words ~35 min at the same recall |
| Features | `src/ber/features.py`: 31 features — blocking, competition (needs full query set), name, address, record, **cluster support** | `n_same_nums` enters the top 10 |
| Model | `src/train_model.py`: LightGBM, entity-level split, per-country report | v1: 0.9402; v2 smoke: 0.9446 |
| Decide | `src/ber/decide.py`: one-owner rule + global threshold or per-entity expected-F0.5 (chosen on validation) | threshold won on the smoke run |
| Output | `src/predict.py` + `ber/io.write_id_lists` | validator PASS with `--check-ids` |

## 7. Execution steps (IST, starting Fri 25 Sep ~17:00)

**Deadline: Sun 27 Sep 23:59 IST.** Uploads: 5 per day (they don't carry over) — use today's.

### Friday 25 Sep — get a valid, scored entry tonight

| Step | What | Done when |
|---|---|---|
| ✅ **1. Data layer** (~1 h) | `src/io.py` + `src/normalize.py`: raw tab-split parser, per-record fields (normalised / core / compact name; normalised address; number set), cached to Parquet for all 7 files | All 7 files cached; row counts match `context.md` |
| ✅ **2. Blocking v1** (~1.5 h) | `src/blocking.py`: per country, combined name+address TF-IDF top-K (K=20) in chunks; write `candidate_pairs.tsv`. **Time the full test run** (F7) | Test candidates written; recall on a train sample logged; runtime known |
| ✅ **3. Scorer + baseline** (~1.5 h) | `src/metrics.py` (official macro F0.5, singleton rule, checked against the 0.714 example); `src/baseline.py`: weighted name/address similarity + one-owner rule, threshold tuned on a train sample | Validation F0.5 logged |
| ✅ **4. Upload #1-#2 files** (~22:00) | `src/write_output.py` -> validator `--check-ids` -> upload -> `submissions/LOG.md` + git tag `sub-01` | **SCORED on the leaderboard** |

### Saturday 26 Sep — the real model

| Step | What | Done when |
|---|---|---|
| **5. Validation harness** | Entity folds + the F1 density simulation + recall / F0.5 per country and singletons | One command gives a full report |
| **6. India recall** | Learned Indian-script word dictionary + phonetic fallback + state-name map + the secondary name and address passes (F4) | India recall >= 96% |
| **7. Features + LightGBM** | Stage 2 features + cluster-support features (F2) -> LightGBM -> expected-F0.5 selection per entity + one-owner rule | Beats the baseline on validation |
| **8. France** | French legal-form and street-abbreviation lists; leave-one-country-out test | Drop measured; no country one-hot |
| **9. Full runs + uploads #2-#5** | Train on all train data, predict test. Mac in chunks; AWS only if Step 2 timing needs it (quota request takes hours — decide by Saturday morning) | Best validation model uploaded before 27 Sep 00:00 (48-hour top-500 credits) |

### Sunday 27 Sep — tune, package, submit

| Step | What | Done when |
|---|---|---|
| **10. Calibrate + error analysis** | Uploads #6-#7: same model, two thresholds (F1 check); review false merges and misses | Threshold fixed |
| **11. Final run by 18:00** | Clean end-to-end run from raw data -> both TSVs -> validator | PASS |
| **12. Package by 21:00** | `code/business_entity_resolution/{src,README.md,requirements.txt}`, filled `Documentation_template.md` (~2-page body), `output/`, `<team_name>_submission.zip`; reproduce once from the zip | Zip reproduces the outputs |
| **13. Final upload + zip submission** | Final `matching_results.tsv` upload; submit the zip; 3-hour buffer before 23:59 | Both submitted |

**Upload budget (15 total):** Fri 1-2 · Sat 4-5 · Sun 3-4, keeping 2 for the last evening.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Out of memory on full data | Process each country separately, in chunks; Parquet cache; heavy runs on AWS |
| Blocking recall too low | Several complementary blocking passes; measure recall after every change |
| France behaves differently | Country-agnostic features, a French suffix/abbreviation list, leave-one-country-out test |
| AWS quota or plan blocks the instance | Quota is 5 vCPUs (checked); Mac in chunks is the plan; decide on an increase by Sat morning |
| Test has more decoys than train (F1) | Density-simulated validation; two uploads to calibrate the threshold |
| India recall below target (F4) | Learned dictionary + phonetic fallback + extra blocking passes |
| Wasted uploads on format errors | Validator with `--check-ids` before every upload |
| Tuning to the public leaderboard | Decide using our own validation; public score is a sanity check only |
| Running out of time | A valid baseline goes in tonight; every later step improves it rather than replacing it |

---

## 8. Open decisions (need answers from the team)

1. **Approve Section 5 as revised by Section 6, and the steps in Section 7.**
2. **Team:** how many members, and who owns what (AWS setup, blocking, features/model,
   write-up/packaging)?
3. ~~AWS account~~ — done: Free plan, us-east-1, $100 credits so far, vCPU quota 5 (increase
   deferred; revisit after the Step 2 timing).
4. **Team name** (needed for the zip filename).
