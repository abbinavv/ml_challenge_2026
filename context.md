# Context — Amazon ML Challenge 2026: Business Entity Resolution

Read this first. It is the verified factual background for the project; the plan lives in
[masterplan.md](masterplan.md). Everything below was checked against the actual files on
2026-09-25, not taken from memory.

---

## 1. The task in one paragraph

Three data sources describe real-world businesses with only a **name**, an **address** and a
**country** — no shared ID, no phone, no email. Source 1 (S1) is a clean, deduplicated reference
list. Sources 2 and 3 (S2, S3) are noisy vendor feeds. For **every S1 test entity**, output the
list of S2/S3 records that are the same business (zero, one or many). Scored by **macro F0.5**:
F0.5 is computed per S1 entity and averaged, so every entity counts equally and precision is
weighted 2x over recall.

```
F0.5 = (1.25 x P x R) / (0.25 x P + R)
Singleton (no true matches): empty prediction = 1.0, any prediction = 0.0
```

---

## 2. Where everything is

| What | Path |
|---|---|
| Dataset bundle (official) | `student_resource/` — download from the Unstop problem page; **never committed** (git-ignored) |
| Train files | `student_resource/dataset/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv` |
| Test files | `student_resource/dataset/test/test_source{1,2,3}.tsv` |
| Official validator | `student_resource/utils/validate_submission.py` (stdlib only) |
| Write-up template | `student_resource/Documentation_template.md` |
| Problem statement (PDF + text) | `docs/problem_statement.pdf`, `docs/problem_statement.txt` |
| Guidelines (PDF + text) | `docs/guidelines.pdf`, `docs/guidelines.txt` |
| Video walkthrough (key slides) | `docs/video_frames/*.jpg` (source: `~/Downloads/6ab509c5b7036_ml_challenge_2026_video.mp4`) |
| Python environment | `.venv/` (Python 3.12, created with `uv`) |

Other sources reviewed: the Unstop problem page (same text as the PDF, plus the dataset
download link), the AWS Builder Center prep blog (logistics + a generic XGBoost demo), and the
AWS Builder Center / Free Tier FAQ (credits and cross-account sharing).

---

## 3. Competition logistics

| Item | Value |
|---|---|
| Window | 25 Sep 2026 00:00 IST -> **27 Sep 2026 23:59 IST** |
| Leaderboard uploads | **5 per day**, max 15 total; only `matching_results.tsv` is scored |
| Leaderboards | Public = subset of test (live); Private = rest (revealed after) |
| Team | 2-4 members |
| Results | Top 50 announced 2 Oct; finale (Top 10) 7 Oct |
| Prizes | INR 1L / 75K / 50K; **interviews (PPIs) for Top 50** |
| Credits | $200 AWS per member (poolable via S3); +$100 for top 500 at the 48-hour mark |
| Queries | Google Form (linked in guidelines); support@unstop.com |

---

## 4. Verified dataset facts

### Sizes

| File | Rows | Countries |
|---|---|---|
| train_source1 | 2,206,821 | US 60% / India 40% |
| train_source2 | 5,034,616 | US 60% / India 40% |
| train_source3 | 5,285,603 | US 60% / India 40% |
| train_ground_truth | 2,206,821 | one row per train S1 |
| test_source1 | **1,732,544** | India 46.8% / US 38.3% / **France 15.0%** |
| test_source2 | 4,887,273 | India 47.3% / US 38.3% / France 14.4% |
| test_source3 | 5,082,316 | India 47.3% / US 38.3% / France 14.4% |

Total ~2.4 GB. All files: UTF-8, LF endings, no BOM, correct column counts, unique IDs,
correct prefixes. Train and test IDs never overlap. Ground truth covers every train S1 exactly
once and every matched ID exists in the correct source file.

### Match structure (train ground truth)

| Fact | Value | Why it matters |
|---|---|---|
| Singletons (0 matches) | **5.58%** | Worth ~5.6% of the score — real but not the main lever |
| Mean matches per S1 | **3.46** (max 11) | Most of the score is recall on multi-match entities |
| Distribution | 1:5.4% 2:17.0% 3:24.1% 4:21.9% 5:14.6% 6:7.5% 7+:4% | |
| Matches split | S2 3.69M / S3 3.94M | Both sources matter equally |
| Cross-country matches | **0** | Country is a free, perfect blocking partition |
| S2/S3 record matched to >1 S1 | **0** | Each S2/S3 record belongs to at most one S1 (assignment constraint) |
| S2/S3 records matching nothing | **26%** (2.68M) | Distractors — the source of false merges |

### Noise observed in real matched clusters

**Names**
- Word shuffles: `Peak Energy Concepts Inc` -> `Peak Energy Inc Concepts`, `Limited One Logistics Private`
- Typos and injected accents: `Cnocepts`, `Peak Énergy`, `Léarning`
- Legal-suffix swaps: Inc / Incorporated / L.L.C. / Llc / Co / Pvt / Private Ltd / Limited
- Website-style names: `baycenter.com`, `0nelogistics.com` (digit 0 for letter o), `almanagementprivate.com`
- Junk and extra text: leading `--`, `<<`; trailing `(ID: 33423)`; `Bay Bay Center`; `Bay Inc Service`
- **Indian scripts:** ~12% of India S2/S3 names are written in Devanagari, Telugu, Kannada,
  Tamil, Bengali, Gujarati, Malayalam, Oriya or Gurmukhi (e.g. `वन लॉजिस्टिक्स प्राइवेट लिमिटेड`
  = One Logistics Private Limited). S1 names are Latin.

**Addresses**
- Parts in a different order: `MN, Moose Lake, 51 Hillside Terrace` vs `#51 HILLSIDE TER, MOOSE LAKE, MN`
- Abbreviations: St/Street, Ter/Terrace, Ct/Court, Dr/Drive; states `NC` vs `North Carolina`
- Numbers written differently: `B-200` vs `B-00200`, `9-` vs `9`, `Shop 6` vs `Shop 06`
- Wrong or altered details: house `8` -> `1`, city `Vernon` -> `Rockville`, `Gwynn Oak` -> `Gwynn Ak`,
  and a wrong city inserted (`BELGAUM` in a Delhi address)
- States written in Indian scripts: `दिल्ली`, `महाराष्ट्र`, `தமிழ்நாடு`
- Extra PO Box, unit and "H.no" parts; **~3% of S2/S3 addresses are empty**

**France (test only — no labels at all)**
- Legal forms: SARL, SAS, SASU, EURL, EI, SA
- Street words and abbreviations: Rue/R., Boulevard/BD, Cours/CRS, Avenue, Impasse, `ter` (as in 107 ter)
- Region vs département: `Nouvelle-Aquitaine` vs `Gironde`, `Hauts-de-France` vs `Nord`/`Pas-de-Calais`
- Typos (`VRDUN`, `GUYRT`), upper-case S2 addresses, website-style names (`maisonluckysas.com`)
- 483 test rows use CSV-style quotes in names (`"""ehpad Club SAS"`) — must parse consistently

### Blocking probe (5% sample of train, 383K true pairs)

| Rule: candidate if it shares... | Pair recall |
|---|---|
| a name word (3+ chars, suffixes removed) | 84.4% |
| an address number | 82.4% |
| **either of the above** | **97.7%** |
| either, but ignoring groups larger than 2,000 records | **68.3%** |

The largest groups are huge (`India / "1"`: 591K records; `US / "partners"`: 268K). Simple
rules reach high recall but produce far too many candidates; capping group size loses 30% of
true pairs. **Blocking is the core engineering problem.**

---

## 5. Validator behaviour (read from source)

- Header compared case-insensitively; use exactly `source1_entity_id\tmatched_entity_ids`.
- A row with no matches **must still contain the tab**: `S1-xxx\t`. A row without a tab is malformed.
- ID lists are split on `,` with **no trimming** — a space after a comma makes an invalid ID.
- The S1 ID column is not trimmed either; write IDs with no surrounding spaces.
- Only `\n` is stripped, so write LF line endings (a `\r` would stick to the last ID).
- The ID-existence check is off by default (`--check-ids` turns it on; uses a few GB of RAM).
- Missing `candidate_pairs.tsv` = warning only; matches not in candidates = warning only.
- It never computes a score.

---

## 6. Where the sources disagree (and the decision taken)

| Topic | Says A | Says B | Decision |
|---|---|---|---|
| Write-up length | Guidelines: 1-2 pages | Problem statement: no limit | ~2-page main body, detail in appendix |
| Final ranking | Problem statement: private LB | Guidelines: both LBs | Trust our own validation, not the public LB |
| Non-existent IDs | Constraint 2: rejected | Validator: lowers score only | Must never happen; run `--check-ids` |
| Submission format | AWS blog: "CSV" | Everything else: TSV | TSV (blog is generic/wrong) |
| Shortlist | Guidelines: Top 100 | Blog: Top 50 (PPIs) | No effect on the build |
| Model licence scope | "Final model" MIT/Apache <=8B | — | Apply to every pretrained model we use |

---

## 7. Machine

| | This Mac | Notes |
|---|---|---|
| CPU | Apple M5, 10 cores | |
| RAM | **16 GB** | The limiting factor at 10M records per split |
| Disk free | ~644 GB | Not a concern |
| Python | 3.12 in `.venv` (system is 3.14) | 3.12 chosen for wheel coverage and AWS parity |
