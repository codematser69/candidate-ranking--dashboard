# Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

## Overview

This repository contains the full solution for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. The goal is to rank the top 100 candidates from a pool of 100,000 profiles against a specific job description for a **Senior AI Engineer (Founding Team)** role at Redrob AI.

The approach combines hand-labeled training data, a LightGBM learning-to-rank model, a hand-tuned composite scoring formula, and 23 behavioral platform signals — all blended into a single final score that powers the top-100 ranking.

---

## The Problem

Given:
- `candidates.jsonl` — 100,000 candidate profiles in JSONL format (~465 MB uncompressed)
- `sample_candidates.json` — 50 sample candidates in pretty-printed JSON
- `job_description.docx` — A detailed job description for a Senior AI Engineer role

Produce:
- A `team_submission.csv` ranking the top 100 candidates (rank 1 = best fit), with a score and a 1-2 sentence reasoning per candidate.

---

## Repository Structure

```
your-repo/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── rank.py                          ← Full pipeline as a single runnable script
├── submission_metadata.yaml         ← Team and methodology metadata
├── notebooks/
│   └── dev_notebook.ipynb           ← Development notebook (Colab-compatible)
└── output/
    └── team_submission.csv          ← Final submission CSV
```

**Single command to reproduce the submission:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./output/team_submission.csv
```

---

## Target Role Summary

**Position:** Senior AI Engineer — Founding Team  
**Company:** Redrob AI (Series A, AI-native talent intelligence platform)  
**Location:** Pune / Noida, India (Hybrid). Candidates from Hyderabad, Mumbai, Delhi NCR also considered.  
**Experience:** 5–9 years (interpretive — judgment matters more than years)  
**Employment:** Full-time

### Hard Requirements (from JD)
- Production experience with **embeddings-based retrieval** (sentence-transformers, BGE, E5, OpenAI embeddings, etc.)
- Production experience with **vector databases / hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS, etc.)
- Strong **Python** and code quality
- Hands-on **ranking evaluation frameworks** (NDCG, MRR, MAP, A/B testing, offline-to-online correlation)

### Preferred (not disqualifying if absent)
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost/neural)
- HR-tech or marketplace product background
- Distributed systems or large-scale inference optimization
- Open-source AI/ML contributions

### Explicit Disqualifiers
- Entire career spent at IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) with no product-company experience
- Pure research roles with no production deployment
- "AI experience" limited to LangChain-over-OpenAI demos (under 12 months, no pre-LLM ML background)
- Not writing production code in the last 18 months
- Notice period over 30 days (still considered, but bar is higher)

---

## Methodology

### Philosophy

The ranking system is designed around the insight stated explicitly in the JD:
> "The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap."

A candidate with a rich career history showing they built a recommendation system at a product company is a better fit than a candidate whose skills list contains every AI keyword but whose title is "Marketing Manager." The system reasons about **what the JD means**, not just what it says.

### Pipeline Architecture (5 Chunks)

#### Chunk 1 — Environment Setup & Data Ingestion

Both datasets (full 100K pool and 50-sample training set) are loaded using the **same ingestion function** to guarantee no schema drift between train and predict. A lookup dictionary is built for constant-time candidate retrieval.

Key outputs:
- `df_full` — 100,000 flattened candidate rows
- `df_sample` — 50 flattened candidate rows (training set)
- `candidates_lookup` — `{candidate_id → raw_dict}` for signal access

#### Chunk 1.5 — Data Cleaning & Quality Audit

A shared `clean_dataframe()` function is applied identically to both datasets. This covers:
- Missing-value audit on key profile fields
- Outlier clipping (`years_of_experience` capped at 50, `notice_period_days` capped at 365, etc.)
- Deduplication by `candidate_id`
- Text normalization (lowercased, whitespace-collapsed) for company, industry, location, and title fields
- Boolean flags for candidates with empty career history, education, or skills

Cleaned datasets are checkpointed to Parquet for fast reload between sessions:
```
output/candidates_full_clean.parquet
output/candidates_sample_clean.parquet
```

#### Chunk 2 — Feature Engineering & Signal Extraction

A single `engineer_features()` function produces all scoring columns for both the 100K pool and the 50-sample set. This guarantees feature columns are computed identically during training and prediction.

**Scoring components:**

| Feature | How it's computed |
|---|---|
| `score_tfidf` | TF-IDF cosine similarity between full candidate text and JD text (50K features, bigrams, sublinear TF) |
| `score_kw_required` | Fraction of required JD keywords found in candidate's full text |
| `score_kw_nice` | Fraction of preferred JD keywords found in candidate's full text |
| `score_exp_fit` | Exponential decay from ideal experience band (5–9 years) |
| `score_recency` | Exponential decay on days since `last_active_date` (half-life: 180 days) |
| `score_notice` | Exponential decay above 30-day notice ideal |
| `score_location` | Rule-based: preferred cities → 1.0, India elsewhere → 0.7, willing to relocate → 0.4, abroad no relocation → 0.1 |
| `score_edu` | Converts education tier (tier_1–tier_5) to a 0–1 score |
| `score_it_services_fit` | Penalty based on fraction of career spent at disqualifying IT services firms |
| `score_signals` | MinMax-normalized average of all 23 Redrob behavioral signals |
| `honeypot_score` | Flags impossible profiles (job durations > 15 years, expert skills with 0 months use, experience exceeding graduation year) |

**Composite Score (hand-tuned blend partner):**

```python
WEIGHTS = {
    "score_tfidf":           0.25,
    "score_kw_required":     0.15,
    "score_exp_fit":         0.12,
    "score_signals":         0.15,
    "score_it_services_fit": 0.10,
    "score_recency":         0.08,
    "score_kw_nice":         0.05,
    "score_notice":          0.05,
    "score_location":        0.04,
    "score_edu":             0.01,
}
composite_score = weighted_sum * (1 - honeypot_score * 0.9)
```

The honeypot penalty multiplicatively suppresses scores for candidates with impossible-looking profiles (engineered traps in the dataset).

#### Chunk 3 — Train on 50 → Predict on 100K

**Step 3.1 — Manual Labeling (Human Ground Truth)**

All 50 sample candidates are reviewed by hand against the job description and assigned a relevance score:
- `0` = not fit
- `1` = weak
- `2` = maybe
- `3` = good fit
- `4` = excellent fit

This is the step that makes the approach non-circular. Previous approaches that derived labels from the composite score itself were training the model to copy its own formula. Manual labeling introduces genuine human judgment as ground truth.

**Step 3.2 — LightGBM LambdaRank**

A LightGBM model is trained on the 50 labeled examples using a `lambdarank` objective (NDCG@10 metric). The model is intentionally kept shallow to limit overfitting on such a small training set:
- `num_leaves`: 7
- `min_data_in_leaf`: 3
- `num_boost_round`: 50
- `learning_rate`: 0.05

Training features:
```
score_tfidf, score_kw_required, score_kw_nice, score_exp_fit,
score_signals, score_it_services_fit, score_recency, score_notice,
score_location, score_edu, total_exp_years, education_tier
```

**Step 3.3 — Predict on Full 100K**

The trained model is applied to all 100,000 candidates using their precomputed feature rows. This is the generalization step — the model has never seen the full pool during training.

**Step 3.4 — 50/50 Blend (Overfitting Safeguard)**

Because 50 labeled examples is a very small training set, the model's raw output (`model_score`) is blended 50/50 with the hand-tuned `composite_score` after MinMax normalization of both:

```python
final_blended_score = 0.5 * model_score_norm + 0.5 * composite_score_norm
```

This guards against the model's predictions being noisy or overfit on specific label patterns from the 50-sample set, while still leveraging the model's learned ranking signal.

#### Chunk 4 — Output, Validation & Submission

- **Reasoning generation** — Per-candidate reasoning strings are auto-generated using actual profile data (title, years of experience, AI skill count, recruiter response rate). Format: `"{title} with {years:.1f} yrs; {n} AI core skills; response rate {rate:.2f}."`
- **Submission CSV** — Built with exactly 100 rows: `candidate_id`, `rank` (1–100), `score` (non-increasing), `reasoning`
- **Honeypot check** — The top 100 must have fewer than 10% honeypot-flagged candidates (threshold: `honeypot_score > 0.4`)
- **Validator** — `validate_submission.py` runs format checks before upload

---

## Candidate Data Schema

Each candidate in `candidates.jsonl` has the following top-level structure:

```
candidate_id              string   e.g. "CAND_0000001"
profile/
  headline                string
  summary                 string
  location                string
  country                 string
  years_of_experience     float
  current_title           string
  current_company         string
  current_company_size    enum     "1-10" to "10001+"
  current_industry        string
career_history[]
  company, title, start_date, end_date, duration_months
  is_current, industry, company_size, description
education[]
  institution, degree, field_of_study
  start_year, end_year, grade, tier  ("tier_1" to "tier_5")
skills[]
  name, proficiency, endorsements, duration_months
certifications[]          name, issuer, year
languages[]               language, proficiency
redrob_signals/           23 behavioral signals (see below)
```

---

## The 23 Redrob Behavioral Signals

| # | Signal | Range | What it measures |
|---|--------|-------|-----------------|
| 1 | `profile_completeness_score` | 0–100 | Profile fill-in percentage |
| 2 | `signup_date` | date | When they joined Redrob |
| 3 | `last_active_date` | date | Last login — core availability signal |
| 4 | `open_to_work_flag` | bool | Self-declared availability |
| 5 | `profile_views_received_30d` | int ≥ 0 | Recruiter interest in last 30 days |
| 6 | `applications_submitted_30d` | int ≥ 0 | Recent active job-seeking behavior |
| 7 | `recruiter_response_rate` | 0.0–1.0 | Fraction of recruiter messages replied to |
| 8 | `avg_response_time_hours` | float ≥ 0 | Median response time to recruiters |
| 9 | `skill_assessment_scores` | dict[str, 0–100] | Platform-assessed skill scores |
| 10 | `connection_count` | int ≥ 0 | Redrob network size |
| 11 | `endorsements_received` | int ≥ 0 | Total skill endorsements received |
| 12 | `notice_period_days` | 0–180 | Stated notice period |
| 13 | `expected_salary_range_inr_lpa` | {min, max} | Salary expectations in INR LPA |
| 14 | `preferred_work_mode` | enum | onsite / hybrid / remote / flexible |
| 15 | `willing_to_relocate` | bool | Willingness to relocate |
| 16 | `github_activity_score` | –1 to 100 | GitHub commits/contributions (–1 = no GitHub) |
| 17 | `search_appearance_30d` | int ≥ 0 | Recruiter search impressions in last 30 days |
| 18 | `saved_by_recruiters_30d` | int ≥ 0 | Recruiter bookmarks in last 30 days |
| 19 | `interview_completion_rate` | 0.0–1.0 | Fraction of scheduled interviews attended |
| 20 | `offer_acceptance_rate` | –1 to 1.0 | Historical offer acceptance rate (–1 = no history) |
| 21 | `verified_email` | bool | Email verification status |
| 22 | `verified_phone` | bool | Phone verification status |
| 23 | `linkedin_connected` | bool | LinkedIn account linked |

**Important:** A flat mean across all 23 signals treats them as equally important, which they are not. Availability signals (`last_active_date`, `recruiter_response_rate`, `open_to_work_flag`) are more predictive of actual hirability than passive credibility signals. Weighting these more heavily in the `score_signals` computation improves ranking quality.

---

## Submission Format

The output CSV must follow this exact format (auto-validated by `validate_submission.py`):

```
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior AI Engineer with 7.2 yrs; 6 AI core skills; response rate 0.81."
CAND_0019884,2,0.973,"Applied ML Engineer with 6.0 yrs; 5 AI core skills; response rate 0.74."
...
CAND_0007729,100,0.412,"Adjacent skills only; included as final filler given engagement signals."
```

**Format rules:**
- Exactly 100 data rows (plus 1 header row)
- Each rank 1–100 exactly once; each `candidate_id` exactly once
- `score` must be monotonically non-increasing by rank (ties allowed; break by `candidate_id` ascending)
- All `candidate_id` values must exist in `candidates.jsonl`
- UTF-8 encoding, `.csv` extension
- Filename: your registered participant ID (e.g. `team_xxx.csv`)

**Run the validator before submitting:**
```bash
python validate_submission.py output/team_submission.csv
```

---

## Scoring Metrics

Submissions are scored against a hidden ground truth using:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| NDCG@10 | 0.50 | Quality of top-10 picks |
| NDCG@50 | 0.30 | Quality of top-50 picks |
| MAP | 0.15 | Precision across all relevance levels |
| P@10 | 0.05 | Fraction of top-10 that are "relevant" (tier 3+) |

**Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10**

There is no live leaderboard. Scores are revealed only after submissions close.

---

## Evaluation Stages

| Stage | What happens | Elimination criteria |
|-------|-------------|---------------------|
| 1. Format validation | Auto-validator runs on every submission | Any spec violation |
| 2. Scoring | Composite score computed against hidden ground truth | Score below advancement cutoff |
| 3. Code reproduction + honeypot check | Full repo requested; ranking step reproduced in sandboxed Docker (5 min, 16 GB RAM, CPU only, no network) | Cannot reproduce; honeypot rate > 10%; missing or fabricated repo |
| 4. Manual review | Reasoning quality (specificity, JD connection, honesty, no hallucination, variation, rank consistency) | Templated/empty reasoning; contradicts rank; skills not in profile |
| 5. Defend-your-work interview | 30-minute video call — walk through architecture and defend design choices | Cannot explain architecture; contradicts code; clearly didn't build it |

---

## Compute Constraints

The **ranking step** (producing the CSV from candidates) must satisfy:

| Constraint | Limit |
|---|---|
| Runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | Off — no external API calls during ranking |
| Disk | ≤ 5 GB intermediate state |

Pre-computation (generating embeddings, training the model) may exceed the 5-minute window and should be documented separately. Only the final ranking step is time-constrained.

---

## Honeypot Warning

The dataset contains approximately 80 **honeypot candidates** with subtly impossible profiles. Examples:
- 8 years of experience at a company founded 3 years ago
- "Expert" proficiency claimed in 10 skills, each with 0 months of use

These are forced to relevance tier 0 in the ground truth. Submissions with honeypot rate > 10% in the top 100 are disqualified at Stage 3.

The `detect_honeypot()` function in `rank.py` catches these patterns and applies a multiplicative penalty to their composite score.

---

## Dataset Traps (Read Carefully)

The dataset is deliberately constructed with the following traps:

1. **Keyword stuffers** — Candidates with every required AI keyword in their skills list but no supporting career history (e.g., "Marketing Manager" with 15 AI skills listed)
2. **Plain-language Tier 5 fits** — Candidates who are genuinely good fits but don't use the exact JD terminology (e.g., built a recommendation system without using the word "RAG")
3. **Behavioral twins** — Two candidates with near-identical profiles but very different engagement signals (one active, one dormant for 6+ months)
4. **Honeypot candidates** — Impossible profiles designed to trip up pure keyword-embedding systems

A good ranking system naturally avoids these traps by reasoning about the substance of career history, not just keyword overlap.

---

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
tqdm
```

Install with:
```bash
pip install pandas numpy scikit-learn lightgbm tqdm
```

Python 3.9+ recommended.

---

## Key Design Decisions

**Why train on 50 labeled examples?**
Manual labeling introduces genuine human judgment as ground truth, avoiding the circular trap of deriving labels from the composite score itself (which would just train the model to copy its own formula).

**Why blend model output with composite score?**
50 labeled examples is a very small training set. The 50/50 blend guards against overfitting or noisy model predictions while still demonstrating real ML re-ranking on top of the hand-tuned features. The blend ratio can be adjusted (e.g., 30/70) based on how well the top-20 sanity check looks.

**Why use a shallow LightGBM tree (7 leaves)?**
On ~50 training rows, a deeper tree would memorize the training data rather than generalize. The shallow structure forces the model to capture only the strongest ranking signals.

**Why does the composite score weight TF-IDF so highly (0.25)?**
TF-IDF captures semantic overlap between the full candidate text (title, headline, summary, skills, career descriptions, education) and the full JD text — including surrounding context, not just a keyword list. This rewards candidates whose actual career narratives align with what the JD describes, even if they don't use the exact terminology.

---

## Submission Checklist

Before uploading, verify:

- [ ] Exactly 100 data rows in the CSV
- [ ] Ranks 1–100 present exactly once each
- [ ] No duplicate `candidate_id` values
- [ ] All `candidate_id` values match `CAND_XXXXXXX` format (7 digits)
- [ ] `score` is non-increasing from rank 1 to 100
- [ ] Honeypot rate in top 100 is ≤ 10%
- [ ] `validate_submission.py` reports "Submission is valid."
- [ ] `submission_metadata.yaml` is filled in and committed to repo
- [ ] README includes single reproduce command
- [ ] Sandbox link is working and handles ≤100 candidates end-to-end
- [ ] At most 3 total submissions (last valid submission counts)

---

## AI Tools Declaration

This project used AI tools as part of the development workflow. All candidate ranking logic, feature engineering, and model training were designed and implemented by the team. No candidate data was fed to any hosted LLM API. The ranking step runs fully offline on CPU within the 5-minute compute budget.

---

## Contact

Fill in your team details in `submission_metadata.yaml` before submitting. For bundle bugs or validator issues, use the official hackathon support channel.
