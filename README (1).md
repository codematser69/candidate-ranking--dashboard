# Candidate Ranking Pipeline — Redrob Hackathon 2026

A staged retrieval-and-ranking system that takes a pool of 100,000 job candidates and a single job description, and returns a ranked top 100 with a written justification for every pick — reproducible end-to-end in under 5 minutes on a CPU-only machine with no network access.

Built for the Redrob AI "Intelligent Candidate Discovery & Ranking Challenge" (target role: Senior AI Engineer, Founding Team).

---

## Table of contents

- [The problem](#the-problem)
- [Our approach](#our-approach)
- [Pipeline architecture](#pipeline-architecture)
- [Design decisions and why we made them](#design-decisions-and-why-we-made-them)
- [Handling the traps in the data](#handling-the-traps-in-the-data)
- [Compute budget](#compute-budget)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Validating a submission](#validating-a-submission)
- [Results](#results)
- [Limitations and what we'd do with more time](#limitations-and-what-wed-do-with-more-time)
- [AI tooling disclosure](#ai-tooling-disclosure)

---

## The problem

Given a real job description and 100,000 unstructured candidate profiles, produce the 100 candidates who are the best fit — ranked, scored, and explained — under constraints that make the naive approach impossible:

- **Almost no labeled data.** Only 50 hand-graded examples are available, nowhere near enough to train a ranking model directly.
- **A hard compute ceiling at submission time.** The final run must complete in 5 minutes on CPU, with no GPU and no internet access, on 16 GB of RAM — while development itself can freely use GPU and internet.
- **An adversarial dataset.** A meaningful fraction of the pool is designed to fool naive scoring: profiles stuffed with job-description keywords that have no real substance behind them, and roughly 80 "honeypot" candidates with career histories that are subtly impossible (e.g. tenure at a company that didn't exist yet). A ranker that leans on shallow textual overlap will happily promote these to the top.

The core engineering challenge isn't "build a ranker" — it's building a ranker that is simultaneously label-efficient, fast enough to run cold, and skeptical enough not to be fooled by candidates who are optimized to look good rather than be good.

## Our approach

We treat this as a funnel, not a single model: cheap, label-free methods eliminate the vast majority of the pool first, and only a small, carefully validated final stage ever sees a label. This means the 50-label budget only has to be "enough" for the last 150 candidates, not for 100,000 — which is what makes the label constraint tractable at all.

We further split the *label* problem in two: a small local language model generates several hundred weak labels offline (never during the timed run) to give the final ranker something to train on, while the 50 hand-graded candidates are reserved exclusively as a validation set we never fit to. This keeps our one source of ground truth uncontaminated.

## Pipeline architecture

```
                     100,000 candidates
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │ Stage 0 · Hard disqualifiers (rules)     │  no ML, no labels
        │ Drops profiles the JD explicitly rules   │
        │ out (e.g. pure-research careers, no      │
        │ recent hands-on coding, consulting-only) │
        └─────────────────────────────────────────┘
                              │  ~93K remain
                              ▼
        ┌─────────────────────────────────────────┐
        │ Stage 1 · Hybrid retrieval               │  no labels
        │ BM25 (lexical) + BGE-M3 (semantic),      │
        │ fused via Reciprocal Rank Fusion          │
        └─────────────────────────────────────────┘
                              │  top 300
                              ▼
        ┌─────────────────────────────────────────┐
        │ Stage 2 · Cross-encoder rerank            │  no labels
        │ BGE-Reranker-v2-m3 scores each of the     │
        │ 300 directly against the JD               │
        └─────────────────────────────────────────┘
                              │  top 150
                              ▼
        ┌─────────────────────────────────────────┐
        │ Feature engineering                       │
        │ ~20 features: experience depth, skill     │
        │ match, career trajectory, keyword-density │
        │ vs. substance ratio, 23 behavioral         │
        │ signals, tenure/company-age sanity checks │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │ Stage 3 · Ranking ensemble                │  labels used here
        │ LightGBM + XGBoost + CatBoost, trained on │
        │ ~800 LLM-weak-labels, rank-averaged,      │
        │ validated against 50 trusted labels       │
        └─────────────────────────────────────────┘
                              │
                              ▼
              Top 100, ranked, with per-candidate
              reasoning → submission CSV
```

Only Stage 3 ever touches a label. Stages 0–2 rely entirely on pretrained retrieval and reranking models, so they scale to the full 100K pool regardless of how little supervised data we have — the label scarcity only constrains the last 150 candidates, where it's actually manageable.

## Design decisions and why we made them

**Retrieval before ranking, not ranking on everything.** Scoring 100,000 candidates with an expensive model isn't necessary and wouldn't fit the compute budget anyway. Cheap lexical + embedding retrieval narrows the field by three orders of magnitude before anything expensive touches the data.

**A cross-encoder in between retrieval and the final ranker.** Bi-encoder embedding similarity is fast but coarse; a cross-encoder that scores query and candidate jointly is far more accurate but too slow to run on 100K candidates. Running it only on the 300 survivors of Stage 1 gets us cross-encoder accuracy at a fraction of the cost.

**Weak labels instead of stretching 50 labels further than they can go.** Any attempt to train directly on 50 examples — or to do things like K-fold target encoding on categorical features — would overfit badly with that little data. Rather than pretend the constraint isn't there, we generate several hundred additional weak labels offline with a small local model and treat the 50 trusted labels as a validation set only, never as training data. This keeps the one piece of ground truth we're confident in from ever leaking into the model that's being judged against it.

**Rank-averaging instead of a stacked ensemble.** A stacked meta-model needs its own held-out data to learn how to combine base models, and 50 rows isn't safely enough to carve out yet another split. Averaging the normalized rank positions from three independently trained models (LightGBM, XGBoost, CatBoost) gets most of the benefit of ensembling without needing more labels than we have.

**Manual tiers and frequency counts instead of target encoding.** Per-category average-outcome encodings (e.g. "average score for candidates from company X") are a classic leakage trap when the label set is small — the encoding can end up encoding the label itself. We use hand-curated tiers and simple counts instead, which are less expressive but don't have that failure mode.

## Handling the traps in the data

The dataset is adversarial by design, and we treat that as a feature-engineering problem rather than a rules problem:

- **Keyword stuffers** — profiles with high textual overlap against the JD but little substantiating detail — are caught by comparing keyword density against corroborating signal (depth of experience, specificity of achievements, consistency across the profile). A profile that scores high on overlap but low on substance gets penalized rather than rewarded.
- **Honeypots** — profiles with subtly impossible histories, like tenure predating a company's founding — are checked directly as a feature (tenure vs. company age, and similar consistency checks) rather than relying on the model to infer impossibility from raw text.
- We deliberately avoided writing brittle, one-off regex rules for these cases, since the trap patterns are various and rule lists don't generalize. Instead, the signals that expose fakeness are engineered as features, and the weak/trusted labels teach the ranker to distrust that combination directly.

## Compute budget

Development and the final timed run operate under different rules, and conflating them is the easiest way to break the submission:

| | Development | Final submission |
|---|---|---|
| Internet | On | **Off** |
| GPU | Allowed | **Not allowed** |
| Time limit | None | **5 minutes** |
| RAM | Unconstrained | **16 GB** |

Embedding vectors and the reranker model are computed once during development and cached to disk; the timed run only loads them, since recomputing from scratch on CPU wouldn't fit in 5 minutes. The LLM weak-labeling step is entirely an offline, one-time data-preparation step — it never executes inside the timed path.

## Repository structure

```
.
├── candidates.jsonl.gz          # 100K-candidate pool (gzipped, ~52 MB)
├── sample_candidates.json       # first 50 candidates, for schema inspection
├── candidate_schema.json        # field-by-field schema
├── job_description.md           # role being ranked against
├── redrob_signals_doc.md        # reference for the 23 behavioral signals
├── trusted_labels.csv           # 50 hand-graded candidates (validation only)
├── weak_labels.csv              # ~800 LLM-generated training labels
├── notebook/                    # full pipeline, 15 runnable cells
├── rank.py                      # end-to-end script: candidates.jsonl → submission.csv
├── validate_submission.py       # format/rules validator for the output CSV
├── submission_metadata.yaml     # team, repo, sandbox link, AI-tool declarations
└── README.md
```

## Getting started

```bash
# 1. Unpack the candidate pool
gunzip -k candidates.jsonl.gz
wc -l candidates.jsonl   # should print 100000

# 2. (Development only, internet + GPU allowed) — download and cache models
python scripts/prefetch_models.py

# 3. Run the pipeline end-to-end
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Validate the output before submitting
python validate_submission.py submission.csv
```

Before submitting, re-run the whole pipeline cold — internet off, GPU off — to confirm it actually fits inside the 5-minute / 16 GB / CPU-only budget. This caught real bugs for us during development (see notes below) and is worth doing more than once.

## Validating a submission

`validate_submission.py` enforces the submission contract directly:

- Exactly one header row (`candidate_id,rank,score,reasoning`) followed by exactly 100 data rows
- `candidate_id` matches `CAND_XXXXXXX` and is unique
- Ranks 1–100 appear exactly once each
- Score is non-increasing down the ranking, with ties broken by ascending `candidate_id`

Run it locally before every submission — it's the same check the format stage of evaluation applies.

## Results

- Runtime on the final cold configuration (CPU only, no network, 16 GB RAM): **comfortably under the 5-minute limit** — the expensive steps (embedding and reranking) are pure disk reads at submission time.
- Honeypot rate in the final top 100: **verified below the 10% disqualification threshold** as part of the pre-submission checklist.
- Reasoning is generated per-candidate from that candidate's own matched features rather than templated boilerplate, so it varies meaningfully across the 100 rows.

## Limitations and what we'd do with more time

- The weak-labeling model (Qwen2.5-1.5B-Instruct, batched CPU inference) trades some label quality for speed; a larger local model or a slower, more careful labeling pass would likely improve Stage 3's training signal.
- Our honeypot and keyword-stuffing features are hand-designed based on the trap patterns we found during development — they generalize to the patterns we identified but are not guaranteed to catch every adversarial pattern in the full 100K pool.
- With more validation data, a proper stacked ensemble (rather than rank-averaging) would likely outperform our current approach; we chose rank-averaging specifically because our label budget couldn't support a meta-model without leakage.

## AI tooling disclosure

AI tools were used during development for code review, architectural discussion, and debugging. No candidate data was sent to any external API — the weak-labeling model runs entirely locally with no network access. Full disclosure is in `submission_metadata.yaml`.

---


