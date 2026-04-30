# Stage 1 — Data Understanding & Scoping

Status: LOCKED (2026-04-21)
Next stage: 2a — Ingestion design

---

## 1. Source of truth

- **Input:** `data/raw/quizzes-raw-data.json`
  - 1,372 quizzes
  - 12,480 questions total
- Old intermediate files (`questions_flat.csv`, `questions_clean_strict_rag_with_metadata.csv`) are **archived** to `legacy/`. They are not used.

## 2. Raw-data observations (for context)

Shape:
- 1,372 quizzes, 12,480 questions.
- Median 8 questions per quiz. Range 0–62.
- 1,083 published, 289 unpublished.
- Only 6 `generatedByAI=true` quizzes — data is human-authored.

Languages at quiz level (raw values):
- english=854, arabic=211, french=195
- Variants: English=35, en=16, fr=12, ar=3
- spanish=1, null=45

Question types:
- MULTIPLE_CHOICE: 12,320 (98.7%)
- FILL_IN_THE_BLANKS: 149 (1.2%)
- TEXT_MULTIPLE_CHOICE: 11 (0.1%) → **merged into MULTIPLE_CHOICE at ingestion** (Step 3 simplification)

Data-integrity red flags detected:
- 897 questions with **no correct answer** (7.2%) — mostly MULTIPLE_CHOICE.
- 897 questions with **>1 correct answer**, but 870 of those are flagged `multipleChoice=false` → the `multipleChoice` field is unreliable.
- 99% of descriptions contain HTML tags.
- 120 questions with fully empty description.
- 1,124 questions with description shorter than 5 chars after HTML strip (typically LaTeX or image-only).
- 3,141 questions (25%) have a non-null `image` field.
- `hintText` is mostly junk ("uhuihuihiu").
- Explanations exist on only 4.2% of questions.

## 3. Scope filters (applied during ingestion)

Rows are **dropped** when any of the following is true:
1. No correct answer: no choice has `isTrue=true`.
2. Image-only: `image` is non-null AND description is empty after HTML stripping.
3. Language not in {en, fr, ar} after normalization (Spanish and unlabeled rows dropped).

All other filters (subject, level, publish, generatedByAI) are **not applied**.

## 4. Trust rules

- **`type` is authoritative.** Do not trust the raw `multipleChoice` flag.
- `multiple_correct_answers` is **derived**: `sum(1 for c in choices if c.isTrue) > 1`.
- Language is **re-detected** from quiz title if the raw label is missing or low-confidence.

## 5. Final column schema

| Column | Source | Tier |
|---|---|---|
| `doc_id` | derived: `{quiz._id}__q{question.order}` | metadata |
| `quiz_id` | `quiz._id` | metadata |
| `quiz_title` | `quiz.title` (with "Quiz:" prefix stripped) | metadata + embedding |
| `language` | normalized `quiz.language` | metadata |
| `subjects` | `quiz.subjects` (list) | metadata + embedding |
| `levels` | `quiz.levels` (list) | metadata only |
| `question_type` | `question.type` | metadata |
| `multiple_correct_answers` | derived: `sum(isTrue) > 1` | metadata |
| `question_text` | `question.description` (HTML-stripped) | payload + embedding |
| `choices_text` | `[c.answer for c in choices]` (cleaned) | payload + embedding |
| `correct_choices_text` | `[c.answer for c in choices if c.isTrue]` | payload (embedding TBD in 2c) |
| `points` | `question.points` | payload |
| `time` | `question.time` | payload |
| `author_name` | `quiz.createdBy.name` | metadata (filter) |
| `author_email` | `quiz.createdBy.email` | metadata (filter) |
| `search_text` | composed in Stage 2c | embedding input |

**Dropped entirely:** `order`, `explanation`, `hintText`, `showHint`, `image`, `stats`, `results`, `covers`, `createdAt`, `updatedAt`, `publish`, `generatedByAI`, quiz-level `description`, raw `multipleChoice`.

## 6. Architecture

- **Intermediate/processed format:** JSONL (readable, streamable, preserves nested structure).
- **Schema validation:** Pydantic v2 models at every stage boundary.
- **Retrieval code:** keep `src/retrieval/` as-is (well-designed, tested).
- **Rebuild:** ingestion, normalization, search_text composition, vector-store build.

## 7. Pipeline stages (RAG framework)

```
1. Data Understanding & Scoping        — this document
2. Data Cleaning & Preparation         — src/data/
   2a. ingest.py           raw JSON         → interim/flat.jsonl
   2b. normalize.py        flat             → interim/normalized.jsonl
   2c. build_index_text.py normalized       → processed/ready.jsonl
3. Embedding & Vector Store            — src/indexing/
4. Retrieval Logic                     — src/retrieval/ (kept)
5. Generation Prompt Design            — src/generation/prompts/
6. RAG Pipeline Assembly               — src/pipeline/
7. Evaluation & Quality Check          — src/eval/
```

## 8. New repo layout

```
quiz-generator/
├── data/
│   ├── raw/            quizzes-raw-data.json
│   ├── interim/        flat.jsonl, normalized.jsonl
│   ├── processed/      ready.jsonl
│   └── vector_store/   (gitignored)
├── src/
│   ├── data/           ingest.py, normalize.py, build_index_text.py
│   ├── indexing/       build_vector_store.py
│   ├── retrieval/      (kept as-is)
│   ├── generation/     prompts/, generator.py
│   ├── pipeline/       quiz_pipeline.py
│   ├── eval/           metrics.py, eval_runner.py
│   └── shared/         schemas.py (Pydantic), utils.py
├── configs/            pipeline.yaml, models.yaml
├── tests/
├── scripts/            CLI entrypoints
├── docs/               01_scope.md, ...
└── legacy/             archived old code + CSVs
```

## 9. Deferred decisions

- **Include `correct_choices_text` in `search_text`?** — decided in Stage 2c (after we see normalized data).
- **Subject-taxonomy collapsing** (e.g., `PHYSICS_1-MECHANICS` → `PHYSICS`) — decided in Stage 2b.
- **Quiz structure** (questions per quiz, difficulty mix, randomization) — Stage 5+ decision.
