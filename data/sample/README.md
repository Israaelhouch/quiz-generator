# Sample corpus — SYNTHETIC DEMO DATA

`quizzes-sample-raw.json` in this folder is **synthetic demo data**, not the real
curriculum corpus.

The production corpus (`data/raw/quizzes-raw-data.json`, ~1,372 quizzes /
12,480 questions) is private and is **not** part of this repository. This file
exists so that anyone who clones the repo can run the whole pipeline —
ingest → normalize → build_index_text → indexing.build → retrieval →
generation — without it.

## What's inside

24 quizzes / 149 questions, all written for this demo:

| | |
|---|---|
| Subjects | `ENGLISH`, `FRENCH`, `ARABIC`, `MATHEMATICS` (+ one out-of-scope `PHYSICS` quiz) |
| Languages | `en`, `fr`, `ar` |
| School phases | `PRIMARY`, `MIDDLE`, `HIGH` across 12 levels |
| Question types | `MULTIPLE_CHOICE`, `FILL_IN_THE_BLANKS`, one `TEXT_MULTIPLE_CHOICE` |
| Maths | LaTeX markup (`\frac`, `\sqrt`, `\rightarrow`, `\left(...\right)`) |

Content is deliberately generic and textbook-flavoured ("What is the past tense
of 'go'?", "Quelle est la capitale de la France ?", `\(2x + 3 = 11\)`). Author
names and emails are obvious placeholders on `@example.com`. **No content, no
metadata and no person from the private corpus appears here.**

The file also carries deliberately messy rows — a duplicate question, numeric
`answer` values, a question with no correct choice, an image-only question, an
empty-choices question, colliding question `order` values, out-of-scope
subject/level rows and a curriculum-violating maths row — so the cleaning
stages have real work to do and the stats files are worth reading.

## Regenerating it

The file is produced by a deterministic generator (fixed seed, no network,
stdlib only), so it can always be reproduced byte-for-byte:

```bash
python -m scripts.make_sample_corpus        # -> data/sample/quizzes-sample-raw.json
# or
make sample-corpus
```

Edit `scripts/make_sample_corpus.py`, not this JSON file — `tests/test_sample_corpus.py`
asserts that the checked-in file matches the generator's output.

## Running the pipeline on it

```bash
make sample-data     # stages 1-3 (no ML dependencies needed)
make sample-build    # stages 1-4, downloads BGE-M3 (~2GB) and builds Chroma
make sample-demo     # retrieval + generation with the mock LLM (no API key)
```

Pipeline outputs land in `data/sample/interim/`, `data/sample/processed/` and
`data/vector_store/sample/` — all gitignored, and all separate from the real
`data/processed/ready_phase1.jsonl` / `data/vector_store/chroma_db_phase1`, so a
sample run can never overwrite a real build.

Row counts through the stages (see the `*_stats.json` files for the full audit):

| Stage | Rows out |
|---|---|
| raw sample JSON | 149 questions in 24 quizzes |
| 1. ingest (scope-filtered) | 142 |
| 2. normalize | 139 |
| 3. build_index_text | 139 |
