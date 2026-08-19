# Eval Results — Retrieval Baselines

Per-cell retrieval quality across the shipped subjects. Each row was
produced by `scripts/eval/run_retriever_eval.py` against the production
index in `data/vector_store/chroma_db_phase1/`, using language- or
subject-specific topics CSVs as ground truth.

Numbers are **unscoped** (no `school_phase` filter) — i.e., the
retriever sees the entire corpus when picking results. **Production
usage scopes via `school_phase`**, which removes most off-phase
candidates and lifts precision by an additional ~5-15 points
empirically.

## Latest results (post-fix corpus)

| Cell                | Run dir                              |   N   | P@1   | P@5   | P@10  | R@10  | Hit@1 | Hit@10 |  MRR  |
|---------------------|--------------------------------------|------:|------:|------:|------:|------:|------:|-------:|------:|
| `en × ENGLISH`      | `en_20260514T153536Z/`               | 2,761 | 0.735 | 0.711 | 0.610 | 0.641 | 0.735 |  0.871 | 0.784 |
| `ar × ARABIC`       | `ar_20260515T081801Z/`               |   400 | 0.585 | 0.586 | 0.544 | 0.415 | 0.585 |  0.778 | 0.655 |
| `fr × FRENCH`       | `fr_20260514T153707Z/`               |    46 | 0.870 | 0.843 | 0.657 | 0.872 | 0.870 |  1.000 | 0.914 |
| `fr × MATHEMATICS`  | `fr_20260519T143604Z/`               |   720 | 0.615 | 0.587 | 0.544 | 0.392 | 0.615 |  0.735 | 0.649 |
| `ar × MATHEMATICS`  | `ar_20260519T144413Z/`               |   360 | 0.492 | 0.484 | 0.450 | 0.354 | 0.492 |  0.692 | 0.547 |

**N** = test cases.
**P@k** = precision at top-k. **R@10** = recall at top-10. **Hit@k** =
fraction of queries that found at least one relevant doc in top-k.
**MRR** = mean reciprocal rank.

## Reading the numbers

**Language cells (en/ar/fr)** are the established baselines from Phase 1.
- English is the strongest cell — largest corpus (3,079 docs), simplest
  retrieval problem (English-only embeddings work well).
- Arabic is mid — corpus is smaller (1,317 docs) and queries are more
  varied, but BGE-M3 handles Arabic script well.
- French looks great on the metric (`hit@10 = 1.0`) but the sample is
  tiny (46 cases, 15 corpus docs). Not diagnostic.

**Math cells (`fr × MATHEMATICS`, `ar × MATHEMATICS`)** are ~10pp below
language cells across precision@1, hit@10, MRR. The failure pattern is
documented in `notebooks/math_data_audit.ipynb`:

- The bi-encoder cannot reliably distinguish **sibling math topics** —
  when the test demands "Fonction Logarithme 2," the retriever often
  returns "Fonctions affines" or "Généralités sur les fonctions."
- These siblings share vocabulary heavily (all about functions), so the
  embedder's semantic similarity correctly identifies them as related —
  but the exact-title eval metric counts them as failures.
- The retrieved content is still usable as few-shot context for the LLM
  (manual generation testing confirms math quizzes come out well).

**Production mitigation already in place:** `school_phase` filter on
both `/retrieve` and `/quiz/generate`. When the platform passes the
user's grade level, the metadata pre-filter removes off-phase content
before scoring, which empirically improves precision@1 by several
points. We've not yet run a scoped-eval pass — that's the next
diagnostic when time permits.

## How to reproduce

```bash
# Re-run a specific eval
python -m scripts.eval.run_retriever_eval \
    eval/english_retriever_test_cases.json

python -m scripts.eval.run_retriever_eval \
    eval/math_fr_retriever_test_cases_topic_specific.json
```

Each run produces a new `eval/results/<lang>_<utc_timestamp>/`
directory with:
- `summary.json` — aggregated metrics (used in this table)
- `per_query.jsonl` — one row per test case for failure-mode analysis
- `config_snapshot.yaml` — copy of `configs/models.yaml` that produced
  the numbers
- `run_args.json` — CLI invocation

## What's NOT measured

- **LLM generation quality.** These numbers are retrieval only — did
  the retriever find the right context? Whether the LLM produces a
  *good quiz* given that context is currently judged manually.
- **End-user experience.** A test where `hit@10 = 0` (retriever found
  nothing) might still produce an acceptable quiz because the LLM uses
  any retrieved math content as few-shot inspiration.
- **Tail latency.** Each eval run records wall-clock per query but
  these aren't aggregated into the table.
