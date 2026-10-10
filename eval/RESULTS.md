# Eval Results — Retrieval Baselines

Per-cell retrieval quality across the shipped subjects. Each row was
produced by `scripts/eval/run_retriever_eval.py` against the production
index in `data/vector_store/chroma_db/`, using language- or
subject-specific topics CSVs as ground truth.

Numbers are **unscoped** (no `school_phase` filter) — i.e., the
retriever sees the entire corpus when picking results. **Production
usage scopes via `school_phase`**, which removes most off-phase
candidates and lifts precision by an additional ~5-15 points
empirically.

## Latest results

The corpus was narrowed to the three language subjects on 2026-10-04
(ADR-0009) and the index rebuilt: 4,411 questions instead of 5,782. The
mathematics cells left the scope and their last measurements are kept in
[Retired cells](#retired-cells) rather than deleted.

| Cell                | Run dir                              |   N   | P@1   | P@5   | P@10  | R@10  | Hit@1 | Hit@10 |  MRR  |
|---------------------|--------------------------------------|------:|------:|------:|------:|------:|------:|-------:|------:|
| `en × ENGLISH`      | `en_20260910T093958Z/`               | 2,761 | 0.806 | 0.795 | 0.684 | 0.638 | 0.806 |  0.875 | 0.828 |
| `ar × ARABIC`       | `ar_20260515T081801Z/`               |   400 | 0.585 | 0.586 | 0.544 | 0.415 | 0.585 |  0.778 | 0.655 |
| `fr × FRENCH`       | `fr_20261010T122124Z/`               |    46 | 0.870 | 0.843 | 0.657 | 0.872 | 0.870 |  1.000 | 0.914 |

**N** = test cases.
**P@k** = precision at top-k. **R@10** = recall at top-10. **Hit@k** =
fraction of queries that found at least one relevant doc in top-k.
**MRR** = mean reciprocal rank.

**Which run each row came from matters, so the run directory is in the table.**
French was re-measured on 2026-10-10 against the rebuilt index and reproduced
its May figures to the fourth decimal (P@1 0.8696 against 0.870, Hit@10 1.000,
MRR 0.9143 against 0.914). That is the expected result rather than a lucky
one: Chroma applies the subject filter *before* scoring, so rows from another
subject never competed for these slots, and removing them cannot move the
number. English and Arabic are earlier runs against the larger index for the
same reason — the English row was re-run on 2026-09-10 against a repaired
answer key (next section; it previously read P@1 0.735), and the Arabic row is
the May run. All three answer keys were re-validated against the rebuilt index
on 2026-10-10 and pass.

Before quoting any of these numbers, read [Limits](#limits-of-these-numbers)
and [Realistic teacher questions](#realistic-teacher-questions-english-2026-09-10).

## English: the answer key was stale (repaired 2026-09-10)

The May English number was measured against a stale answer key.
`eval/topics_english.csv` was built on 2026-05-13; the doc_id collision fix
landed on 2026-05-18 and gave colliding questions new ids such as `__q3_2`.
The Arabic and maths keys were rebuilt after that fix; the English one never
was. Against the current index it omitted 205 questions belonging to its
topics (178 of them re-numbered by the fix), listed one id that no longer
exists, and carried 120 duplicate ids left over from the collisions. A
retriever returning any of the missing questions was scored as wrong.

It was repaired in three separate steps, each measured on its own so that
every change in the score can be attributed:

| Answer key | Run | P@1 | P@5 | Hit@10 | MRR | P@1 well-specified | P@1 `vague` |
|---|---|--:|--:|--:|--:|--:|--:|
| Stale key (published; reproduced 2026-09-08) | `en_20260908T125614Z/` | 0.735 | 0.711 | 0.871 | 0.784 | 0.854 | 0.197 |
| + step 1: refresh doc_ids from the index | `en_20260910T083401Z/` | 0.795 | 0.782 | 0.872 | 0.819 | 0.925 | 0.209 |
| + step 2: safe title merges | `en_20260910T090721Z/` | 0.802 | 0.790 | 0.875 | 0.825 | 0.934 | 0.209 |
| + step 3: reviewed title merges | `en_20260910T093958Z/` | 0.806 | 0.795 | 0.875 | 0.828 | 0.938 | 0.209 |

- **Step 1 — refresh.** Every topic's doc_ids brought in line with the index,
  keeping the topic names and the hand merges (`scripts/eval/refresh_topics.py`).
  Re-scoring the *earlier* run's retrieved lists against the refreshed key
  reproduces the new run exactly: the gain is entirely the answer key, not a
  change in retrieval.
- **Step 2 — safe merges.** Titles that differ from a topic's titles only in
  case, spacing, punctuation or a leading article were folded in (6 titles,
  47 questions). Predicted 0.8015 by re-scoring; measured 0.8019.
- **Step 3 — reviewed merges.** 11 more titles (typos, numbered parts), each
  checked against sample questions from both titles. Four candidates were
  left out: three judged wrong — among them the one that would have raised the
  score the most — and one only partly on topic. Predicted 0.8055; measured
  0.8059.

The final key is reproducible: rebuilding it from the original CSV step by
step, or in a single pass, gives identical doc_ids for every topic. The
validator now refuses to start an eval whose answer key disagrees with the
index, so a key cannot go stale silently again.

## Retired cells

Mathematics shipped in v1.1.0 and left the scope on 2026-10-04 (ADR-0009):
retrieval over formula-bearing questions is a different task from retrieval
over prose, and in this corpus it was also the only place French content
existed in volume, which made the French cell look larger than the language
itself is here. These are the last measurements taken while it was in scope,
against the 5,782-row index.

| Cell                | Run dir                              |   N   | P@1   | P@5   | Hit@10 |  MRR  |
|---------------------|--------------------------------------|------:|------:|------:|-------:|------:|
| `fr × MATHEMATICS`  | `fr_20260519T143604Z/`               |   720 | 0.615 | 0.587 |  0.735 | 0.649 |
| `ar × MATHEMATICS`  | `ar_20260519T144413Z/`               |   360 | 0.492 | 0.484 |  0.692 | 0.547 |

Measured against their own ceilings (below) these were 87% and 74%, so the
gap to the language cells was mostly the test sets rather than the retriever.
The test cases and answer keys still exist locally; the cells can be restored
by adding MATHEMATICS back to `configs/scope.yaml` and rebuilding, which
`tests/test_scope.py` deliberately makes a conscious act rather than an
accident.

## Limits of these numbers

### Some test cases cannot be passed

Some queries appear several times, each time labelled with a different
correct quiz. A retrieved question belongs to exactly one quiz, so all but one
of those cases must fail, however good the retriever is. The best P@1 a
perfect retriever could reach on each test set:

| Cell | Cases | Cannot be passed | Perfect-retriever P@1 | Measured P@1 | Share of ceiling |
|---|--:|--:|--:|--:|--:|
| `en × ENGLISH` | 2,761 | 313 | 0.887 | 0.806 | 91% |
| `ar × ARABIC` | 400 | 43 | 0.892 | 0.585 | 66% |
| `fr × FRENCH` | 46 | 6 | 0.870 | 0.870 | 100% |
| `fr × MATHEMATICS` † | 720 | 210 | 0.708 | 0.615 | 87% |
| `ar × MATHEMATICS` † | 360 | 120 | 0.667 | 0.492 | 74% |

† retired on 2026-10-04 — see [Retired cells](#retired-cells).

The worst single query is `'english grammar exercises'`, labelled with 206
different quizzes. That is most of why the `vague` query type scores so
poorly: on the repaired English key, well-specified queries reach P@1
0.938 and `vague` ones 0.209.

### The queries are templates that contain the answer

The English and Arabic test cases were generated from fixed templates —
`'{topic}'`, `'{topic} exercises'`, `'How do I learn {topic}?'` and so on — so
90% of English and 100% of Arabic non-vague queries contain the target quiz
title word for word (for Arabic, once vowel marks are ignored). `search_text`
embeds that same title. These cells
therefore mostly measure whether retrieval can match a title, not whether it
understands how a teacher phrases a request. The French queries are the
exception and are genuinely semantic — 2% contain the title — which is one
reason that cell scores as it does on 46 cases. The retired maths sets sat in
between (62% of French and 34% of Arabic maths queries contained the title).
The generator that produced the test cases is not in the repository, so they
cannot be regenerated; this is also why the realistic set below was written by
hand.

### Retrieval is not exactly reproducible

Between two runs on the same index and configuration, Chroma's approximate
(HNSW) search returned a different 60-document candidate pool for 1,313 of
2,761 queries. Every document found in both runs had a bit-identical distance,
so the query embeddings did not change; the search did. Aggregate metrics
moved by at most ±0.0005, but individual cases can flip, occasionally losing
their best match entirely. Compare runs, not single queries.

## Realistic teacher questions (English, 2026-09-10)

Because the template queries contain their answer, a second English test set
was written: 50 questions phrased the way a teacher might ask, none containing
the words of its target quiz titles. Each lists every quiz that is a fair
answer, since a realistic request rarely maps to exactly one. The questions were
reviewed before any were scored, and their answers were labelled without using
the search model.

| Test set | P@1 | P@5 | Hit@10 | MRR |
|---|--:|--:|--:|--:|
| Template questions, well-specified (2,259) | 0.938 | 0.924 | 0.989 | 0.955 |
| Realistic questions (50) | 0.760 | 0.572 | 0.940 | 0.826 |

- **The search finds the right quiz but ranks it lower.** 94% of realistic
  questions have a correct quiz in the top 10, but only 76% have one first.
- **It is weakest when a teacher describes a problem** instead of naming the
  grammar point: P@1 0.59 for questions like "students mix up 'she cooks' and
  'she is cooking'", against 0.76 for example sentences, 0.88 for casual
  phrasing and 1.00 for skill-and-level requests.
- **The sample is small.** With 50 questions the 95% range for P@1 is
  0.64–0.88. Use this set to find weaknesses, not to compare close
  configurations.
- **The first run scored 0.680.** A later audit of every question's answers,
  across all 206 topics, found fair quizzes the first labelling had missed
  because it only considered topics with at least 8 questions. It added answers
  to 13 questions — 9 that had passed and 4 that had failed — and the search
  results were identical in both runs. Six questions share a word with one of
  their correct quiz titles; without them P@1 is about 0.73.

## Reading the numbers

**English** is the strongest cell: the largest corpus (3,079 docs) and, on
the repaired key, P@1 0.806 against a possible 0.887.

**Arabic** is the weakest *relative to what its test set allows*: 0.585 of a
possible 0.892. BGE-M3 handles Arabic script well; where the rest is lost has
not been investigated.

**French** looks great on the metric (`hit@10 = 1.0`) but the sample is
tiny (46 cases, 15 corpus docs). Not diagnostic.

**Maths**, while it was in scope, sat below the language cells in raw numbers,
but much of that gap was its test set. 40–50% of maths cases shared a query
with a different target title, which capped P@1 at 0.708 for French maths and
0.667 for Arabic maths. Measured against those ceilings, French maths reached
87%, close to English at 91%; Arabic maths reached 74%. Sibling topics are
genuinely
close — a query about functions can fairly match "Fonction Logarithme 2",
"Fonctions affines" or "Généralités sur les fonctions" — so this is a
test-design problem at least as much as a retrieval one. The retrieved content
is still usable as few-shot context for the LLM (manual generation testing
confirms math quizzes come out well).

**Production mitigation already in place:** `school_phase` filter on
both `/retrieve` and `/quiz/generate`. When the platform passes the
user's grade level, the metadata pre-filter removes off-phase content
before scoring, which empirically improves precision@1 by several
points. We've not yet run a scoped-eval pass — that's the next
diagnostic when time permits.

## How to reproduce

```bash
# Check an answer key against the index (the eval also does this first, and stops on failure)
python -m scripts.eval.validate_test_cases eval/english_retriever_test_cases.json

# After rebuilding the index, refresh the answer keys (dry run unless --write)
python -m scripts.eval.refresh_topics \
    --aliases eval/topic_aliases.safe.yaml \
    --aliases eval/topic_aliases.reviewed.yaml

# Re-run an eval
python -m scripts.eval.run_retriever_eval \
    eval/english_retriever_test_cases.json

# The realistic English set
python -m scripts.eval.run_retriever_eval \
    eval/english_realistic_test_cases.json
```

The same runs are available as `make eval-validate`, `make eval` and
`make eval-realistic`.

The test cases (including the realistic set), topics CSVs and alias files are
derived from the private corpus and are not part of this repository.

Each run produces a new `eval/results/<lang>_<utc_timestamp>/`
directory with:
- `summary.json` — aggregated metrics (used in this table)
- `per_query.jsonl` — one row per test case for failure-mode analysis
- `config_snapshot.yaml` — copy of `configs/models.yaml` that produced
  the numbers
- `pipeline_snapshot.yaml` — copy of `configs/pipeline.yaml`, the
  `search_text` recipe the index was built with
- `provenance.json` — git SHA, config hashes, and the index this run
  searched: payload hash, model, dimension, collection, row count, recipe.
  Warnings here name anything the run could not account for
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
