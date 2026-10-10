# Cells and plan

Which (language × subject) cells this service ships, which are beta, which are
out of scope, and the acceptance criteria each one is held to.

---

## Goal

Move the quiz generator from "MVP that works" to "production-grade for Tunisian
teachers." Strategy: cell-based development — define scope, then iterate per
cell to acceptance criteria via eval-driven tuning.

---

## Current scope (locked)

A *cell* is a `(language, subject)` pair. Each cell has its own quality bar,
its own failure modes, and its own tuning. The locked scope is:

| ID  | Cell                | Language | Subject     | Status | Notes |
|-----|---------------------|----------|-------------|--------|-------|
| C1  | `ar × ARABIC`       | ar       | ARABIC      | ✅ v1.0 | Arabic literature / grammar — diacritics in source, usually absent in queries |
| C2  | `en × ENGLISH`      | en       | ENGLISH     | ✅ v1.0 | Best data coverage in corpus — easiest cell |
| C3  | `fr × FRENCH`       | fr       | FRENCH      | ✅ v1.0 | Limited corpus (~15 rows) — beta status |
| C4  | `fr × MATHEMATICS`  | fr       | MATHEMATICS | ⛔ Out of scope since 2026-10-04 | Shipped in v1.1.0, 1,003 docs. Removed with ADR-0009: formula-bearing retrieval is a different task from prose, and this cell was the only place French existed in volume, which made the French cell look larger than the language is here. |
| C5  | `ar × MATHEMATICS`  | ar       | MATHEMATICS | ⛔ Out of scope since 2026-10-04 | Shipped in v1.1.0, 368 docs. Removed with ADR-0009, alongside C4. |

### Out of scope

- **Higher-ed math** (PREPARATORY, LICENCE levels). Calculus notation
  (`\int`, `\sum`, `\lim`) is absent from the current scope's corpus;
  adding it requires LaTeX-rendering hardening and possibly SymPy
  verification. Queued for a later phase.
- **Physics, Chemistry, Sciences, Computer Science, History, Technique** —
  future scope. These reuse the same retrieval + generation stack as the
  current cells, so adding them is a data-and-eval task, not an
  infrastructure task. Each will need a curriculum rule in
  `src/quiz_generator/curriculum/curriculum_rules.py` if its (subject, phase) → language
  mapping is constrained.

---

## School levels (coarse grouping, distinct from project scope)

This is about Tunisian school structure, not project planning. Levels are
grouped on top of the existing fine-grained `levels` field, derived from
`levels[0]` prefix at index time. Stored as a Chroma metadata scalar field
for native pre-filtering.

| Level group | Maps from `levels[0]` prefix | In current scope? |
|-------------|------------------------------|-------------------|
| `PRIMARY`   | `PRIMARY_SCHOOL_*`           | ✅ yes |
| `MIDDLE`    | `MIDDLE_SCHOOL_*`            | ✅ yes |
| `HIGH`      | `HIGH_SCHOOL_*`              | ✅ yes |

PRIMARY_SCHOOL was originally excluded under "MIDDLE + HIGH have most demand."
That rationale didn't survive contact with the data — Arabic primary alone
has 802 rows vs 515 for high school. Since the project scope is
language-only (Arabic, English, French — no math, sciences, etc.), there's
no infrastructure reason to exclude primary; the same retrieval and
generation stack handles it. Callers who want to scope per query can still
use the `levels` filter in the API to restrict to a specific school level.

---

## Plan

Three stages. Don't move forward until each is "done."

### Stage B — Build golden eval set
Per-cell test queries with hand-picked relevant `doc_ids` as ground truth.

- `notebooks/eval_dataset_english.ipynb` → `eval/topics_english.csv`, `eval/golden_set_english.jsonl`
- `notebooks/eval_dataset_arabic.ipynb` → `eval/topics_arabic.csv`, `eval/golden_set_arabic.jsonl`
- `notebooks/eval_dataset_french.ipynb` → `eval/topics_french.csv`, `eval/golden_set_french.jsonl`
  (BETA — 15 rows, sanity check only, not a tuning target)

**Approach:** aggregate by `quiz_title`, treat all `doc_id`s sharing a title
as the relevant set for any query about that topic. ~6 queries per cell for
en / ar; 2–3 for fr (corpus too small for more).

### Stage C — Retriever eval & tuning
Component-isolated: feed golden queries into the retriever, score per-cell.

- Metrics: precision@k, recall@k, MRR, nDCG.
- Compare bi-encoder only vs bi-encoder + reranker.
- Tune `default_max_distance` and `candidate_pool_size` per cell.
- Output: `notebooks/retriever_eval.ipynb` with per-cell numbers.

### Stage D — Generator eval & tuning
End-to-end: golden query → retriever → LLM → quiz output.

- Per-cell prompt tuning.
- Validate: structural correctness, factual accuracy, coverage, no leakage of
  source choices.
- Output: `notebooks/generator_eval.ipynb`.

---

## Branch / workflow

```
main                            (production; tag v1.0.0 = Phase 1 release)
└── dev                         (integration)
    └── feature/math-subject    (current — Phase 2 math, 7 commits ahead)
```

Pattern per release:
- Develop on a `feature/<topic>` branch off `dev`.
- Atomic commits with "why" in the messages.
- When validated end-to-end manually → merge to `dev`, then `dev` → `main`.
- Tag `main` with the new semver (`v1.0.0` for Phase 1, `v1.1.0` for Phase 2
  math, etc.).
- Delete the feature branch after merge.

See `CHANGELOG.md` for what shipped per release.

---

## A note on filenames

The data artefacts were renamed on 2026-10-04: `flat.jsonl`,
`normalized.jsonl`, `payload.jsonl` and `chroma_db/`
dropped the suffix, which named a project phase that ended when maths
shipped in v1.1.0. `configs/phase1_scope.yaml` became `configs/scope.yaml`
in the same series.

---

## References

- `docs/data_audit.md` — what the raw corpus contains and what is wrong with it
  (MathJax/KaTeX rendering, auth, error handling)
- `CHANGELOG.md` — per-release shipped/known-issue breakdown
- `configs/scope.yaml` — declarative scope filter (subjects, levels, languages)
- `src/quiz_generator/curriculum/curriculum_rules.py` — Tunisian curriculum compliance rules
  (drops mistagged rows at normalize time)
- `notebooks/math_data_audit.ipynb` — Phase-2 discovery notebook
- `notebooks/level_categoris.ipynb` — level taxonomy exploration
- `eval/results/` — eval baselines per language / subject
