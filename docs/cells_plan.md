# Cells and Plan

**Status:** Active
**Current branch:** `feature/math-subject` (Phase 2 math)
**Last updated:** 2026-05-19

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
| C4  | `fr × MATHEMATICS`  | fr       | MATHEMATICS | ✅ Phase 2 (`feature/math-subject`) | High-school math in French (1,003 docs). Sibling-topic confusion at ~10pp below language cells. |
| C5  | `ar × MATHEMATICS`  | ar       | MATHEMATICS | ✅ Phase 2 (`feature/math-subject`) | Middle + primary math in Arabic (368 docs). |

### Out of scope

- **Higher-ed math** (PREPARATORY, LICENCE levels). Calculus notation
  (`\int`, `\sum`, `\lim`) is absent from the current scope's corpus;
  adding it requires LaTeX-rendering hardening and possibly SymPy
  verification. Queued for a later phase.
- **Physics, Chemistry, Sciences, Computer Science, History, Technique** —
  future scope. These reuse the same retrieval + generation stack as the
  current cells, so adding them is a data-and-eval task, not an
  infrastructure task. Each will need a curriculum rule in
  `src/data/curriculum_rules.py` if its (subject, phase) → language
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

Some data/config artifacts carry a `phase1` suffix for historical reasons:
`configs/phase1_scope.yaml`, `data/processed/ready_phase1.jsonl`,
`data/vector_store/chroma_db_phase1/`. These predate the descriptive-scope
naming and are kept as-is to avoid invasive renames across hardcoded paths.
The conceptual scope (what's locked / next / future) is defined in this
document; the filenames are just labels.

---

## References

- `docs/scope.md` — original project scope (locked)
- `docs/frontend_integration.md` — what the platform team needs to know
  (MathJax/KaTeX rendering, auth, error handling)
- `CHANGELOG.md` — per-release shipped/known-issue breakdown
- `configs/phase1_scope.yaml` — declarative scope filter (subjects + levels + languages)
- `configs/phase2_math_audit.yaml` — audit scope used during Phase-2 discovery
- `src/data/curriculum_rules.py` — Tunisian curriculum compliance rules
  (drops mistagged rows at normalize time)
- `notebooks/math_data_audit.ipynb` — Phase-2 discovery notebook
- `notebooks/level_categoris.ipynb` — level taxonomy exploration
- `eval/results/` — eval baselines per language / subject
