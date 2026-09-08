# Decisions

Two records. Both exist because a future reader — including future you —
would otherwise have no way to tell a deliberate choice from an accident, and
in one case would not know about a security exposure that is dormant today
but becomes real if the deployment shape changes.

Format: **context → options → decision → consequences** (CLAUDE.md §2).

> Earlier decisions (chunking strategy, embedder choice, reranker pool size,
> the distance floor) are not written up here. They are reconstructable from
> `configs/models.yaml` and `eval/RESULTS.md`.

---

## ADR-0001 — Lint and type debt is baselined, not swept

**Status:** accepted · 2026-09-08

**Context.** Turning on ruff and `mypy --strict` against the existing codebase
produced 168 lint violations and 93 type errors across 23 files. CI cannot be
introduced red.

**Options.**
1. Fix all 261 findings in one change.
2. Weaken the tools until the codebase passes.
3. Baseline what exists, gate on what is new, shrink the baseline over time.

**Decision.** Option 3, except for ruff's 62 *safe* autofixes — import
sorting, dead imports, `datetime.UTC` — which were applied in their own
commit, verified by the suite staying at 267 passed.

**Consequences.** `make lint` is green and genuinely gates new code. The
allow-lists in `pyproject.toml` are counted and annotated, and only shrink.

**Three baselined entries are findings, not cosmetics.** Each needs its own
change with its own test, and none belonged in a config-only commit:

- **`B905` — `zip()` without `strict=`** in `retrieval/retriever.py`,
  `retrieval/reranker.py`, `indexing/build.py`, `indexing/query.py`. Today a
  length mismatch between, say, documents and their scores **truncates
  silently**. `strict=True` would raise instead. That is probably the correct
  behaviour and possibly a latent bug.
- **`unreachable`** at `data/curriculum_rules.py:79` and
  `data/domain_rules.py:70`. Both loop over a parameter annotated `list[str]`
  and then guard `if subj is None: continue`. Either the guard is dead code or
  the annotation is wrong. Which one depends on whether the ingest corpus
  really contains null subjects — a data question, not a typing question.
- **`arg-type`** at `pipeline/orchestrator.py:196,198`. A plain `str` is passed
  where `Literal['en','fr','ar']` and `Literal['MULTIPLE_CHOICE',
  'FILL_IN_THE_BLANKS']` are declared. Pydantic catches it at runtime, so it
  raises rather than corrupts — but it raises deep in generation instead of at
  the API edge where the bad value arrived.

Also baselined: `F821` at `indexing/config.py:30`, where `load_models_config`
is annotated `-> "ModelsConfig"` but `ModelsConfig` is defined *inside* the
function body. `from __future__ import annotations` means it is never
evaluated, so nothing raises — but the annotation is unresolvable, so
`typing.get_type_hints()` fails on it and no checker can use it.

---

## ADR-0002 — Four chromadb advisories are accepted, conditionally

**Status:** accepted · 2026-09-08 · **review 2026-12-01**

> **Read this before changing how Chroma is deployed.** The exemption below
> depends entirely on this project running Chroma as a local embedded store.
> If that ever changes, these four ignores must be removed first.

**Context.** `pip-audit` reports four vulnerabilities in `chromadb==1.5.9`:

| ID | Issue |
|---|---|
| PYSEC-2026-311 | pre-authentication code injection |
| CVE-2026-45830 | missing authorization validation |
| CVE-2026-45831 | `SimpleRBACAuthorizationProvider` flaw |
| CVE-2026-45833 | authenticated code injection |

None has a fix version, and 1.5.9 is the latest release — there is nothing to
upgrade to.

**Options.**
1. Fail CI until upstream ships a fix (red for an unknown duration).
2. Drop `pip-audit` from the gate.
3. Ignore these four IDs specifically, with justification and a review date.

**Decision.** Option 3.

**Evidence.** All four live in Chroma's *server*: its authentication, its
authorization providers, its request handling. This project never runs that
server. The only client construction in the codebase is
`chromadb.PersistentClient(path=...)` at `src/indexing/vector_store.py:121` —
an embedded, on-disk store. There is no `HttpClient` anywhere in `src/`,
`scripts/` or `docker-compose.yml`, no Chroma port is exposed, and no auth
provider is configured. The vulnerable code paths are unreachable.

**Consequences.** `make audit` and the CI security job stay green and keep
catching *new* advisories, which is the point of the gate. The exemption is
narrow — four specific IDs, not the package and not the tool — and carries a
review date.

**The condition, stated plainly:** switching Chroma to client/server mode, or
exposing its port, turns all four of these into live exposures in a service
whose CI will still be green. The ignores are in `Makefile` (the `audit`
target) and `.github/workflows/ci.yml`. Delete them there, in the same change
that moves Chroma off `PersistentClient`.
