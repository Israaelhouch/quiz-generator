"""Taxonomy — the known set of valid values for each filter field.

Discovered at index time by scanning ready.jsonl, persisted inside
build_summary.json, and loaded at query time to:
  - validate user inputs (warn on typos like 'HIGH_SCHOOL_4TH_GRAD_MATH')
  - feed a frontend dropdown listing available levels/subjects
  - document what the vector store actually contains

Self-adapting: whatever the data contains IS the taxonomy. New levels
appear automatically on the next build; nothing to hand-maintain.

ONE exception to "whatever the data contains": levels can be prefix-filtered
via `level_prefixes`. A kept row may carry SECONDARY level tags that are
themselves out of scope — `src/data/scope.py::decide_in_scope` only checks
`levels[0]`, so a row tagged
`["HIGH_SCHOOL_1ST_GRADE", "LICENCE_1ST_GRADE"]` is (correctly) kept, but
without filtering the taxonomy would then advertise `LICENCE_1ST_GRADE` as
a legal dropdown value backed by a handful of documents. Callers that know
the scope pass `level_prefixes=SCHOOL_LEVEL_PREFIXES` to suppress those.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


# In-scope level prefixes for the Tunisian school curriculum.
#
# Mirrors two other places that must stay in sync:
#   - configs/phase1_scope.yaml → scope.level_prefixes (ingest filter)
#   - src/data/normalize.py::derive_school_phase (school_phase derivation)
#
# PREPARATORY_* and LICENCE_* are deliberately absent — higher education is
# out of scope (see docs/cells_plan.md). Add a prefix here when the
# corresponding phase enters scope.
SCHOOL_LEVEL_PREFIXES: tuple[str, ...] = (
    "PRIMARY_SCHOOL",
    "MIDDLE_SCHOOL",
    "HIGH_SCHOOL",
)


def _keep_level(value: str, prefixes: tuple[str, ...] | None) -> bool:
    """True when `value` should enter the taxonomy.

    `prefixes=None` disables filtering entirely (the historical behaviour,
    and what generic/test callers want).
    """
    if not prefixes:
        return True
    return value.startswith(prefixes)


@dataclass
class Taxonomy:
    """The enumerated values present in the indexed corpus."""

    languages: set[str] = field(default_factory=set)
    question_types: set[str] = field(default_factory=set)
    subjects: set[str] = field(default_factory=set)
    levels: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_rows(
        cls,
        rows: list[dict],
        *,
        level_prefixes: tuple[str, ...] | None = None,
    ) -> "Taxonomy":
        """Scan a corpus and collect every distinct enum value observed.

        Args:
            rows: the indexed corpus (ready_phase1.jsonl rows).
            level_prefixes: when given, only level values starting with one
                of these prefixes are collected. Pass
                `SCHOOL_LEVEL_PREFIXES` to keep out-of-scope secondary tags
                (PREPARATORY_*, LICENCE_*) out of the published taxonomy.
                None (default) collects every level, unfiltered.
        """
        languages: set[str] = set()
        question_types: set[str] = set()
        subjects: set[str] = set()
        levels: set[str] = set()
        for row in rows:
            lang = str(row.get("language") or "").strip()
            if lang:
                languages.add(lang)
            qt = str(row.get("question_type") or "").strip()
            if qt:
                question_types.add(qt)
            for s in row.get("subjects") or []:
                if s is None:
                    continue
                val = str(s).strip()
                if val:
                    subjects.add(val)
            for lvl in row.get("levels") or []:
                if lvl is None:
                    continue
                val = str(lvl).strip()
                if val and _keep_level(val, level_prefixes):
                    levels.add(val)
        return cls(
            languages=languages,
            question_types=question_types,
            subjects=subjects,
            levels=levels,
        )

    @classmethod
    def from_build_summary(
        cls,
        summary_path: Path,
        *,
        level_prefixes: tuple[str, ...] | None = None,
    ) -> "Taxonomy":
        """Load from build_summary.json. Returns an empty taxonomy if file missing.

        `level_prefixes` is applied on load as well as at build time, so an
        index built before the filter existed is corrected at read time —
        no reindex needed to stop advertising out-of-scope levels.
        """
        if not summary_path.exists():
            return cls()
        with summary_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f) or {}
            except json.JSONDecodeError:
                return cls()
        tax = data.get("taxonomy") or {}
        raw_levels = [str(lvl).strip() for lvl in (tax.get("levels") or [])]
        return cls(
            languages=set(tax.get("languages") or []),
            question_types=set(tax.get("question_types") or []),
            subjects=set(tax.get("subjects") or []),
            levels={lvl for lvl in raw_levels if lvl and _keep_level(lvl, level_prefixes)},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, list[str]]:
        """Sorted lists, suitable for embedding in JSON."""
        return {
            "languages": sorted(self.languages),
            "question_types": sorted(self.question_types),
            "subjects": sorted(self.subjects),
            "levels": sorted(self.levels),
        }

    # ------------------------------------------------------------------
    # Query helpers (frontend + retriever will call these)
    # ------------------------------------------------------------------

    def list_levels(self) -> list[str]:
        return sorted(self.levels)

    def list_subjects(self) -> list[str]:
        return sorted(self.subjects)

    def list_languages(self) -> list[str]:
        return sorted(self.languages)

    def list_question_types(self) -> list[str]:
        return sorted(self.question_types)

    def is_empty(self) -> bool:
        return not (self.languages or self.question_types or self.subjects or self.levels)

    # ------------------------------------------------------------------
    # Validation (soft — logged, not raised)
    # ------------------------------------------------------------------

    def validate_language(self, language: str | None) -> bool:
        """Warn on unknown language. Return True if known or empty-taxonomy."""
        if not language or not self.languages:
            return True
        if language not in self.languages:
            logger.warning(
                "language=%r not in known taxonomy. Known: %s",
                language, sorted(self.languages),
            )
            return False
        return True

    def validate_question_type(self, question_type: str | None) -> bool:
        if not question_type or not self.question_types:
            return True
        if question_type not in self.question_types:
            logger.warning(
                "question_type=%r not in known taxonomy. Known: %s",
                question_type, sorted(self.question_types),
            )
            return False
        return True

    def validate_subject(self, subject: str | None) -> bool:
        if not subject or not self.subjects:
            return True
        if subject not in self.subjects:
            logger.warning(
                "subject=%r not in known taxonomy. Known subjects: %s",
                subject, sorted(self.subjects),
            )
            return False
        return True

    def validate_level(self, level: str) -> bool:
        if not self.levels:
            return True
        if level not in self.levels:
            # Offer a small hint of similar names (first 5 lexicographically close)
            hint = sorted(self.levels)[:5]
            logger.warning(
                "level=%r not in known taxonomy. Example known levels: %s",
                level, hint,
            )
            return False
        return True

    def validate_levels(self, levels: list[str] | None) -> list[str]:
        """Validate a list of level names; warn per unknown. Returns input unchanged."""
        if not levels:
            return []
        for lvl in levels:
            self.validate_level(lvl)
        return levels
