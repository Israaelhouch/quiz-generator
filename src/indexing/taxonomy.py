"""Taxonomy — the known set of valid values for each filter field.

Discovered at index time by scanning ready.jsonl, persisted inside
build_summary.json, and loaded at query time to:
  - validate user inputs (warn on typos like 'HIGH_SCHOOL_4TH_GRAD_MATH')
  - feed a frontend dropdown listing available levels/subjects
  - document what the vector store actually contains

Self-adapting: whatever the data contains IS the taxonomy. New levels
appear automatically on the next build; nothing to hand-maintain.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path


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
    def from_rows(cls, rows: list[dict]) -> "Taxonomy":
        """Scan a corpus and collect every distinct enum value observed."""
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
                if val:
                    levels.add(val)
        return cls(
            languages=languages,
            question_types=question_types,
            subjects=subjects,
            levels=levels,
        )

    @classmethod
    def from_build_summary(cls, summary_path: Path) -> "Taxonomy":
        """Load from build_summary.json. Returns an empty taxonomy if file missing."""
        if not summary_path.exists():
            return cls()
        with summary_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f) or {}
            except json.JSONDecodeError:
                return cls()
        tax = data.get("taxonomy") or {}
        return cls(
            languages=set(tax.get("languages") or []),
            question_types=set(tax.get("question_types") or []),
            subjects=set(tax.get("subjects") or []),
            levels=set(tax.get("levels") or []),
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
    # Validation (soft — warnings, not errors)
    # ------------------------------------------------------------------

    def validate_language(self, language: str | None) -> bool:
        """Warn on unknown language. Return True if known or empty-taxonomy."""
        if not language or not self.languages:
            return True
        if language not in self.languages:
            warnings.warn(
                f"language={language!r} not in known taxonomy. "
                f"Known: {sorted(self.languages)}",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    def validate_question_type(self, question_type: str | None) -> bool:
        if not question_type or not self.question_types:
            return True
        if question_type not in self.question_types:
            warnings.warn(
                f"question_type={question_type!r} not in known taxonomy. "
                f"Known: {sorted(self.question_types)}",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    def validate_subject(self, subject: str | None) -> bool:
        if not subject or not self.subjects:
            return True
        if subject not in self.subjects:
            warnings.warn(
                f"subject={subject!r} not in known taxonomy. "
                f"Known subjects: {sorted(self.subjects)}",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    def validate_level(self, level: str) -> bool:
        if not self.levels:
            return True
        if level not in self.levels:
            # Offer a small hint of similar names (first 5 lexicographically close)
            hint = sorted(self.levels)[:5]
            warnings.warn(
                f"level={level!r} not in known taxonomy. Example known levels: {hint}",
                UserWarning,
                stacklevel=2,
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
