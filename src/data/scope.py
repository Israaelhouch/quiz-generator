"""Scope filtering — narrow the corpus to a defined scope (current or future).

Reads a YAML config (e.g. `configs/phase1_scope.yaml`) and exposes a single
function `decide_in_scope(row, scope) -> tuple[bool, str]`:

    in_scope, reason = decide_in_scope(flat_row_dict, scope_obj)
    if not in_scope:
        # row is dropped with `reason` recorded in stats

The reason string is one of:
    - "no_subjects"
    - "subject_out_of_scope"
    - "level_out_of_scope"
    - "" (empty when in_scope=True)

Note: language is NOT checked at this stage. At ingest time the row only
has `language_raw` (the source label, possibly empty), and language
detection happens later in normalize.py. The `languages` list in the
scope config is enforced indirectly: normalize.py already drops rows
whose resolved language isn't in {en, fr, ar}.

Pure-dict — no Pydantic dependency, easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScopeConfig:
    """Parsed scope filter rules."""
    name: str
    subjects: frozenset[str]
    level_prefixes: tuple[str, ...]
    languages: frozenset[str]


def load_scope(config_path: Path) -> ScopeConfig:
    """Load + validate a scope YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Scope config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    scope_block = raw.get("scope") or {}

    name = scope_block.get("name", "unnamed_scope")
    subjects = scope_block.get("subjects") or []
    level_prefixes = scope_block.get("level_prefixes") or []
    languages = scope_block.get("languages") or []

    if not subjects:
        raise ValueError(f"Scope {name!r}: at least one subject is required")
    if not level_prefixes:
        raise ValueError(f"Scope {name!r}: at least one level_prefix is required")
    if not languages:
        raise ValueError(f"Scope {name!r}: at least one language is required")

    return ScopeConfig(
        name=str(name),
        subjects=frozenset(s.upper() for s in subjects),
        level_prefixes=tuple(level_prefixes),
        languages=frozenset(l.lower() for l in languages),
    )


def decide_in_scope(row: dict[str, Any], scope: ScopeConfig) -> tuple[bool, str]:
    """Apply the scope filter to a single flattened row.

    Returns (in_scope, reason). reason is "" when in_scope=True, otherwise
    one of the documented drop reasons.
    """
    # 1. Must have at least one subject (Q2 = drop rows with no subject)
    subjects = row.get("subjects") or []
    if not subjects:
        return False, "no_subjects"

    # 2. ANY subject must be in scope (Q1 = permissive match)
    subjects_upper = [str(s).upper() for s in subjects]
    if not any(s in scope.subjects for s in subjects_upper):
        return False, "subject_out_of_scope"

    # 3. First level must match one of the configured prefixes
    levels = row.get("levels") or []
    if not levels:
        return False, "level_out_of_scope"
    first_level = str(levels[0])
    if not any(first_level.startswith(p) for p in scope.level_prefixes):
        return False, "level_out_of_scope"

    # Language is NOT checked here — at ingest the row only has
    # `language_raw` (source label), not the resolved `language`.
    # normalize.py drops rows whose resolved language isn't in
    # {en, fr, ar}, which enforces the scope's language constraint.
    return True, ""
