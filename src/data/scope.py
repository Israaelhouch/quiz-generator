"""Scope filtering — narrow the corpus to a defined scope (current or future).

Reads a YAML config (e.g. `configs/scope.yaml`) and exposes a single
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

The config is validated on load: unknown keys are rejected, and a value of
the wrong shape (`subjects: ENGLISH` instead of a list) is an error rather
than a frozenset of single characters that silently empties the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.shared.yaml_config import read_yaml_mapping


@dataclass(frozen=True)
class ScopeConfig:
    """Parsed scope filter rules."""
    name: str
    subjects: frozenset[str]
    level_prefixes: tuple[str, ...]
    languages: frozenset[str]


class _ScopeBlock(BaseModel):
    """The `scope:` block of a scope config. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed_scope"
    subjects: list[str] = Field(min_length=1)
    level_prefixes: list[str] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)


class _ScopeFile(BaseModel):
    """A scope config file: one `scope:` block and nothing else."""

    model_config = ConfigDict(extra="forbid")

    scope: _ScopeBlock


def load_scope(config_path: Path) -> ScopeConfig:
    """Load and validate a scope YAML file.

    Raises FileNotFoundError if the file is absent, and ValidationError if a
    key is unknown, a list is empty, or a value has the wrong shape.
    """
    raw = read_yaml_mapping(config_path, what="Scope")
    block = _ScopeFile.model_validate(raw).scope

    return ScopeConfig(
        name=block.name,
        subjects=frozenset(subject.upper() for subject in block.subjects),
        level_prefixes=tuple(block.level_prefixes),
        languages=frozenset(language.lower() for language in block.languages),
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
