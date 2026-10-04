"""The scope filter decides which rows enter the corpus at all.

It had no tests. A mistake here does not crash anything: the pipeline runs,
the index builds, the eval scores — on a corpus missing a cell. These tests
pin both halves: the config loader refusing a malformed scope, and the filter
itself keeping and dropping the rows it is supposed to.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quiz_generator.ingestion.scope import ScopeConfig, decide_in_scope, load_scope

_VALID = """
scope:
  name: test_scope
  subjects: [ENGLISH, MATHEMATICS]
  level_prefixes: [PRIMARY_SCHOOL, HIGH_SCHOOL]
  languages: [en, fr]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scope.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_the_repository_scope_config_loads() -> None:
    """configs/scope.yaml is what every pipeline run is filtered through."""
    scope = load_scope(Path("configs/scope.yaml"))

    assert scope.name == "current"
    assert scope.subjects == frozenset({"ENGLISH", "ARABIC", "FRENCH"})
    assert scope.level_prefixes == ("PRIMARY_SCHOOL", "MIDDLE_SCHOOL", "HIGH_SCHOOL")
    assert scope.languages == frozenset({"en", "fr", "ar"})


def test_mathematics_is_out_of_scope() -> None:
    """Maths left the scope on 2026-10-04 (ADR-0009). Re-adding it is a corpus
    change that invalidates every published metric, so it should not happen by
    accident — this test is the tripwire."""
    scope = load_scope(Path("configs/scope.yaml"))

    assert "MATHEMATICS" not in scope.subjects
    assert decide_in_scope({"subjects": ["MATHEMATICS"], "levels": ["HIGH_SCHOOL_2"]}, scope) == (
        False,
        "subject_out_of_scope",
    )


def test_subjects_are_upper_cased_and_languages_lower_cased(tmp_path: Path) -> None:
    """Rows carry 'ENGLISH' and 'en'; the config may be written either way."""
    scope = load_scope(
        _write(
            tmp_path,
            "scope:\n  name: mixed\n  subjects: [english]\n"
            "  level_prefixes: [PRIMARY_SCHOOL]\n  languages: [EN]\n",
        )
    )

    assert scope.subjects == frozenset({"ENGLISH"})
    assert scope.languages == frozenset({"en"})


def test_a_missing_scope_config_is_an_error(tmp_path: Path) -> None:
    """Defaulting would filter the corpus by rules nobody wrote down."""
    with pytest.raises(FileNotFoundError):
        load_scope(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("", "empty file"),
        ("- ENGLISH\n- MATHEMATICS\n", "a list where a mapping belongs"),
    ],
)
def test_a_scope_config_that_is_not_a_mapping_is_an_error(
    tmp_path: Path, text: str, why: str
) -> None:
    with pytest.raises(ValueError):
        load_scope(_write(tmp_path, text))


def test_a_subject_list_written_as_a_bare_string_is_refused(tmp_path: Path) -> None:
    """`subjects: ENGLISH` used to become frozenset({'E','N','G','L','I','S','H'}),
    so every row fell out of scope and the corpus came out empty."""
    with pytest.raises(ValidationError):
        load_scope(
            _write(
                tmp_path,
                "scope:\n  name: broken\n  subjects: ENGLISH\n"
                "  level_prefixes: [PRIMARY_SCHOOL]\n  languages: [en]\n",
            )
        )


@pytest.mark.parametrize("key", ["subject", "levels_prefixes", "language"])
def test_a_mistyped_key_is_refused_rather_than_ignored(tmp_path: Path, key: str) -> None:
    """Near-miss spellings of the four real keys."""
    text = (
        f"scope:\n  name: typo\n  {key}: [ENGLISH]\n"
        "  subjects: [ENGLISH]\n  level_prefixes: [PRIMARY_SCHOOL]\n  languages: [en]\n"
    )
    with pytest.raises(ValidationError):
        load_scope(_write(tmp_path, text))


@pytest.mark.parametrize("field", ["subjects", "level_prefixes", "languages"])
def test_an_empty_list_is_refused(tmp_path: Path, field: str) -> None:
    """An empty subjects list would drop every row; an empty level_prefixes
    list would drop every row; neither is a scope anybody means to express."""
    lines = {
        "subjects": "[ENGLISH]",
        "level_prefixes": "[PRIMARY_SCHOOL]",
        "languages": "[en]",
    }
    lines[field] = "[]"
    text = "scope:\n  name: empty\n" + "".join(
        f"  {name}: {value}\n" for name, value in lines.items()
    )
    with pytest.raises(ValidationError):
        load_scope(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def scope(tmp_path: Path) -> ScopeConfig:
    return load_scope(_write(tmp_path, _VALID))


def test_a_row_in_scope_is_kept_with_no_reason(scope: ScopeConfig) -> None:
    in_scope, reason = decide_in_scope(
        {"subjects": ["ENGLISH"], "levels": ["HIGH_SCHOOL_2"]}, scope
    )

    assert (in_scope, reason) == (True, "")


def test_a_row_without_subjects_is_dropped(scope: ScopeConfig) -> None:
    """7.2% of the raw corpus carries no subject at all — see docs/data_audit.md."""
    assert decide_in_scope({"subjects": [], "levels": ["HIGH_SCHOOL_2"]}, scope) == (
        False,
        "no_subjects",
    )
    assert decide_in_scope({"levels": ["HIGH_SCHOOL_2"]}, scope) == (False, "no_subjects")


def test_a_subject_outside_the_scope_is_dropped(scope: ScopeConfig) -> None:
    assert decide_in_scope({"subjects": ["PHYSICS"], "levels": ["HIGH_SCHOOL_2"]}, scope) == (
        False,
        "subject_out_of_scope",
    )


def test_any_in_scope_subject_keeps_the_row(scope: ScopeConfig) -> None:
    """Permissive match: a row tagged PHYSICS *and* MATHEMATICS stays, because
    dropping it would lose maths content over a secondary tag."""
    in_scope, _ = decide_in_scope(
        {"subjects": ["PHYSICS", "MATHEMATICS"], "levels": ["PRIMARY_SCHOOL_4"]}, scope
    )

    assert in_scope is True


def test_subject_matching_ignores_case(scope: ScopeConfig) -> None:
    in_scope, _ = decide_in_scope({"subjects": ["english"], "levels": ["HIGH_SCHOOL_1"]}, scope)

    assert in_scope is True


def test_a_row_without_levels_is_dropped(scope: ScopeConfig) -> None:
    assert decide_in_scope({"subjects": ["ENGLISH"], "levels": []}, scope) == (
        False,
        "level_out_of_scope",
    )


def test_a_level_outside_the_prefixes_is_dropped(scope: ScopeConfig) -> None:
    assert decide_in_scope({"subjects": ["ENGLISH"], "levels": ["LICENCE_1"]}, scope) == (
        False,
        "level_out_of_scope",
    )


def test_only_the_first_level_is_inspected(scope: ScopeConfig) -> None:
    """Documented limitation, not an accident: a kept row can still carry
    out-of-scope tags after the first one. That is what made /taxonomy
    advertise 12 phantom levels until Taxonomy learned to filter on load
    (see CHANGELOG, Unreleased)."""
    kept, _ = decide_in_scope(
        {"subjects": ["ENGLISH"], "levels": ["HIGH_SCHOOL_3", "LICENCE_1"]}, scope
    )
    dropped, reason = decide_in_scope(
        {"subjects": ["ENGLISH"], "levels": ["LICENCE_1", "HIGH_SCHOOL_3"]}, scope
    )

    assert kept is True
    assert (dropped, reason) == (False, "level_out_of_scope")
