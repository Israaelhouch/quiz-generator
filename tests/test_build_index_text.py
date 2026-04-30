"""Tests for Stage 2c — search_text composition."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_index_text import (
    DEFAULT_RECIPE_FLAGS,
    DEFAULT_SEPARATORS,
    _nonempty_strings,
    _summarize_lengths,
    _token_count,
    compose_search_text,
)


def _row(**overrides) -> dict:
    base = {
        "doc_id": "x",
        "quiz_title": "Immunity 1",
        "subjects": ["SCIENCE"],
        "levels": ["PRIMARY_SCHOOL_6TH_GRADE"],
        "question_text": "What is a pathogen?",
        "choices_text": ["any molecule", "used to combat infections", "none"],
        "correct_choices_text": ["any molecule"],
    }
    base.update(overrides)
    return base


def test_default_recipe_includes_subjects_title_question_choices() -> None:
    text = compose_search_text(
        _row(),
        flags=dict(DEFAULT_RECIPE_FLAGS),
        separators=dict(DEFAULT_SEPARATORS),
        normalize_latex_flag=False,
    )
    assert text == "SCIENCE. Immunity 1. What is a pathogen?. any molecule | used to combat infections | none"


def test_latex_normalization_when_enabled() -> None:
    """With the flag on, LaTeX in question_text is converted to plain math text."""
    row = _row(question_text=r"Calculer \(\sin(x) + \frac{1}{2}\)")
    text = compose_search_text(
        row,
        flags=dict(DEFAULT_RECIPE_FLAGS),
        separators=dict(DEFAULT_SEPARATORS),
        normalize_latex_flag=True,
    )
    assert "\\sin" not in text
    assert "\\frac" not in text
    assert "sin" in text  # function name preserved


def test_latex_normalization_off_keeps_raw() -> None:
    row = _row(question_text=r"Calculer \(\sin(x)\)")
    text = compose_search_text(
        row,
        flags=dict(DEFAULT_RECIPE_FLAGS),
        separators=dict(DEFAULT_SEPARATORS),
        normalize_latex_flag=False,
    )
    assert "\\sin" in text  # raw preserved when flag is off


def test_default_recipe_excludes_correct_answers() -> None:
    """Critical: correct_choices_text must NOT leak into search_text by default."""
    text = compose_search_text(
        _row(),
        flags=dict(DEFAULT_RECIPE_FLAGS),
        separators=dict(DEFAULT_SEPARATORS),
    )
    # 'any molecule' appears in choices_text (which IS included), but we verify
    # there's no duplication from the correct-answers section.
    assert text.count("any molecule") == 1


def test_include_correct_answers_flag_adds_them() -> None:
    flags = dict(DEFAULT_RECIPE_FLAGS)
    flags["include_correct_answers"] = True
    text = compose_search_text(
        _row(),
        flags=flags,
        separators=dict(DEFAULT_SEPARATORS),
    )
    # 'any molecule' now appears twice — once in choices, once as correct answer.
    assert text.count("any molecule") == 2


def test_missing_optional_fields_are_skipped() -> None:
    row = _row(subjects=[], quiz_title="", choices_text=[])
    text = compose_search_text(
        row,
        flags=dict(DEFAULT_RECIPE_FLAGS),
        separators=dict(DEFAULT_SEPARATORS),
    )
    assert text == "What is a pathogen?"


def test_nonempty_strings_filters_falsy_and_whitespace() -> None:
    assert _nonempty_strings(["A", "", "  ", None, "B"]) == ["A", "B"]
    assert _nonempty_strings(None) == []
    assert _nonempty_strings([]) == []


def test_include_levels_flag_adds_levels_section() -> None:
    flags = dict(DEFAULT_RECIPE_FLAGS)
    flags["include_levels"] = True
    text = compose_search_text(
        _row(),
        flags=flags,
        separators=dict(DEFAULT_SEPARATORS),
    )
    assert "PRIMARY_SCHOOL_6TH_GRADE" in text


def test_token_count_whitespace() -> None:
    assert _token_count("") == 0
    assert _token_count("one two three") == 3
    assert _token_count("  padded   spaces   ") == 2


def test_summarize_lengths_reports_five_number_summary() -> None:
    summary = _summarize_lengths([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert summary["min"] == 10
    assert summary["max"] == 100
    assert summary["mean"] == 55.0
    assert summary["median"] == 55.0
    assert summary["p95"] == 100


def test_summarize_lengths_empty() -> None:
    summary = _summarize_lengths([])
    assert summary["min"] == 0.0
    assert summary["max"] == 0.0


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All Stage 2c tests passed.")
