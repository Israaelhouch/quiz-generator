"""Tests for Stage 2c — search_text composition."""

from __future__ import annotations

import sys

from quiz_generator.ingestion.build_index_text import (
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
    assert (
        text
        == "SCIENCE. Immunity 1. What is a pathogen?. any molecule | used to combat infections | none"
    )


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


# ---------------------------------------------------------------------------
# Recipe config — the search_text recipe decides what text gets embedded, so
# a config this loader accepts quietly is an index nobody can account for.
# ---------------------------------------------------------------------------


def test_the_repository_pipeline_config_loads() -> None:
    from pathlib import Path as _Path

    from quiz_generator.ingestion.build_index_text import load_recipe

    name, flags, separators, threshold, latex = load_recipe(_Path("configs/pipeline.yaml"))

    assert name == "default"
    assert flags == DEFAULT_RECIPE_FLAGS
    assert separators == DEFAULT_SEPARATORS
    assert threshold == 100
    assert latex is True


def test_a_missing_pipeline_config_is_an_error(tmp_path) -> None:
    """It used to return the defaults, so an index could be built from a
    config that was never read and the run would look successful."""
    import pytest

    from quiz_generator.ingestion.build_index_text import load_recipe

    with pytest.raises(FileNotFoundError):
        load_recipe(tmp_path / "absent.yaml")


def test_a_recipe_naming_an_unknown_flag_is_refused(tmp_path) -> None:
    """`include_choice` (singular) used to be dropped in silence: the index
    would be built without answer choices while the config said otherwise."""
    import pytest
    from pydantic import ValidationError

    from quiz_generator.ingestion.build_index_text import load_recipe

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "search_text:\n  recipe: default\n  recipes:\n    default:\n      include_choice: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_recipe(config)


def test_selecting_an_undefined_recipe_is_refused(tmp_path) -> None:
    """It used to fall back to the defaults — so an A/B test between two
    recipes could run the control twice and report it as a comparison."""
    import pytest

    from quiz_generator.ingestion.build_index_text import load_recipe

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "search_text:\n  recipe: title_only\n  recipes:\n    default:\n"
        "      include_choices: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="title_only"):
        load_recipe(config)


def test_a_nonsensical_token_threshold_is_refused(tmp_path) -> None:
    import pytest
    from pydantic import ValidationError

    from quiz_generator.ingestion.build_index_text import load_recipe

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "search_text:\n  recipe: default\n  recipes:\n    default: {}\n"
        "  token_warning_threshold: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_recipe(config)
