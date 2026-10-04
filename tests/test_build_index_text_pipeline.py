"""The search-text stage end to end.

`tests/test_build_index_text.py` covers `compose_search_text` and the recipe
loader. The stage function that drives them had no test, and it produces the
single string every embedding is computed from: get it wrong and retrieval is
wrong, with nothing in the metrics to say why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quiz_generator.ingestion.build_index_text import build_index_text

RECIPE = """
search_text:
  recipe: default
  recipes:
    default:
      include_subjects: true
      include_quiz_title: true
      include_question: true
      include_choices: true
      include_correct_answers: false
      include_levels: false
  separators:
    part: ". "
    subjects: ", "
    choices: " | "
  token_warning_threshold: 15
  normalize_latex: false
"""


def _normalized(**overrides: Any) -> dict[str, Any]:
    """One normalize output row — the shape this stage consumes."""
    row: dict[str, Any] = {
        "doc_id": "quiz1__q0",
        "quiz_id": "quiz1",
        "quiz_title": "Present Simple",
        "language": "en",
        "subjects": ["ENGLISH"],
        "levels": ["MIDDLE_SCHOOL_1ST_GRADE"],
        "school_phase": "MIDDLE",
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text": "She ____ to school every day.",
        "choices_text": ["goes", "go"],
        "correct_choices_text": ["goes"],
        "choices_media": [None, None],
        "points": 1.0,
        "time": 30,
        "author_name": None,
        "author_email": None,
    }
    row.update(overrides)
    return row


def _run(tmp_path: Path, rows: list[dict], recipe: str = RECIPE) -> tuple[Any, list[dict]]:
    src = tmp_path / "normalized.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    config = tmp_path / "pipeline.yaml"
    config.write_text(recipe, encoding="utf-8")
    out = tmp_path / "payload.jsonl"

    stats = build_index_text(
        input_path=src,
        output_path=out,
        stats_path=tmp_path / "payload_stats.json",
        config_path=config,
    )

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return stats, written


# ---------------------------------------------------------------------------
# What gets embedded
# ---------------------------------------------------------------------------


def test_the_search_text_is_composed_from_the_recipe(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_normalized()])

    assert stats.output_rows == 1
    assert rows[0]["search_text"] == (
        "ENGLISH. Present Simple. She ____ to school every day.. goes | go"
    )


def test_the_correct_answer_is_not_embedded_by_default(tmp_path: Path) -> None:
    """Including it would let retrieval match on the answer rather than the
    question — the model still receives it, through the payload."""
    _, rows = _run(tmp_path, [_normalized(correct_choices_text=["UNIQUEANSWERTOKEN"])])

    assert "UNIQUEANSWERTOKEN" not in rows[0]["search_text"]


def test_every_other_field_is_passed_through_untouched(tmp_path: Path) -> None:
    """This stage adds one field. A row that loses its levels or its doc_id
    here is unretrievable later."""
    source = _normalized()

    _, rows = _run(tmp_path, [source])

    for key, value in source.items():
        assert rows[0][key] == value


# ---------------------------------------------------------------------------
# Rows that do not survive
# ---------------------------------------------------------------------------


def test_a_row_with_nothing_to_embed_is_skipped_and_counted(tmp_path: Path) -> None:
    empty = _normalized(
        doc_id="quiz1__q1", quiz_title="", question_text="", choices_text=[], subjects=[]
    )

    stats, rows = _run(tmp_path, [_normalized(), empty])

    assert len(rows) == 1
    assert stats.empty_search_text_rows == 1


def test_a_row_rejected_by_the_output_schema_is_counted(tmp_path: Path) -> None:
    """It used to be dropped by a bare `except ValidationError: continue` with
    no counter anywhere — rows left the pipeline with no record at all, which
    CLAUDE.md Part II §7 forbids."""
    bad = _normalized(doc_id="quiz1__q1", points="not-a-number")

    stats, rows = _run(tmp_path, [_normalized(), bad])

    assert len(rows) == 1
    assert stats.schema_validation_failed == 1
    assert stats.output_rows == len(rows)


# ---------------------------------------------------------------------------
# The stats record
# ---------------------------------------------------------------------------


def test_the_stats_name_the_recipe_that_produced_the_file(tmp_path: Path) -> None:
    """The recipe decides what was embedded, so a run that does not record it
    cannot be accounted for later."""
    stats, _ = _run(tmp_path, [_normalized()])

    assert stats.recipe == "default"
    assert stats.recipe_flags["include_choices"] is True
    assert stats.recipe_flags["include_correct_answers"] is False


def test_rows_over_the_token_threshold_are_counted(tmp_path: Path) -> None:
    """The embedder's context is finite; long rows are silently truncated by
    it, so they are flagged here instead."""
    long_row = _normalized(question_text=" ".join(["word"] * 40))

    stats, _ = _run(tmp_path, [_normalized(), long_row])

    assert stats.token_threshold == 15
    assert stats.rows_over_token_threshold == 1


def test_the_stats_file_is_written_and_matches_the_returned_record(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_normalized()])
    written = json.loads((tmp_path / "payload_stats.json").read_text(encoding="utf-8"))

    assert written["output_rows"] == stats.output_rows == len(rows)
    assert written["recipe"] == "default"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_output_is_written_atomically(tmp_path: Path) -> None:
    _run(tmp_path, [_normalized()])

    assert not (tmp_path / "payload.jsonl.tmp").exists()
    assert not (tmp_path / "payload_stats.json.tmp").exists()


def test_re_running_produces_the_same_output(tmp_path: Path) -> None:
    first_stats, first_rows = _run(tmp_path, [_normalized()])
    second_stats, second_rows = _run(tmp_path, [_normalized()])

    assert first_rows == second_rows
    assert first_stats.model_dump() == second_stats.model_dump()


def test_blank_lines_in_the_input_are_skipped(tmp_path: Path) -> None:
    src = tmp_path / "normalized.jsonl"
    src.write_text(json.dumps(_normalized()) + "\n\n\n", encoding="utf-8")
    config = tmp_path / "pipeline.yaml"
    config.write_text(RECIPE, encoding="utf-8")

    stats = build_index_text(
        input_path=src,
        output_path=tmp_path / "payload.jsonl",
        stats_path=tmp_path / "stats.json",
        config_path=config,
    )

    assert stats.input_rows == 1
