"""The normalize stage end to end — the orchestration, not the row helpers.

`tests/test_normalize.py` covers `normalize_row`, `dedup_rows` and the text
helpers. The stage function that drives them had no test, and it is the one
that removes 869 of 6,651 rows: cleaning, the curriculum rules, deduplication
and the stats record all meet here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quiz_generator.ingestion.normalize import normalize


def _choice(answer: str, is_true: bool = False) -> dict[str, Any]:
    return {"answer": answer, "isTrue": is_true, "media": None}


def _flat(**overrides: Any) -> dict[str, Any]:
    """One ingest output row — the shape normalize consumes."""
    row: dict[str, Any] = {
        "doc_id": "quiz1__q0",
        "quiz_id": "quiz1",
        "quiz_title_raw": "Quiz: Present Simple",
        "language_raw": "english",
        "subjects": ["ENGLISH"],
        "levels": ["MIDDLE_SCHOOL_1ST_GRADE"],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": "<p>She ____ to school every day.</p>",
        "choices_raw": [_choice("goes", True), _choice("go")],
        "points": 1.0,
        "time": 30,
        "author_name": "A Teacher",
        "author_email": "teacher@example.test",
    }
    row.update(overrides)
    return row


def _aliases(tmp_path: Path, mapping: str = "MECHANIC: PHYSICS\n") -> Path:
    path = tmp_path / "aliases.yaml"
    path.write_text(mapping, encoding="utf-8")
    return path


def _run(tmp_path: Path, rows: list[dict]) -> tuple[Any, list[dict]]:
    flat = tmp_path / "flat.jsonl"
    flat.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "normalized.jsonl"

    stats = normalize(
        input_path=flat,
        output_path=out,
        stats_path=tmp_path / "normalized_stats.json",
        aliases_path=_aliases(tmp_path),
    )

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return stats, written


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_text_is_stripped_of_markup_and_the_quiz_prefix(tmp_path: Path) -> None:
    """99% of source descriptions carry HTML, and titles are prefixed 'Quiz:'."""
    stats, rows = _run(tmp_path, [_flat()])

    assert stats.output_rows == 1
    assert rows[0]["question_text"] == "She ____ to school every day."
    assert rows[0]["quiz_title"] == "Present Simple"


def test_the_school_phase_is_derived_from_the_levels(tmp_path: Path) -> None:
    """Retrieval filters on phase, which the source only encodes inside level
    names like MIDDLE_SCHOOL_1ST_GRADE."""
    _, rows = _run(tmp_path, [_flat()])

    assert rows[0]["school_phase"] == "MIDDLE"


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


def test_a_subject_locked_language_overrides_the_source_label(tmp_path: Path) -> None:
    """~15% of rows carry a wrong language tag. For a language subject the
    subject is the stronger signal: ARABIC content is not English."""
    row = _flat(
        subjects=["ARABIC"],
        language_raw="english",
        quiz_title_raw="الجملة الفعلية",
        question_text_raw="اختر الفعل الصحيح في الجملة التالية",
        choices_raw=[_choice("ذهب", True), _choice("يذهب")],
    )

    stats, rows = _run(tmp_path, [row])

    assert rows[0]["language"] == "ar"
    assert stats.language_corrections == {"english->ar": 1}


def test_an_unsupported_language_survives_as_french_a_known_limitation(
    tmp_path: Path,
) -> None:
    """The detector reads a short Spanish sentence as French, so the row is
    relabelled rather than dropped. It matters because the French cell is 15
    questions: a handful of misrouted rows would be a visible fraction of it.

    Not currently reachable — the corpus has one Spanish quiz and its subject
    is out of scope, so it never gets this far. This test exists so the
    behaviour is recorded rather than discovered later."""
    row = _flat(
        subjects=["SPANISH"],
        language_raw="spanish",
        quiz_title_raw="El presente",
        question_text_raw="Elige el verbo correcto para completar la frase",
        choices_raw=[_choice("va", True), _choice("ir")],
    )

    _, rows = _run(tmp_path, [row])

    assert rows[0]["language"] == "fr"


def test_a_row_rejected_by_the_output_schema_is_not_counted_as_written(
    tmp_path: Path,
) -> None:
    """`output_rows` came from the deduplicated list, not from the rows that
    survived validation on the way out, so the stats could claim a row the
    file does not contain."""
    second = _flat(
        doc_id="quiz1__q1",
        question_text_raw="<p>They ____ football every Sunday afternoon.</p>",
        points="not-a-number",
    )

    stats, rows = _run(tmp_path, [_flat(), second])

    assert stats.dropped.get("schema_validation_failed") == 1
    assert stats.output_rows == len(rows)


# ---------------------------------------------------------------------------
# Dropping
# ---------------------------------------------------------------------------


def test_a_question_that_is_only_an_image_is_dropped(tmp_path: Path) -> None:
    row = _flat(question_text_raw='<p><img src="q.png"></p>')

    stats, rows = _run(tmp_path, [row])

    assert rows == []
    assert stats.dropped.get("description_is_image_only") == 1


def test_a_question_with_no_visible_text_is_dropped(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_flat(question_text_raw="<p>  </p>")])

    assert rows == []
    assert stats.dropped.get("empty_description") == 1


def test_a_row_whose_choices_are_all_blank_is_dropped(tmp_path: Path) -> None:
    """Placeholder rows: the source has choices=['','',''], which ingest's
    empty-list check cannot see."""
    stats, rows = _run(tmp_path, [_flat(choices_raw=[_choice(""), _choice("   ")])])

    assert rows == []
    assert stats.dropped.get("all_choices_empty") == 1


def test_a_curriculum_violation_is_dropped(tmp_path: Path) -> None:
    """Tunisian primary maths is taught in Arabic, so a French primary maths
    row is a source mistag rather than content."""
    row = _flat(
        subjects=["MATHEMATICS"],
        levels=["PRIMARY_SCHOOL_5TH_GRADE"],
        language_raw="french",
        quiz_title_raw="Les fractions",
        question_text_raw="Choisissez la fraction la plus grande parmi les suivantes",
        choices_raw=[_choice("trois quarts", True), _choice("un tiers")],
    )

    stats, rows = _run(tmp_path, [row])

    assert rows == []
    assert any("curriculum" in reason for reason in stats.dropped)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_identical_questions_collapse_into_one_row(tmp_path: Path) -> None:
    first = _flat(doc_id="quiz1__q0")
    second = _flat(doc_id="quiz2__q0", quiz_id="quiz2")

    stats, rows = _run(tmp_path, [first, second])

    assert len(rows) == 1
    assert stats.duplicate_groups == 1
    assert stats.duplicate_rows_dropped == 1
    assert stats.dropped.get("duplicate_question") == 1


def test_deduplication_unions_the_levels_of_the_rows_it_merges(tmp_path: Path) -> None:
    """The same question used at two levels is reachable from either, so the
    surviving row has to carry both — otherwise a level filter loses it."""
    first = _flat(levels=["MIDDLE_SCHOOL_1ST_GRADE"])
    second = _flat(doc_id="quiz2__q0", quiz_id="quiz2", levels=["MIDDLE_SCHOOL_2ND_GRADE"])

    _, rows = _run(tmp_path, [first, second])

    assert set(rows[0]["levels"]) == {"MIDDLE_SCHOOL_1ST_GRADE", "MIDDLE_SCHOOL_2ND_GRADE"}


def test_questions_differing_only_in_their_choices_are_kept_apart(tmp_path: Path) -> None:
    first = _flat()
    second = _flat(
        doc_id="quiz2__q0",
        quiz_id="quiz2",
        choices_raw=[_choice("goes", True), _choice("is going")],
    )

    _, rows = _run(tmp_path, [first, second])

    assert len(rows) == 2


# ---------------------------------------------------------------------------
# The stats record must describe the file that was written
# ---------------------------------------------------------------------------


def test_output_rows_matches_the_rows_actually_written(tmp_path: Path) -> None:
    """`output_rows` is the number every downstream count is checked against;
    if a row is rejected on the way out, the stats must not still claim it."""
    stats, rows = _run(tmp_path, [_flat(), _flat(doc_id="quiz1__q1", question_text_raw="")])

    assert stats.output_rows == len(rows)


def test_the_stats_file_is_written_and_matches_the_returned_record(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_flat()])
    written = json.loads((tmp_path / "normalized_stats.json").read_text(encoding="utf-8"))

    assert written["output_rows"] == stats.output_rows == len(rows)
    assert written["by_language"] == {"en": 1}
    assert written["by_type"] == {"MULTIPLE_CHOICE": 1}


def test_re_running_produces_the_same_output(tmp_path: Path) -> None:
    first_stats, first_rows = _run(tmp_path, [_flat()])
    second_stats, second_rows = _run(tmp_path, [_flat()])

    assert first_rows == second_rows
    assert first_stats.model_dump() == second_stats.model_dump()


def test_blank_lines_in_the_input_are_skipped(tmp_path: Path) -> None:
    flat = tmp_path / "flat.jsonl"
    flat.write_text(json.dumps(_flat()) + "\n\n\n", encoding="utf-8")

    stats = normalize(
        input_path=flat,
        output_path=tmp_path / "normalized.jsonl",
        stats_path=tmp_path / "stats.json",
        aliases_path=_aliases(tmp_path),
    )

    assert stats.input_rows == 1


def test_the_output_is_written_atomically(tmp_path: Path) -> None:
    _run(tmp_path, [_flat()])

    assert not (tmp_path / "normalized.jsonl.tmp").exists()
    assert not (tmp_path / "normalized_stats.json.tmp").exists()
