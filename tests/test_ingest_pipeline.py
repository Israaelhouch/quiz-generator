"""The ingest stage end to end — the orchestration, not the filter helpers.

`tests/test_ingest.py` covers the pure functions in `filters.py`. Everything
between them was untested: the flattening, the per-quiz `order` counter, the
scope integration, the stats record, both validation paths, and the atomic
write. This stage decides what the corpus contains, so a mistake here is a
silently smaller index rather than a crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quiz_generator.data.ingest import ingest
from quiz_generator.shared.schemas import IngestStats


def _choice(answer: str, is_true: bool = False) -> dict[str, Any]:
    return {"answer": answer, "isTrue": is_true}


def _question(order: int = 0, **overrides: Any) -> dict[str, Any]:
    question: dict[str, Any] = {
        "order": order,
        "type": "MULTIPLE_CHOICE",
        "description": f"<p>Question {order}</p>",
        "choices": [_choice("right", True), _choice("wrong")],
    }
    question.update(overrides)
    return question


def _quiz(quiz_id: str = "quiz1", **overrides: Any) -> dict[str, Any]:
    quiz: dict[str, Any] = {
        "_id": quiz_id,
        "title": "Present Simple",
        "language": "english",
        "subjects": ["ENGLISH"],
        "levels": ["MIDDLE_SCHOOL_1ST_GRADE"],
        "createdBy": {"name": "A Teacher", "email": "teacher@example.test"},
        "questions": [_question(0), _question(1)],
    }
    quiz.update(overrides)
    return quiz


def _run(tmp_path: Path, quizzes: list[dict], **kwargs: Any) -> tuple[IngestStats, list[dict]]:
    """Ingest `quizzes` into tmp_path and return (stats, rows)."""
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(quizzes), encoding="utf-8")
    out = tmp_path / "flat.jsonl"
    stats_path = tmp_path / "flat_stats.json"

    stats = ingest(input_path=raw, output_path=out, stats_path=stats_path, **kwargs)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return stats, rows


def _scope_file(tmp_path: Path, subjects: list[str]) -> Path:
    path = tmp_path / "scope.yaml"
    path.write_text(
        "scope:\n  name: test_scope\n"
        f"  subjects: [{', '.join(subjects)}]\n"
        "  level_prefixes: [MIDDLE_SCHOOL, HIGH_SCHOOL]\n"
        "  languages: [en, fr, ar]\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_every_question_of_a_clean_quiz_becomes_one_row(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_quiz()])

    assert stats.input_quizzes == 1
    assert stats.input_questions == 2
    assert stats.output_rows == 2
    assert len(rows) == 2
    assert stats.dropped == {}


def test_a_row_carries_the_quiz_context_its_question_lacks(tmp_path: Path) -> None:
    """Language, subject and level live on the quiz; retrieval filters on them
    per question, so they have to be copied down."""
    _, rows = _run(tmp_path, [_quiz()])
    row = rows[0]

    assert row["quiz_id"] == "quiz1"
    assert row["quiz_title_raw"] == "Present Simple"
    assert row["language_raw"] == "english"
    assert row["subjects"] == ["ENGLISH"]
    assert row["levels"] == ["MIDDLE_SCHOOL_1ST_GRADE"]


def test_text_is_left_raw_for_the_normalize_stage(tmp_path: Path) -> None:
    """Ingest must not clean: `_raw` fields are stripped later, and stripping
    twice is how double-decoded entities appear."""
    _, rows = _run(tmp_path, [_quiz()])

    assert rows[0]["question_text_raw"] == "<p>Question 0</p>"


def test_multiple_correct_answers_is_derived_not_copied(tmp_path: Path) -> None:
    """The source's own flag is wrong on 870 rows, so it is recomputed."""
    two_correct = _question(0, choices=[_choice("a", True), _choice("b", True)])
    quiz = _quiz(questions=[two_correct, _question(1)])

    _, rows = _run(tmp_path, [quiz])

    assert rows[0]["multiple_correct_answers"] is True
    assert rows[1]["multiple_correct_answers"] is False


def test_text_multiple_choice_is_merged_into_multiple_choice(tmp_path: Path) -> None:
    quiz = _quiz(questions=[_question(0, type="TEXT_MULTIPLE_CHOICE")])

    _, rows = _run(tmp_path, [quiz])

    assert rows[0]["question_type"] == "MULTIPLE_CHOICE"


# ---------------------------------------------------------------------------
# doc_id — the collision that silently ate 6% of the corpus
# ---------------------------------------------------------------------------


def test_duplicate_order_values_get_distinct_doc_ids(tmp_path: Path) -> None:
    """~20% of source quizzes repeat `order`. Chroma overwrites on a repeated
    id, so a collision here deletes a question from the index without a word."""
    quiz = _quiz(questions=[_question(5), _question(5), _question(5)])

    _, rows = _run(tmp_path, [quiz])
    ids = [row["doc_id"] for row in rows]

    assert ids == ["quiz1__q5", "quiz1__q5_2", "quiz1__q5_3"]
    assert len(set(ids)) == 3


def test_the_order_counter_does_not_leak_between_quizzes(tmp_path: Path) -> None:
    """Two quizzes each with order=0 must both keep the plain `q0` form."""
    _, rows = _run(
        tmp_path, [_quiz("a", questions=[_question(0)]), _quiz("b", questions=[_question(0)])]
    )

    assert [row["doc_id"] for row in rows] == ["a__q0", "b__q0"]


# ---------------------------------------------------------------------------
# Structural drops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"choices": []}, "empty_choices"),
        ({"choices": [_choice("a"), _choice("b")]}, "no_correct_answer"),
        ({"description": "", "image": "pic.png"}, "image_only"),
    ],
)
def test_a_structurally_unusable_question_is_dropped_with_its_reason(
    tmp_path: Path, overrides: dict, reason: str
) -> None:
    quiz = _quiz(questions=[_question(0, **overrides)])

    stats, rows = _run(tmp_path, [quiz])

    assert rows == []
    assert stats.dropped == {reason: 1}
    assert stats.output_rows == 0


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_an_unknown_question_type_is_a_validation_failure_not_a_filter_drop(
    tmp_path: Path,
) -> None:
    """`decide_drop` also rejects unknown types, but RawQuestion's Literal gets
    there first, so the row is counted as a validation failure and the sample
    names the offending field. The filter branch stays as a guard for callers
    that do not validate first."""
    quiz = _quiz(questions=[_question(0, type="ORDERING")])

    stats, rows = _run(tmp_path, [quiz])

    assert rows == []
    assert stats.question_validation_errors == 1
    assert "type" in stats.validation_failure_samples[0].error
    assert "invalid_type" not in stats.dropped


def test_without_a_scope_file_nothing_is_dropped_for_scope(tmp_path: Path) -> None:
    quiz = _quiz(subjects=["CHEMISTRY"], levels=["LICENCE_1"])

    stats, rows = _run(tmp_path, [quiz])

    assert len(rows) == 2
    assert not any(key.startswith("scope_") for key in stats.dropped)


def test_an_out_of_scope_subject_is_dropped_and_labelled(tmp_path: Path) -> None:
    scope = _scope_file(tmp_path, ["ENGLISH"])
    quizzes = [_quiz("keep"), _quiz("drop", subjects=["CHEMISTRY"])]

    stats, rows = _run(tmp_path, quizzes, scope_path=scope)

    assert {row["quiz_id"] for row in rows} == {"keep"}
    assert stats.dropped == {"scope_subject_out_of_scope": 2}


def test_an_out_of_scope_level_is_dropped_and_labelled(tmp_path: Path) -> None:
    scope = _scope_file(tmp_path, ["ENGLISH"])
    quizzes = [_quiz("drop", levels=["LICENCE_1"])]

    stats, _ = _run(tmp_path, quizzes, scope_path=scope)

    assert stats.dropped == {"scope_level_out_of_scope": 2}


def test_structural_filters_run_before_the_scope_filter(tmp_path: Path) -> None:
    """A row that is both unusable and out of scope is counted once, under the
    structural reason — otherwise the two drop tallies would double-count."""
    scope = _scope_file(tmp_path, ["ENGLISH"])
    quiz = _quiz("x", subjects=["CHEMISTRY"], questions=[_question(0, choices=[])])

    stats, _ = _run(tmp_path, [quiz], scope_path=scope)

    assert stats.dropped == {"empty_choices": 1}


# ---------------------------------------------------------------------------
# The stats record
# ---------------------------------------------------------------------------


def test_the_stats_file_is_written_and_matches_what_was_returned(tmp_path: Path) -> None:
    stats, rows = _run(tmp_path, [_quiz()])
    written = json.loads((tmp_path / "flat_stats.json").read_text(encoding="utf-8"))

    assert written["output_rows"] == len(rows) == stats.output_rows
    assert written["kept_by_language_raw"] == {"english": 2}
    assert written["kept_by_type"] == {"MULTIPLE_CHOICE": 2}


def test_the_stats_record_names_the_scope_that_produced_it(tmp_path: Path) -> None:
    """Drop counts are meaningless without the scope they were measured under:
    two stats files from different scopes used to be indistinguishable."""
    scope = _scope_file(tmp_path, ["ENGLISH"])

    stats, _ = _run(tmp_path, [_quiz()], scope_path=scope)

    assert stats.scope_name == "test_scope"
    assert stats.scope_config_path == str(scope)
    assert stats.scope_config_sha256 is not None and len(stats.scope_config_sha256) == 64


def test_no_scope_means_no_scope_provenance(tmp_path: Path) -> None:
    stats, _ = _run(tmp_path, [_quiz()])

    assert (stats.scope_name, stats.scope_config_path, stats.scope_config_sha256) == (
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_a_malformed_quiz_is_counted_and_the_others_still_run(tmp_path: Path) -> None:
    """One bad quiz must not abort the ingest of 1,371 good ones."""
    broken = {"_id": "broken", "questions": [_question(0)], "subjects": "ENGLISH"}

    stats, rows = _run(tmp_path, [broken, _quiz("good")])

    assert stats.quiz_validation_errors == 1
    assert stats.dropped["quiz_validation_failed"] == 1
    assert {row["quiz_id"] for row in rows} == {"good"}


def test_a_malformed_question_is_counted_without_losing_its_siblings(tmp_path: Path) -> None:
    quiz = _quiz(questions=[{"order": "not-an-int", "type": "MULTIPLE_CHOICE"}, _question(1)])

    stats, rows = _run(tmp_path, [quiz])

    assert stats.question_validation_errors == 1
    assert len(rows) == 1


def test_a_validation_failure_is_sampled_with_its_message_not_its_payload(
    tmp_path: Path,
) -> None:
    """A count alone is not actionable; the payload would carry corpus content
    and author names, so only the field and the rule are recorded."""
    quiz = _quiz(questions=[{"order": "not-an-int", "type": "MULTIPLE_CHOICE"}])

    stats, _ = _run(tmp_path, [quiz])
    sample = stats.validation_failure_samples[0]

    assert sample.level == "question"
    assert sample.quiz_id == "quiz1"
    assert "order" in sample.error
    assert "A Teacher" not in sample.error
    assert "teacher@example.test" not in sample.error


def test_the_failure_sample_is_capped(tmp_path: Path) -> None:
    """A systematically broken export must not write a huge stats file."""
    bad = [{"order": "nope", "type": "MULTIPLE_CHOICE"} for _ in range(50)]

    stats, _ = _run(tmp_path, [_quiz(questions=bad)])

    assert stats.question_validation_errors == 50
    assert len(stats.validation_failure_samples) == 5


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_output_is_written_atomically(tmp_path: Path) -> None:
    """The stage used to write straight into flat.jsonl, so a crash partway
    left a truncated file that the next stage reads as a complete corpus."""
    _run(tmp_path, [_quiz()])

    assert not (tmp_path / "flat.jsonl.tmp").exists()
    assert not (tmp_path / "flat_stats.json.tmp").exists()


def test_re_running_produces_the_same_output(tmp_path: Path) -> None:
    """Idempotence (CLAUDE.md Part II §3): the same input yields the same rows,
    not appended duplicates."""
    first_stats, first_rows = _run(tmp_path, [_quiz()])
    second_stats, second_rows = _run(tmp_path, [_quiz()])

    assert first_rows == second_rows
    assert first_stats.model_dump() == second_stats.model_dump()


def test_limit_quizzes_stops_early(tmp_path: Path) -> None:
    stats, _ = _run(tmp_path, [_quiz("a"), _quiz("b"), _quiz("c")], limit_quizzes=2)

    assert stats.input_quizzes == 2


def test_a_top_level_object_instead_of_an_array_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"quizzes": [_quiz()]}), encoding="utf-8")

    with pytest.raises(ValueError, match="array"):
        ingest(
            input_path=raw,
            output_path=tmp_path / "flat.jsonl",
            stats_path=tmp_path / "stats.json",
        )
