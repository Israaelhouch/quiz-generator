"""Tests for Stage 3 — pure-Python helpers in src/indexing.

Skip the heavy paths (actual model loading, Chroma client) — those are
integration tests best run on the user's machine. Here we validate:
  - metadata serialization shape
  - ID deduplication
  - empty-text drop behavior
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.taxonomy import SCHOOL_LEVEL_PREFIXES, Taxonomy
from src.indexing.vector_store import build_ids, row_to_metadata

# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------
# These signals moved from `warnings.warn` to `logger.warning`. The warnings
# module dedupes by (message, category, module, lineno) per process, so in a
# long-running server each of these fired ONCE and was then silent forever —
# exactly backwards for an operational signal. Tests assert on log records now.

import contextlib as _contextlib
import logging as _logging


@_contextlib.contextmanager
def _capture_logs(logger_name: str, level: int = _logging.WARNING):
    """Collect records emitted by `logger_name` inside the block."""
    target = _logging.getLogger(logger_name)
    records: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=level)
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(level)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)




SCALAR_FIELDS = [
    "quiz_id",
    "quiz_title",
    "language",
    "question_type",
    "multiple_correct_answers",
    "author_name",
    "author_email",
    "points",
    "time",
]
# After the scalar-subject refactor, only `levels` stays as a JSON list.
# `subject` is derived separately from `subjects[0]`.
LIST_FIELDS = ["levels"]


def test_row_to_metadata_preserves_scalars() -> None:
    row = {
        "quiz_id": "q1",
        "quiz_title": "Immunity",
        "language": "en",
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "author_name": "Prof X",
        "author_email": "x@y.z",
        "points": 1.5,
        "time": 30,
        "subjects": ["SCIENCE"],
        "levels": ["PRIMARY_SCHOOL_6TH_GRADE"],
        "question_text": "not included",
        "choices_text": ["not included"],
    }
    meta = row_to_metadata(row, scalar_fields=SCALAR_FIELDS, list_fields_as_json=LIST_FIELDS)

    assert meta["quiz_id"] == "q1"
    assert meta["quiz_title"] == "Immunity"
    assert meta["language"] == "en"
    assert meta["multiple_correct_answers"] is False
    assert meta["points"] == 1.5
    assert meta["time"] == 30
    assert meta["author_name"] == "Prof X"
    # Choices/question_text not requested -> not present
    assert "question_text" not in meta
    assert "choices_text" not in meta


def test_row_to_metadata_derives_scalar_subject() -> None:
    """Primary subject becomes a scalar Chroma field (natively filterable)."""
    row = {"quiz_id": "x", "subjects": ["PHYSICS"], "levels": ["L1"]}
    meta = row_to_metadata(row, scalar_fields=["quiz_id"], list_fields_as_json=LIST_FIELDS)
    assert meta["subject"] == "PHYSICS"
    assert "subjects_json" not in meta  # no longer stored as JSON


def test_row_to_metadata_drops_secondary_subjects() -> None:
    """The 3 rows with 2 subjects lose the second; only first survives."""
    row = {"quiz_id": "x", "subjects": ["PHYSICS", "MATHEMATICS"], "levels": []}
    meta = row_to_metadata(row, scalar_fields=["quiz_id"], list_fields_as_json=LIST_FIELDS)
    assert meta["subject"] == "PHYSICS"


def test_row_to_metadata_no_subject_when_empty_list() -> None:
    row = {"quiz_id": "x", "subjects": [], "levels": ["L1"]}
    meta = row_to_metadata(row, scalar_fields=["quiz_id"], list_fields_as_json=LIST_FIELDS)
    assert "subject" not in meta


def test_row_to_metadata_levels_still_as_json_list() -> None:
    row = {"quiz_id": "x", "subjects": ["MATH"], "levels": ["L1", "L2", "L3"]}
    meta = row_to_metadata(row, scalar_fields=["quiz_id"], list_fields_as_json=LIST_FIELDS)
    assert meta["levels_json"] == '["L1", "L2", "L3"]'
    assert "levels" not in meta


def test_row_to_metadata_levels_as_booleans() -> None:
    """Essential multi-valued filter — expanded to per-value True booleans."""
    row = {
        "quiz_id": "x",
        "subjects": [],
        "levels": ["HIGH_SCHOOL_4TH_GRADE_MATHEMATICS", "HIGH_SCHOOL_4TH_GRADE_TECHNIQUES"],
    }
    meta = row_to_metadata(
        row,
        scalar_fields=["quiz_id"],
        list_fields_as_json=[],
        list_fields_as_booleans=["levels"],
    )
    assert meta["levels_HIGH_SCHOOL_4TH_GRADE_MATHEMATICS"] is True
    assert meta["levels_HIGH_SCHOOL_4TH_GRADE_TECHNIQUES"] is True
    # No False padding — absent keys = False by convention
    assert "levels_HIGH_SCHOOL_2ND_GRADE_SCIENCE" not in meta
    # No JSON list stored
    assert "levels_json" not in meta


def test_row_to_metadata_boolean_expansion_skips_empty_values() -> None:
    row = {"quiz_id": "x", "subjects": [], "levels": ["L1", "", None, "   ", "L2"]}
    meta = row_to_metadata(
        row,
        scalar_fields=["quiz_id"],
        list_fields_as_json=[],
        list_fields_as_booleans=["levels"],
    )
    assert meta.get("levels_L1") is True
    assert meta.get("levels_L2") is True
    # Noise values don't produce keys
    assert "levels_" not in meta
    assert "levels_   " not in meta


def test_row_to_metadata_drops_none_and_empty_strings() -> None:
    row = {
        "quiz_id": "q1",
        "quiz_title": "",          # empty string -> dropped
        "language": "fr",
        "question_type": None,      # None -> dropped
        "points": None,             # None -> dropped
        "time": 0,                  # 0 is a valid scalar, kept
        "author_name": "   ",       # whitespace -> dropped
        "subjects": [],
        "levels": [],
    }
    meta = row_to_metadata(row, scalar_fields=SCALAR_FIELDS, list_fields_as_json=LIST_FIELDS)
    assert "quiz_title" not in meta
    assert "question_type" not in meta
    assert "points" not in meta
    assert "author_name" not in meta
    assert meta["time"] == 0
    assert "subject" not in meta  # empty subjects -> no scalar
    assert meta["levels_json"] == "[]"


def test_row_to_metadata_arabic_preserved_in_subject_scalar() -> None:
    row = {"quiz_id": "q", "subjects": ["الرياضيات"], "levels": []}
    meta = row_to_metadata(row, scalar_fields=["quiz_id"], list_fields_as_json=LIST_FIELDS)
    assert meta["subject"] == "الرياضيات"


def test_build_ids_uses_doc_id_when_unique() -> None:
    rows = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
    assert build_ids(rows) == ["a", "b", "c"]


def test_build_ids_deduplicates_collisions() -> None:
    rows = [{"doc_id": "x"}, {"doc_id": "x"}, {"doc_id": "x"}]
    assert build_ids(rows) == ["x", "x__dup1", "x__dup2"]


def test_build_ids_fallback_for_missing_doc_id() -> None:
    rows = [{}, {"doc_id": ""}, {"doc_id": "q"}]
    ids = build_ids(rows)
    assert ids[0].startswith("row_")
    assert ids[1].startswith("row_")
    assert ids[2] == "q"


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_taxonomy_from_rows_collects_all_distinct_values() -> None:
    rows = [
        {"language": "en", "question_type": "MCQ", "subjects": ["SCIENCE"], "levels": ["L1"]},
        {"language": "fr", "question_type": "MCQ", "subjects": ["MATH"], "levels": ["L1", "L2"]},
        {"language": "fr", "question_type": "FITB", "subjects": [], "levels": []},
    ]
    tax = Taxonomy.from_rows(rows)
    assert tax.languages == {"en", "fr"}
    assert tax.question_types == {"MCQ", "FITB"}
    assert tax.subjects == {"SCIENCE", "MATH"}
    assert tax.levels == {"L1", "L2"}


def test_taxonomy_from_rows_ignores_empty_and_none() -> None:
    rows = [
        {"language": "  ", "subjects": [None, "", " "], "levels": [""]},
        {"language": "en", "subjects": ["MATH"], "levels": ["L1"]},
    ]
    tax = Taxonomy.from_rows(rows)
    assert tax.languages == {"en"}
    assert tax.subjects == {"MATH"}
    assert tax.levels == {"L1"}


def test_taxonomy_to_dict_returns_sorted_lists() -> None:
    tax = Taxonomy(
        languages={"fr", "en", "ar"},
        subjects={"PHYSICS", "MATH"},
        levels={"L2", "L1"},
        question_types={"MCQ"},
    )
    d = tax.to_dict()
    assert d["languages"] == ["ar", "en", "fr"]
    assert d["subjects"] == ["MATH", "PHYSICS"]
    assert d["levels"] == ["L1", "L2"]


def test_taxonomy_validate_level_warns_on_unknown() -> None:
    tax = Taxonomy(levels={"HIGH_SCHOOL_4TH_GRADE_MATH"})
    with _capture_logs("src.indexing.taxonomy") as records:
        assert tax.validate_level("HIGH_SCHOOL_4TH_GRAD_MATH") is False   # typo
        assert tax.validate_level("HIGH_SCHOOL_4TH_GRADE_MATH") is True   # exact
    # Only the typo call produced a signal.
    messages = [rec.getMessage() for rec in records]
    assert len(messages) == 1, messages
    assert "not in known taxonomy" in messages[0]


def test_taxonomy_validate_empty_does_not_warn() -> None:
    """No taxonomy loaded → validation is a no-op (nothing logged)."""
    tax = Taxonomy()
    with _capture_logs("src.indexing.taxonomy") as records:
        tax.validate_level("anything")
        tax.validate_subject("anything")
        tax.validate_language("xx")
    assert records == []


def test_taxonomy_from_build_summary_missing_file() -> None:
    tax = Taxonomy.from_build_summary(Path("/nonexistent/build_summary.json"))
    assert tax.is_empty()


def test_taxonomy_list_methods() -> None:
    tax = Taxonomy(
        languages={"fr", "en"},
        subjects={"MATH", "PHYSICS"},
        levels={"L2", "L1"},
        question_types={"MCQ", "FITB"},
    )
    assert tax.list_languages() == ["en", "fr"]
    assert tax.list_subjects() == ["MATH", "PHYSICS"]
    assert tax.list_levels() == ["L1", "L2"]
    assert tax.list_question_types() == ["FITB", "MCQ"]


# ---------------------------------------------------------------------------
# Taxonomy — out-of-scope level leak (regression)
# ---------------------------------------------------------------------------
# scope.decide_in_scope() only inspects levels[0], so a row whose FIRST level
# is in scope is kept even when it carries out-of-scope SECONDARY tags. The
# taxonomy must not then advertise those secondary tags as legal filter
# values — /taxonomy feeds the platform's dropdowns, and picking a phantom
# level yields an empty retrieval and an HTTP 502.

def test_taxonomy_from_rows_filters_out_of_scope_secondary_levels() -> None:
    rows = [{
        "language": "ar",
        "question_type": "MULTIPLE_CHOICE",
        "subjects": ["ARABIC"],
        # levels[0] is in scope, so scope.decide_in_scope kept this row
        "levels": [
            "HIGH_SCHOOL_1ST_GRADE",
            "LICENCE_1ST_GRADE",       # out of scope
            "PREPARATORY_1ST_MP",      # out of scope
            "MIDDLE_SCHOOL_2ND_GRADE", # in scope
        ],
    }]
    tax = Taxonomy.from_rows(rows, level_prefixes=SCHOOL_LEVEL_PREFIXES)
    assert tax.levels == {"HIGH_SCHOOL_1ST_GRADE", "MIDDLE_SCHOOL_2ND_GRADE"}
    # Non-level fields are untouched by the filter
    assert tax.languages == {"ar"}
    assert tax.subjects == {"ARABIC"}


def test_taxonomy_from_rows_unfiltered_by_default() -> None:
    """Default stays self-adapting — no prefix opinion unless asked."""
    rows = [{"language": "ar", "subjects": [], "levels": ["LICENCE_1ST_GRADE", "L1"]}]
    tax = Taxonomy.from_rows(rows)
    assert tax.levels == {"LICENCE_1ST_GRADE", "L1"}


def test_taxonomy_from_build_summary_filters_on_load() -> None:
    """An index built BEFORE the filter existed is corrected at read time,
    so no reindex is needed to stop publishing phantom levels."""
    import tempfile

    summary = {
        "rows_indexed": 3,
        "taxonomy": {
            "languages": ["ar", "en"],
            "question_types": ["MULTIPLE_CHOICE"],
            "subjects": ["ARABIC"],
            "levels": [
                "HIGH_SCHOOL_1ST_GRADE",
                "LICENCE_2ND_GRADE",
                "PREPARATORY_2ND_PC",
                "PRIMARY_SCHOOL_4TH_GRADE",
            ],
        },
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "build_summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        filtered = Taxonomy.from_build_summary(path, level_prefixes=SCHOOL_LEVEL_PREFIXES)
        assert filtered.list_levels() == [
            "HIGH_SCHOOL_1ST_GRADE",
            "PRIMARY_SCHOOL_4TH_GRADE",
        ]
        # Other fields survive the round-trip intact
        assert filtered.list_languages() == ["ar", "en"]

        # Without the argument, historical behaviour is preserved
        unfiltered = Taxonomy.from_build_summary(path)
        assert "LICENCE_2ND_GRADE" in unfiltered.levels


def test_taxonomy_rejects_every_known_phantom_level() -> None:
    """The exact 12 values that leaked into the live /taxonomy response."""
    phantom = [
        "LICENCE_1ST_GRADE", "LICENCE_2ND_GRADE", "LICENCE_3RD_GRADE",
        "PREPARATORY_1ST_BG", "PREPARATORY_1ST_MP", "PREPARATORY_1ST_MPI",
        "PREPARATORY_1ST_PC", "PREPARATORY_1ST_TECHNO",
        "PREPARATORY_2ND_BG", "PREPARATORY_2ND_MP",
        "PREPARATORY_2ND_PC", "PREPARATORY_2ND_TECHNO",
    ]
    rows = [{"language": "ar", "subjects": [], "levels": ["HIGH_SCHOOL_1ST_GRADE"] + phantom}]
    tax = Taxonomy.from_rows(rows, level_prefixes=SCHOOL_LEVEL_PREFIXES)
    assert tax.levels == {"HIGH_SCHOOL_1ST_GRADE"}


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All Stage 3 tests passed.")
