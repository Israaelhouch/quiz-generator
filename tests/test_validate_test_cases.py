"""Tests for the eval answer-key checks in scripts/eval/validate_test_cases.py.

The retrieval eval scores a retrieved question as correct only if its doc_id is
listed under the target topic in eval/topics_<lang>.csv. Until these checks
existed nothing compared those CSVs with the index, so a stale answer key
passed validation and silently marked correct retrievals as wrong. All data
here is synthetic.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.run_retriever_eval import run_eval, validate_or_die
from scripts.eval.validate_test_cases import (
    TOPICS_FILE_BY_LANG_SUBJECT,
    IndexDoc,
    check_ground_truth_against_index,
    cross_check,
    load_index,
    validate,
)
from scripts.eval.validate_test_cases import (
    TestCase as RetrieverTestCase,
)

EN = ("en", "ENGLISH")


def _doc(title: str, language: str = "en", subject: str = "ENGLISH") -> IndexDoc:
    return IndexDoc(language=language, subject=subject, quiz_title=title)


# ---------------------------------------------------------------------------
# check_ground_truth_against_index — pure logic
# ---------------------------------------------------------------------------


def test_check_ground_truth_complete_key_reports_no_problems() -> None:
    index = {"quiz1__q0": _doc("Irregular Verbs"), "quiz1__q1": _doc("Irregular Verbs")}
    topics = {EN: {"Irregular Verbs": {"quiz1__q0", "quiz1__q1"}}}
    assert not check_ground_truth_against_index(topics, index).has_problems


def test_check_ground_truth_doc_id_absent_from_index_is_unknown() -> None:
    index = {"quiz1__q0": _doc("Irregular Verbs")}
    topics = {EN: {"Irregular Verbs": {"quiz1__q0", "quiz1__xq6"}}}
    problems = check_ground_truth_against_index(topics, index)
    assert [i.doc_id for i in problems.unknown_ids] == ["quiz1__xq6"]
    assert not problems.wrong_cell_ids
    assert not problems.unlisted_docs


def test_check_ground_truth_doc_id_in_another_cell_is_wrong_cell() -> None:
    index = {
        "quiz1__q0": _doc("Irregular Verbs"),
        "quiz9__q0": _doc("Les verbes", language="fr", subject="FRENCH"),
    }
    topics = {EN: {"Irregular Verbs": {"quiz1__q0", "quiz9__q0"}}}
    problems = check_ground_truth_against_index(topics, index)
    assert [(i.doc_id, i.detail) for i in problems.wrong_cell_ids] == [("quiz9__q0", "fr x FRENCH")]


def test_check_ground_truth_collision_suffixed_id_not_listed_is_flagged() -> None:
    """Regression: the answer key predated the doc_id collision fix.

    The fix kept the first question with a repeated `order` as `__q3` and gave
    the next one `__q3_2`. A key built before it lists only `__q3`, so the
    retriever returning `__q3_2` — same quiz, same title — was scored wrong.
    """
    index = {"quiz1__q3": _doc("Writing Ads"), "quiz1__q3_2": _doc("Writing Ads")}
    topics = {EN: {"Writing Ads": {"quiz1__q3"}}}
    problems = check_ground_truth_against_index(topics, index)
    assert [(i.doc_id, i.detail) for i in problems.unlisted_docs] == [
        ("quiz1__q3_2", "Writing Ads")
    ]


def test_check_ground_truth_unlisted_doc_under_merged_title_is_flagged() -> None:
    """A hand-merged topic owns every title its listed doc_ids carry.

    The canonical name need not exist on any indexed question — 'The Present
    Perfect' is spelled differently everywhere in the real corpus.
    """
    index = {
        "quiz1__q0": _doc("present perfect"),
        "quiz2__q0": _doc("The present perfect"),
        "quiz2__q1": _doc("The present perfect"),
    }
    topics = {EN: {"The Present Perfect": {"quiz1__q0", "quiz2__q0"}}}
    problems = check_ground_truth_against_index(topics, index)
    assert [(i.doc_id, i.detail) for i in problems.unlisted_docs] == [
        ("quiz2__q1", "The present perfect")
    ]


def test_check_ground_truth_unrelated_title_not_listed_is_ignored() -> None:
    index = {"quiz1__q0": _doc("Irregular Verbs"), "quiz5__q0": _doc("Pets")}
    topics = {EN: {"Irregular Verbs": {"quiz1__q0"}}}
    assert not check_ground_truth_against_index(topics, index).has_problems


def test_check_ground_truth_same_title_in_other_cell_is_not_flagged() -> None:
    """The retriever pre-filters on (language, subject); other cells are unreachable."""
    index = {
        "quiz1__q0": _doc("Vocabulary"),
        "quiz8__q0": _doc("Vocabulary", language="fr", subject="FRENCH"),
    }
    topics = {EN: {"Vocabulary": {"quiz1__q0"}}}
    assert not check_ground_truth_against_index(topics, index).has_problems


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------


def test_load_index_first_subject_defines_the_cell(tmp_path: Path) -> None:
    """subjects[0] is the scalar the vector store copies for pre-filtering."""
    ready = tmp_path / "ready.jsonl"
    rows = [
        {"doc_id": "a__q0", "language": "en", "subjects": ["ENGLISH"], "quiz_title": "Pets"},
        {
            "doc_id": "b__q0",
            "language": "en",
            "subjects": ["MATHEMATICS", "ENGLISH"],
            "quiz_title": "Pets",
        },
        {"doc_id": "c__q0", "language": "en", "subjects": [], "quiz_title": None},
    ]
    ready.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    index = load_index(ready)
    assert index["a__q0"] == IndexDoc("en", "ENGLISH", "Pets")
    assert index["b__q0"].subject == "MATHEMATICS"
    assert index["c__q0"] == IndexDoc("en", "", "")


# ---------------------------------------------------------------------------
# validate() and the harness gate — file level
# ---------------------------------------------------------------------------


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listed: str) -> tuple[Path, Path]:
    """Write a one-topic key listing `listed`, an index holding __q3 and __q3_2."""
    topics_csv = tmp_path / "topics_english.csv"
    with topics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quiz_title", "doc_ids"])
        writer.writeheader()
        writer.writerow({"quiz_title": "Writing Ads", "doc_ids": listed})
    monkeypatch.setitem(TOPICS_FILE_BY_LANG_SUBJECT, EN, topics_csv)

    ready = tmp_path / "ready.jsonl"
    ready.write_text(
        "".join(
            json.dumps(
                {
                    "doc_id": d,
                    "language": "en",
                    "subjects": ["ENGLISH"],
                    "quiz_title": "Writing Ads",
                }
            )
            + "\n"
            for d in ("quiz1__q3", "quiz1__q3_2")
        ),
        encoding="utf-8",
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "query": "writing ads",
                    "language": "en",
                    "subject": "ENGLISH",
                    "top_k": 1,
                    "query_type": "direct",
                    "target_quiz_title": "Writing Ads",
                }
            ]
        ),
        encoding="utf-8",
    )
    return cases, ready


def test_validate_stale_answer_key_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cases, ready = _fixture(tmp_path, monkeypatch, listed="quiz1__q3")
    assert validate(cases, ready) == 1
    out = capsys.readouterr().out
    assert "unlisted questions:    1" in out
    assert "quiz1__q3_2" in out


def test_validate_complete_answer_key_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases, ready = _fixture(tmp_path, monkeypatch, listed="quiz1__q3,quiz1__q3_2")
    assert validate(cases, ready) == 0


def test_validate_missing_index_payload_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases, _ready = _fixture(tmp_path, monkeypatch, listed="quiz1__q3,quiz1__q3_2")
    assert validate(cases, tmp_path / "absent.jsonl") == 1


def test_validate_or_die_stale_answer_key_refuses_to_run_the_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness must stop before spending ~30 minutes on a stale key."""
    cases, ready = _fixture(tmp_path, monkeypatch, listed="quiz1__q3")
    with pytest.raises(SystemExit) as exc:
        validate_or_die(cases, ready)
    assert exc.value.code == 1


def test_validate_or_die_complete_answer_key_returns_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases, ready = _fixture(tmp_path, monkeypatch, listed="quiz1__q3,quiz1__q3_2")
    assert [c.target_quiz_title for c in validate_or_die(cases, ready)] == ["Writing Ads"]


# ---------------------------------------------------------------------------
# Several correct quizzes per test case
# ---------------------------------------------------------------------------


def _case(**overrides: object) -> RetrieverTestCase:
    fields: dict[str, object] = {
        "query": "my students mix up since and for",
        "language": "en",
        "subject": "ENGLISH",
        "top_k": 5,
        "query_type": "goal",
        "target_quiz_title": "The Present Perfect",
    }
    fields.update(overrides)
    return RetrieverTestCase(**fields)


def test_testcase_without_also_correct_titles_parses_as_before() -> None:
    """Existing test-case files carry no also_correct_quiz_titles field."""
    case = _case()
    assert case.also_correct_quiz_titles == []
    assert case.target_titles == ["The Present Perfect"]


def test_testcase_target_titles_lists_primary_first_without_duplicates() -> None:
    case = _case(
        also_correct_quiz_titles=["Used to Vs. Would", "The Present Perfect", "Used to Vs. Would"]
    )
    assert case.target_titles == ["The Present Perfect", "Used to Vs. Would"]


def test_cross_check_missing_also_correct_title_is_reported_by_name() -> None:
    index = {("en", "The Present Perfect"): 97}
    case = _case(also_correct_quiz_titles=["Used to Vs Wuold"])
    missing, capped = cross_check([case], index)
    assert [(i, title) for i, _c, title in missing] == [(0, "Used to Vs Wuold")]
    assert capped == []


def test_cross_check_recall_cap_uses_all_correct_titles() -> None:
    index = {("en", "A"): 2, ("en", "B"): 2}
    _missing, capped = cross_check(
        [_case(target_quiz_title="A", also_correct_quiz_titles=["B"], top_k=5)], index
    )
    assert [n for _i, _c, n in capped] == [4]


class _FixedRetriever:
    """Returns the same doc_ids for every query."""

    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def retrieve(self, **_kwargs: object) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(doc_id=d, distance=0.1) for d in self.doc_ids]


def test_run_eval_counts_a_hit_on_any_correct_quiz() -> None:
    ground_truth = {("en", "A"): {"a1"}, ("en", "B"): {"b1"}}
    retriever = _FixedRetriever(["b1", "x1"])
    only_a, a_or_b = run_eval(
        [
            _case(target_quiz_title="A"),
            _case(target_quiz_title="A", also_correct_quiz_titles=["B"]),
        ],
        retriever,
        ground_truth,
    )
    assert only_a["precision_at_1"] == 0.0
    assert a_or_b["precision_at_1"] == 1.0
    assert a_or_b["n_relevant"] == 2
    assert a_or_b["also_correct_quiz_titles"] == ["B"]
