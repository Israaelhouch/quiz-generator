"""Tests for Stage 2a — ingestion filter logic.

These tests are stdlib-only on purpose: they validate `src/data/filters.py`
so we can verify ingestion behavior without needing Pydantic installed
in every environment. Full-pipeline tests live in test_ingest_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.filters import (
    count_correct,
    decide_drop,
    derive_multiple_correct_answers,
    has_correct_answer,
    is_image_only,
    strip_html_to_plain,
)


def test_strip_html_decodes_entities_and_removes_tags() -> None:
    assert strip_html_to_plain("<p>hello&nbsp;world</p>") == "hello world"
    assert strip_html_to_plain("<p></p>") == ""
    assert strip_html_to_plain(None) == ""
    assert strip_html_to_plain("   ") == ""


def test_has_correct_answer_requires_content_and_flag() -> None:
    assert has_correct_answer([{"answer": "a", "isTrue": True}])
    assert not has_correct_answer([{"answer": "a", "isTrue": False}])
    # Choice flagged true but empty and no media -> not considered a real correct answer.
    assert not has_correct_answer([{"answer": "", "isTrue": True, "media": None}])
    # Media-only correct answer still counts.
    assert has_correct_answer([{"answer": "", "isTrue": True, "media": "img.png"}])
    assert not has_correct_answer([])


def test_is_image_only_detects_image_without_visible_text() -> None:
    assert is_image_only("<p></p>", "http://x/img.png")
    assert is_image_only(None, "http://x/img.png")
    assert not is_image_only("Real text", "http://x/img.png")
    assert not is_image_only(None, None)


def test_count_and_derive_multiple_correct_answers() -> None:
    two = [{"isTrue": True}, {"isTrue": True}, {"isTrue": False}]
    one = [{"isTrue": True}, {"isTrue": False}]
    zero = [{"isTrue": False}, {"isTrue": False}]
    assert count_correct(two) == 2
    assert derive_multiple_correct_answers(two) is True
    assert derive_multiple_correct_answers(one) is False
    assert derive_multiple_correct_answers(zero) is False


def test_decide_drop_reports_first_failing_rule() -> None:
    # empty_choices comes before everything
    assert decide_drop({"type": "MULTIPLE_CHOICE", "choices": []}) == (True, "empty_choices")

    # invalid_type overrides no_correct_answer
    q_bad_type = {"type": "POLL", "choices": [{"isTrue": False}]}
    assert decide_drop(q_bad_type) == (True, "invalid_type")

    # no_correct_answer
    q_no_correct = {
        "type": "MULTIPLE_CHOICE",
        "choices": [{"answer": "a", "isTrue": False}, {"answer": "b", "isTrue": False}],
    }
    assert decide_drop(q_no_correct) == (True, "no_correct_answer")

    # image_only
    q_image = {
        "type": "MULTIPLE_CHOICE",
        "description": "<p></p>",
        "image": "http://x/img.png",
        "choices": [{"answer": "a", "isTrue": True}],
    }
    assert decide_drop(q_image) == (True, "image_only")

    # Valid question survives
    q_ok = {
        "type": "MULTIPLE_CHOICE",
        "description": "<p>What is 2+2?</p>",
        "image": None,
        "choices": [{"answer": "4", "isTrue": True}, {"answer": "5", "isTrue": False}],
    }
    assert decide_drop(q_ok) == (False, "")


if __name__ == "__main__":
    # Simple runner so this file can be executed without pytest.
    test_strip_html_decodes_entities_and_removes_tags()
    test_has_correct_answer_requires_content_and_flag()
    test_is_image_only_detects_image_without_visible_text()
    test_count_and_derive_multiple_correct_answers()
    test_decide_drop_reports_first_failing_rule()
    print("All Stage 2a filter tests passed.")
