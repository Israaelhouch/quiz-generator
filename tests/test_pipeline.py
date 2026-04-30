"""Tests for Stage 6 — QuizPipeline orchestrator.

Uses test-injection hooks (`_retriever`, `_llm_client`) to bypass the real
Retriever and Ollama, so these tests run without ML stack or a server.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.llm_client import MockClient
from src.generation.schemas import GeneratedQuiz
from src.pipeline import QuizPipeline
from src.pipeline.cli import (
    render_human,
    render_json,
    render_retrieval_human,
    retrieval_to_dict,
    save_run_to_file,
)
from src.retrieval.schemas import RetrievedQuestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _retrieved(doc_id: str, question_text: str = "What is X?") -> RetrievedQuestion:
    return RetrievedQuestion(
        doc_id=doc_id,
        quiz_id="quiz-1",
        quiz_title="Test Quiz",
        language="en",
        question_type="MULTIPLE_CHOICE",
        question_text=question_text,
        choices=["A", "B", "C", "D"],
        correct_answers=["A"],
        subjects=["SCIENCE"],
        levels=["PRIMARY_SCHOOL_6TH_GRADE"],
        multiple_correct_answers=False,
        author_name=None,
        author_email=None,
        search_text="search text",
        metadata={},
        distance=0.2,
    )


class _FakeRetriever:
    """Stand-in for src.retrieval.Retriever. Returns a canned list."""

    def __init__(self, results: list[RetrievedQuestion]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def retrieve(self, **kwargs) -> list[RetrievedQuestion]:
        self.calls.append(kwargs)
        return self.results


# A minimal models.yaml that the pipeline can load. We point persist_directory
# at /tmp because the validator requires the field but we never use it (the
# Retriever is injected). Use whatever exists.
def _write_minimal_config(tmp_dir: Path) -> Path:
    """Create a minimal valid models.yaml in tmp_dir and return the path."""
    config_dir = tmp_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_dir / "models.yaml"
    cfg.write_text(
        "model:\n"
        "  name: BAAI/bge-m3\n"
        "  embedding_dim: 1024\n"
        "vector_store:\n"
        "  type: chroma\n"
        "  persist_directory: /tmp/unused_chroma\n"
        "  collection_name: t\n"
        "llm:\n"
        "  provider: ollama\n"
        "  model: qwen2.5:7b\n"
        "  default_temperature: 0.5\n"
        "  max_attempts: 2\n",
        encoding="utf-8",
    )
    return cfg


# Canned valid LLM JSON for one MCQ
_VALID_LLM_RESPONSE = json.dumps({
    "questions": [
        {
            "question_text": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "correct_answers": ["4"],
            "explanation": "Basic arithmetic.",
            "difficulty": "easy",
        }
    ]
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_generates_quiz_end_to_end(tmp_path: Path) -> None:
    """Happy path — pipeline.generate() returns a validated GeneratedQuiz."""
    cfg = _write_minimal_config(tmp_path)
    fake_retriever = _FakeRetriever([_retrieved("id-1"), _retrieved("id-2")])
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we test low-pool warning separately
        quiz = pipeline.generate(
            topic="arithmetic",
            language="en",
            count=1,
            few_shot_count=2,
        )

    assert isinstance(quiz, GeneratedQuiz)
    assert len(quiz.questions) == 1
    assert quiz.questions[0].question_text == "What is 2+2?"
    assert quiz.language == "en"
    assert len(mock_llm.calls) == 1  # one LLM call, no retry needed


def test_pipeline_low_pool_warns_but_proceeds(tmp_path: Path) -> None:
    """Decision 2a: when retrieval returns fewer examples than few_shot_count,
    pipeline warns but still generates."""
    cfg = _write_minimal_config(tmp_path)
    # Only 1 retrieved doc, but few_shot_count=5 → low-pool warning
    fake_retriever = _FakeRetriever([_retrieved("id-only-one")])
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiz = pipeline.generate(
            topic="x",
            language="en",
            count=1,
            few_shot_count=5,
        )

    # Did warn
    assert any("low retrieval pool" in str(w.message).lower() for w in caught)
    # But still produced a valid quiz
    assert isinstance(quiz, GeneratedQuiz)
    assert len(quiz.questions) == 1


def test_pipeline_uses_config_defaults_for_temperature_and_attempts(tmp_path: Path) -> None:
    """Temperature / max_attempts default to config values when not passed."""
    cfg = _write_minimal_config(tmp_path)  # config has temperature=0.5
    fake_retriever = _FakeRetriever([_retrieved("id-1")])
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.generate(topic="x", language="en", count=1, few_shot_count=1)

    # The MockClient saw temperature=0.5 (from config), not the GenerationRequest default 0.75
    assert mock_llm.calls[0]["temperature"] == 0.5


def test_pipeline_multi_levels_warns_about_first_only(tmp_path: Path) -> None:
    """Passing multiple levels triggers the 'using only first level' warning
    on the GenerationRequest side."""
    cfg = _write_minimal_config(tmp_path)
    fake_retriever = _FakeRetriever([_retrieved("id-1")])
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.generate(
            topic="x",
            language="en",
            count=1,
            few_shot_count=1,
            levels=["LEVEL_A", "LEVEL_B"],
        )

    assert any("forwarding only the first" in str(w.message) for w in caught)


def test_render_human_includes_question_and_choices() -> None:
    quiz = GeneratedQuiz.model_validate({
        "language": "en",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "What is 2+2?",
                "choices": ["3", "4"],
                "correct_answers": ["4"],
                "explanation": "Trivial.",
            }
        ],
    })
    out = render_human(quiz, topic="arith")
    assert "What is 2+2?" in out
    assert "4" in out  # the choice
    assert "* 4" in out  # marked as correct
    assert "Trivial." in out


def test_render_json_is_valid_json_with_questions_list() -> None:
    quiz = GeneratedQuiz.model_validate({
        "language": "fr",
        "subject": "MATHEMATICS",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "2+2?",
                "choices": ["3", "4"],
                "correct_answers": ["4"],
            }
        ],
    })
    out = render_json(quiz, topic="arith")
    parsed = json.loads(out)
    assert parsed["topic"] == "arith"
    assert parsed["language"] == "fr"
    assert parsed["subject"] == "MATHEMATICS"
    assert len(parsed["questions"]) == 1
    assert parsed["questions"][0]["correct_answers"] == ["4"]


def test_pipeline_exposes_retrieval_via_last_retrieval(tmp_path: Path) -> None:
    """After generate(), pipeline.last_retrieval holds the chunks the LLM saw."""
    cfg = _write_minimal_config(tmp_path)
    examples = [_retrieved("ex-A"), _retrieved("ex-B"), _retrieved("ex-C")]
    fake_retriever = _FakeRetriever(examples)
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.generate(topic="x", language="en", count=1, few_shot_count=3)

    assert len(pipeline.last_retrieval) == 3
    assert [c.doc_id for c in pipeline.last_retrieval] == ["ex-A", "ex-B", "ex-C"]


def test_pipeline_retrieves_only_once(tmp_path: Path) -> None:
    """The retriever is called exactly once per generate() — no double-fetch."""
    cfg = _write_minimal_config(tmp_path)
    fake_retriever = _FakeRetriever([_retrieved("ex-1")])
    mock_llm = MockClient(canned_response=_VALID_LLM_RESPONSE)

    pipeline = QuizPipeline(
        config_path=cfg,
        _retriever=fake_retriever,
        _llm_client=mock_llm,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.generate(topic="x", language="en", count=1, few_shot_count=1)

    assert len(fake_retriever.calls) == 1


def test_render_retrieval_human_shows_full_text() -> None:
    """Full text by default — no truncation of question or choices."""
    long_q = "What is the integral of f(x) = sin(x) over the interval [0, π]?"
    long_choice = "By integration: ∫₀^π sin(x) dx = [-cos(x)]₀^π = 2"
    chunk = RetrievedQuestion(
        doc_id="full-text-id",
        quiz_id="q-1",
        quiz_title="Calculus integrals",
        language="fr",
        question_type="MULTIPLE_CHOICE",
        question_text=long_q,
        choices=[long_choice, "0", "1", "π"],
        correct_answers=[long_choice],
        subjects=["MATHEMATICS"],
        levels=["HIGH_SCHOOL_4TH_GRADE_MATHEMATICS"],
        multiple_correct_answers=False,
        author_name="Prof",
        author_email=None,
        search_text=long_q,
        metadata={},
        distance=0.123,
    )
    out = render_retrieval_human([chunk], topic="integrals")
    assert long_q in out                  # full question, no truncation
    assert long_choice in out             # full choice text
    assert "* " + long_choice in out      # marked as correct
    assert "Calculus integrals" in out    # title
    assert "MATHEMATICS" in out           # subject visible


def test_render_retrieval_human_handles_empty() -> None:
    out = render_retrieval_human([], topic="x")
    assert "(no chunks retrieved" in out


def test_retrieval_to_dict_preserves_all_fields() -> None:
    chunk = _retrieved("ex-1", question_text="What?")
    rows = retrieval_to_dict([chunk])
    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == "ex-1"
    assert row["question_text"] == "What?"
    assert row["choices"] == ["A", "B", "C", "D"]
    assert row["correct_answers"] == ["A"]
    assert row["distance"] == 0.2


def _quiz_for_save() -> "GeneratedQuiz":
    return GeneratedQuiz.model_validate({
        "language": "en",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "2+2?",
                "choices": ["3", "4"],
                "correct_answers": ["4"],
            }
        ],
    })


def test_save_run_to_file_writes_quiz_and_retrieval(tmp_path: Path) -> None:
    """Saved file contains BOTH retrieval AND quiz — single complete record."""
    target = tmp_path / "last_run.json"
    path = save_run_to_file(
        quiz=_quiz_for_save(),
        retrieval=[_retrieved("ex-1")],
        topic="my topic",
        language="en",
        path=target,
    )
    assert path == target
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["topic"] == "my topic"
    assert data["language"] == "en"
    # Retrieval present
    assert len(data["retrieval"]) == 1
    assert data["retrieval"][0]["doc_id"] == "ex-1"
    # Quiz present
    assert len(data["quiz"]["questions"]) == 1
    assert data["quiz"]["questions"][0]["question_text"] == "2+2?"
    assert data["quiz"]["questions"][0]["correct_answers"] == ["4"]


def test_save_run_to_file_overwrites_previous_run(tmp_path: Path) -> None:
    """Calling twice replaces the file — no append, no accumulation."""
    target = tmp_path / "last_run.json"
    save_run_to_file(
        quiz=_quiz_for_save(),
        retrieval=[_retrieved("first")],
        topic="t1",
        language="en",
        path=target,
    )
    save_run_to_file(
        quiz=_quiz_for_save(),
        retrieval=[_retrieved("second")],
        topic="t2",
        language="en",
        path=target,
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["topic"] == "t2"
    assert data["retrieval"][0]["doc_id"] == "second"


def test_render_json_includes_retrieval_when_passed() -> None:
    quiz = GeneratedQuiz.model_validate({
        "language": "en",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "2+2?",
                "choices": ["3", "4"],
                "correct_answers": ["4"],
            }
        ],
    })
    chunk = _retrieved("ex-1")
    out = render_json(quiz, topic="arith", retrieval=[chunk])
    parsed = json.loads(out)
    assert "retrieval" in parsed
    assert parsed["retrieval"][0]["doc_id"] == "ex-1"


def test_render_json_omits_retrieval_when_not_passed() -> None:
    quiz = GeneratedQuiz.model_validate({
        "language": "en",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "x?",
                "choices": ["A", "B"],
                "correct_answers": ["A"],
            }
        ],
    })
    out = render_json(quiz, topic="x")
    parsed = json.loads(out)
    assert "retrieval" not in parsed


def test_render_json_preserves_unicode() -> None:
    """Arabic and French characters must round-trip cleanly (no \\uXXXX escapes)."""
    quiz = GeneratedQuiz.model_validate({
        "language": "ar",
        "questions": [
            {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": "ما هو ٢+٢؟",
                "choices": ["٣", "٤"],
                "correct_answers": ["٤"],
            }
        ],
    })
    out = render_json(quiz, topic="حساب")
    assert "ما هو" in out  # raw Arabic, not \u-escaped


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
           if n.startswith("test_")]
    for name, fn in fns:
        sig = inspect.signature(fn)
        if "tmp_path" in sig.parameters:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
        else:
            fn()
    print(f"All {len(fns)} pipeline tests passed.")
