"""Tests for the FastAPI HTTP surface (Stage 6 — API endpoint).

Uses FastAPI's TestClient with a fake QuizPipeline injected via app.state.
The lifespan in src.api.server skips the heavy real load when a pipeline
is already attached, so these tests run without ML stack or Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fake pipeline / retriever — match the real surface QuizPipeline exposes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Stand-in with the methods the API endpoints actually call."""

    def __init__(
        self,
        retrieve_results: list[Any] | None = None,
        languages: list[str] | None = None,
        question_types: list[str] | None = None,
        subjects: list[str] | None = None,
        levels: list[str] | None = None,
    ) -> None:
        self.retrieve_results = retrieve_results or []
        self.languages = languages or ["en", "fr", "ar"]
        self.question_types = question_types or ["MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"]
        self.subjects = subjects or ["MATHEMATICS", "PHYSICS", "SCIENCE"]
        self.levels = levels or ["PRIMARY_SCHOOL_6TH_GRADE"]
        self.calls: list[dict] = []

    # API methods used by the endpoints
    def list_languages(self) -> list[str]: return self.languages
    def list_question_types(self) -> list[str]: return self.question_types
    def list_subjects(self) -> list[str]: return self.subjects
    def list_levels(self) -> list[str]: return self.levels

    def retrieve(self, **kwargs) -> list[Any]:
        self.calls.append(kwargs)
        return self.retrieve_results


class _FakeRetrieved:
    """Mimics RetrievedQuestion — just the fields _retrieved_to_dict reads."""

    def __init__(self, doc_id: str = "ex-1") -> None:
        self.doc_id = doc_id
        self.quiz_id = "quiz-1"
        self.quiz_title = "Test Quiz"
        self.language = "en"
        self.question_type = "MULTIPLE_CHOICE"
        self.question_text = "What is X?"
        self.choices = ["A", "B", "C", "D"]
        self.correct_answers = ["A"]
        self.subjects = ["SCIENCE"]
        self.levels = ["PRIMARY_SCHOOL_6TH_GRADE"]
        self.multiple_correct_answers = False
        self.author_name = None
        self.author_email = None
        self.distance = 0.2


class _FakeQuestion:
    """Mimics GeneratedQuestion's `model_dump()` interface."""

    def __init__(self, text: str = "Q?") -> None:
        self.text = text

    def model_dump(self) -> dict:
        return {
            "question_type": "MULTIPLE_CHOICE",
            "question_text": self.text,
            "choices": ["A", "B"],
            "correct_answers": ["A"],
            "multiple_correct_answers": False,
            "explanation": "",
            "difficulty": None,
        }


class _FakeQuiz:
    """Mimics GeneratedQuiz."""

    def __init__(self, language: str = "en", subject: str | None = None,
                 level: str | None = None, n_questions: int = 1) -> None:
        self.language = language
        self.subject = subject
        self.level = level
        self.questions = [_FakeQuestion(f"Q{i}") for i in range(n_questions)]


class _FakePipeline:
    """Stand-in for QuizPipeline. Implements the methods the API uses."""

    def __init__(
        self,
        retriever: _FakeRetriever | None = None,
        quiz: _FakeQuiz | None = None,
        last_retrieval: list[Any] | None = None,
        raise_on_generate: Exception | None = None,
    ) -> None:
        self.retriever = retriever or _FakeRetriever()
        self._quiz = quiz or _FakeQuiz()
        self.last_retrieval = last_retrieval or []
        self._raise = raise_on_generate
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs) -> _FakeQuiz:
        self.generate_calls.append(kwargs)
        if self._raise:
            raise self._raise
        return self._quiz


# ---------------------------------------------------------------------------
# Helper — build a TestClient with a fake pipeline injected
# ---------------------------------------------------------------------------


def _make_client(pipeline: _FakePipeline):
    """Build a TestClient that uses the injected pipeline.

    The lifespan in src.api.server skips real loading when app.state.pipeline
    is already set — that's how we avoid loading BGE-M3 in tests.
    """
    from fastapi.testclient import TestClient
    from src.api.server import app

    app.state.pipeline = pipeline
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_returns_ok_when_pipeline_loaded() -> None:
    client = _make_client(_FakePipeline())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pipeline_loaded"] is True


def test_taxonomy_returns_lists() -> None:
    pipeline = _FakePipeline(
        retriever=_FakeRetriever(
            languages=["en", "fr"],
            subjects=["MATHEMATICS"],
            levels=["L1", "L2"],
            question_types=["MULTIPLE_CHOICE"],
        )
    )
    client = _make_client(pipeline)
    r = client.get("/taxonomy")
    assert r.status_code == 200
    body = r.json()
    assert body["languages"] == ["en", "fr"]
    assert body["subjects"] == ["MATHEMATICS"]
    assert body["levels"] == ["L1", "L2"]
    assert body["question_types"] == ["MULTIPLE_CHOICE"]


def test_retrieve_returns_chunks() -> None:
    pipeline = _FakePipeline(
        retriever=_FakeRetriever(
            retrieve_results=[_FakeRetrieved("a"), _FakeRetrieved("b")]
        )
    )
    client = _make_client(pipeline)
    r = client.post(
        "/retrieve",
        json={"query": "x", "language": "en", "top_k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [x["doc_id"] for x in body["results"]] == ["a", "b"]


def test_retrieve_rejects_missing_required_fields() -> None:
    client = _make_client(_FakePipeline())
    r = client.post("/retrieve", json={"query": "x"})  # missing language
    assert r.status_code == 422  # Pydantic validation


def test_retrieve_rejects_unknown_language() -> None:
    client = _make_client(_FakePipeline())
    r = client.post(
        "/retrieve",
        json={"query": "x", "language": "de"},  # unsupported
    )
    assert r.status_code == 422


def test_generate_returns_quiz_without_retrieval_by_default() -> None:
    pipeline = _FakePipeline(
        quiz=_FakeQuiz(language="fr", subject="MATHEMATICS", n_questions=3),
        last_retrieval=[_FakeRetrieved("ex-1")],
    )
    client = _make_client(pipeline)
    r = client.post(
        "/quiz/generate",
        json={
            "topic": "primitives",
            "language": "fr",
            "count": 3,
            "subject": "MATHEMATICS",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "primitives"
    assert body["language"] == "fr"
    assert body["subject"] == "MATHEMATICS"
    assert len(body["questions"]) == 3
    assert "retrieval" not in body  # default include_retrieval=False


def test_generate_includes_retrieval_when_requested() -> None:
    pipeline = _FakePipeline(
        quiz=_FakeQuiz(n_questions=2),
        last_retrieval=[_FakeRetrieved("ex-1"), _FakeRetrieved("ex-2")],
    )
    client = _make_client(pipeline)
    r = client.post(
        "/quiz/generate",
        json={
            "topic": "x",
            "language": "en",
            "count": 2,
            "include_retrieval": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "retrieval" in body
    assert len(body["retrieval"]) == 2
    assert body["retrieval"][0]["doc_id"] == "ex-1"


def test_generate_validates_count_bounds() -> None:
    """count must be between 1 and 20."""
    client = _make_client(_FakePipeline())
    r = client.post(
        "/quiz/generate",
        json={"topic": "x", "language": "en", "count": 0},
    )
    assert r.status_code == 422

    r = client.post(
        "/quiz/generate",
        json={"topic": "x", "language": "en", "count": 100},
    )
    assert r.status_code == 422


def test_generate_returns_502_on_generation_error() -> None:
    """When the pipeline raises GenerationError, return 502 not 500."""
    from src.generation.generator import GenerationError
    pipeline = _FakePipeline(
        raise_on_generate=GenerationError("Retriever returned 0 examples"),
    )
    client = _make_client(pipeline)
    r = client.post(
        "/quiz/generate",
        json={"topic": "x", "language": "en", "count": 1},
    )
    assert r.status_code == 502
    assert "Retriever returned 0 examples" in r.json()["detail"]


def test_generate_returns_400_on_value_error() -> None:
    """ValueError → 400 (bad input, not server failure)."""
    pipeline = _FakePipeline(
        raise_on_generate=ValueError("unknown subject 'XYZ'"),
    )
    client = _make_client(pipeline)
    r = client.post(
        "/quiz/generate",
        json={"topic": "x", "language": "en", "count": 1, "subject": "XYZ"},
    )
    assert r.status_code == 400


def test_generate_forbids_extra_fields() -> None:
    """ConfigDict(extra='forbid') means typos aren't silently ignored."""
    client = _make_client(_FakePipeline())
    r = client.post(
        "/quiz/generate",
        json={"topic": "x", "language": "en", "count": 1, "tipo": "MCQ"},
    )
    assert r.status_code == 422


def test_generate_rejects_tuning_knobs_in_request() -> None:
    """`temperature`, `few_shot_count`, `max_attempts` were intentionally
    moved to configs/models.yaml. They must NOT be accepted as request
    fields — sending them should fail validation, not silently override
    the config."""
    client = _make_client(_FakePipeline())
    for forbidden in ("temperature", "few_shot_count", "max_attempts"):
        r = client.post(
            "/quiz/generate",
            json={
                "topic": "x",
                "language": "en",
                "count": 1,
                forbidden: 0.5 if forbidden == "temperature" else 3,
            },
        )
        assert r.status_code == 422, (
            f"Expected 422 for forbidden field {forbidden!r}, got {r.status_code}"
        )


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
           if n.startswith("test_")]
    for name, fn in fns:
        fn()
    print(f"All {len(fns)} API tests passed.")
