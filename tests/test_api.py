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
# Hermetic environment
# ---------------------------------------------------------------------------
# docker-compose injects the REAL runtime configuration into the container
# (API_KEYS from .env, RATE_LIMIT_PER_MINUTE, CORS_ALLOW_ORIGINS...). Without
# this scrub the suite behaves differently depending on whose machine and
# which .env it runs under: enabling auth in production turned every
# pre-existing endpoint test into a 401.
#
# Tests that care about these settings opt in explicitly via `_env(...)`.
# This runs before src.api.server is first imported (the server import is
# lazy, inside _make_client), which matters because the CORS middleware is
# installed at import time from the environment.
import os as _os_bootstrap

for _leaky in (
    "API_KEYS",
    "RATE_LIMIT_PER_MINUTE",
    "CORS_ALLOW_ORIGINS",
    "INCLUDE_AUTHOR_METADATA",
    "RUNS_LOG_MAX_BYTES",
    "LOG_RUNS",
):
    _os_bootstrap.environ.pop(_leaky, None)


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

    def generate_detailed(self, **kwargs):
        """What the API actually calls. Returns quiz+retrieval+timings
        together, so nothing is read back off shared instance state."""
        from src.pipeline.orchestrator import GenerationResult
        quiz = self.generate(**kwargs)
        return GenerationResult(
            quiz=quiz,
            retrieval=self.last_retrieval,
            timings={"retrieve_seconds": 0.1, "generate_seconds": 0.2,
                     "total_seconds": 0.3, "n_examples_used": len(self.last_retrieval)},
        )


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


# GenerationError covers two situations that are NOT the same fault, and the
# API must tell them apart:
#   * nothing retrieved  -> the caller asked for something we don't have (400)
#   * LLM gave up        -> genuine upstream failure (502)
# Both bodies stay sanitised: str(exc) embeds diagnose_empty() output, which
# is corpus intelligence (size, per-subject language counts), not something a
# caller needs.

_CORPUS_LEAK = (
    "Retriever returned 0 examples for topic='x'. Diagnostic: "
    "Total rows in corpus: 5,782; subject='MATHEMATICS' fr: 1,018"
)


def _assert_no_corpus_internals(detail: str) -> None:
    assert "request_id=" in detail          # the thread back to the full log line
    assert "5,782" not in detail
    assert "Diagnostic" not in detail
    assert "MATHEMATICS" not in detail
    assert "Retriever returned" not in detail


def test_generate_returns_400_when_nothing_was_retrieved() -> None:
    """Empty retrieval is the caller's fault, so 400 — and Cloudflare passes
    4xx bodies through unchanged, so the hint actually reaches the user."""
    from src.generation.generator import GenerationError

    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"):
        client = _make_client(
            _FakePipeline(raise_on_generate=GenerationError(_CORPUS_LEAK))
        )
        r = client.post(
            "/quiz/generate",
            json={"topic": "x", "language": "en", "count": 1},
        )

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "No matching content" in detail    # actionable
    _assert_no_corpus_internals(detail)


def test_generate_returns_502_when_the_llm_gives_up() -> None:
    """Context existed, the model still couldn't produce a valid quiz —
    that's upstream, not the caller."""
    from src.generation.generator import GenerationError

    exhausted = (
        "Generation failed after 3 attempts. Last error: Question 0 failed "
        "validation: correct_answer 'x' not found verbatim in choices"
    )
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"):
        client = _make_client(
            _FakePipeline(raise_on_generate=GenerationError(exhausted))
        )
        r = client.post(
            "/quiz/generate",
            json={"topic": "x", "language": "en", "count": 1},
        )

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "Generation failed" in detail
    assert "request_id=" in detail
    # Internal validation chatter stays in the log.
    assert "verbatim" not in detail
    assert "attempts" not in detail


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



# ---------------------------------------------------------------------------
# Phase 1 hardening — auth, rate limiting, correlation IDs, error opacity
# ---------------------------------------------------------------------------

import contextlib
import os as _os

from src.api.security import reset_rate_limits


@contextlib.contextmanager
def _env(**overrides: str | None):
    """Temporarily set/unset environment variables (script-runner friendly)."""
    previous = {k: _os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_auth_disabled_when_api_keys_unset() -> None:
    """Dev default stays frictionless — no key required."""
    with _env(API_KEYS=None):
        client = _make_client(_FakePipeline())
        assert client.get("/taxonomy").status_code == 200


def test_missing_api_key_rejected_when_configured() -> None:
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        r = client.post("/quiz/generate", json={"topic": "x", "language": "en"})
        assert r.status_code == 401
        assert "API key" in r.json()["detail"]


def test_invalid_api_key_rejected() -> None:
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        r = client.get("/taxonomy", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401


def test_valid_api_key_accepted_via_header() -> None:
    with _env(API_KEYS="sekret-1,sekret-2"):
        client = _make_client(_FakePipeline())
        assert client.get("/taxonomy", headers={"X-API-Key": "sekret-2"}).status_code == 200


def test_valid_api_key_accepted_as_bearer_token() -> None:
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        r = client.get("/taxonomy", headers={"Authorization": "Bearer sekret-1"})
        assert r.status_code == 200


def test_health_stays_open_without_a_key() -> None:
    """The Docker healthcheck calls /health with no credentials — if this
    ever requires a key the container flips to unhealthy and restarts."""
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["pipeline_loaded"] is True


def test_rate_limit_returns_429_with_retry_after() -> None:
    reset_rate_limits()
    try:
        with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="2"):
            client = _make_client(_FakePipeline())
            body = {"query": "x", "language": "en"}
            assert client.post("/retrieve", json=body).status_code == 200
            assert client.post("/retrieve", json=body).status_code == 200
            blocked = client.post("/retrieve", json=body)
            assert blocked.status_code == 429
            assert int(blocked.headers["Retry-After"]) >= 1
            assert "Rate limit exceeded" in blocked.json()["detail"]
    finally:
        reset_rate_limits()


def test_rate_limit_disabled_when_zero() -> None:
    reset_rate_limits()
    try:
        with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"):
            client = _make_client(_FakePipeline())
            for _ in range(5):
                r = client.post("/retrieve", json={"query": "x", "language": "en"})
                assert r.status_code == 200
    finally:
        reset_rate_limits()


def test_request_id_generated_and_echoed() -> None:
    with _env(API_KEYS=None):
        client = _make_client(_FakePipeline())
        r = client.get("/health")
        assert r.headers.get("X-Request-ID")
        assert len(r.headers["X-Request-ID"]) >= 8


def test_inbound_request_id_is_preserved() -> None:
    """A trace started at Nginx / the platform survives through this service."""
    with _env(API_KEYS=None):
        client = _make_client(_FakePipeline())
        r = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
        assert r.headers["X-Request-ID"] == "trace-abc-123"


def test_unhandled_exception_body_is_opaque() -> None:
    """A 500 must not hand the caller exception text — messages carry file
    paths, config values and SDK internals."""
    from fastapi.testclient import TestClient
    from src.api.server import app

    leaky = RuntimeError("/app/configs/models.yaml exploded with key sk-live-XYZ")
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"):
        app.state.pipeline = _FakePipeline(raise_on_generate=leaky)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/quiz/generate", json={"topic": "x", "language": "en"})

        assert r.status_code == 500
        detail = r.json()["detail"]
        assert "sk-live-XYZ" not in detail
        assert "models.yaml" not in detail
        assert "RuntimeError" not in detail
        assert "request_id=" in detail



# ---------------------------------------------------------------------------
# Phase 2 hardening — PII exposure and run-log rotation
# ---------------------------------------------------------------------------

def test_author_pii_absent_from_retrieval_by_default() -> None:
    """author_name / author_email identify real teachers who wrote the source
    corpus. They must not ship to callers unless explicitly re-enabled."""
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0", INCLUDE_AUTHOR_METADATA=None):
        pipeline = _FakePipeline(last_retrieval=[_FakeRetrieved("ex-1")])
        client = _make_client(pipeline)
        r = client.post(
            "/quiz/generate",
            json={"topic": "x", "language": "en", "count": 1,
                  "include_retrieval": True},
        )
        assert r.status_code == 200
        chunk = r.json()["retrieval"][0]
        assert "author_name" not in chunk
        assert "author_email" not in chunk
        # The useful debugging fields are still there.
        assert "doc_id" in chunk and "distance" in chunk


def test_author_pii_restorable_via_env() -> None:
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0", INCLUDE_AUTHOR_METADATA="1"):
        pipeline = _FakePipeline(last_retrieval=[_FakeRetrieved("ex-1")])
        client = _make_client(pipeline)
        r = client.post(
            "/quiz/generate",
            json={"topic": "x", "language": "en", "count": 1,
                  "include_retrieval": True},
        )
        chunk = r.json()["retrieval"][0]
        assert "author_name" in chunk
        assert "author_email" in chunk


def test_run_log_rotates_past_the_size_cap() -> None:
    """runs.jsonl grew forever. One line with 12 chunks is 20-40 KB."""
    import tempfile
    from pathlib import Path as _Path

    from src.api import server as _server

    original = _server.RUNS_LOG_PATH
    with tempfile.TemporaryDirectory() as td:
        log = _Path(td) / "runs.jsonl"
        _server.RUNS_LOG_PATH = log
        try:
            with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0",
                      LOG_RUNS="1", RUNS_LOG_MAX_BYTES="10"):
                client = _make_client(_FakePipeline())
                body = {"topic": "x", "language": "en", "count": 1}

                assert client.post("/quiz/generate", json=body).status_code == 200
                assert log.exists()
                assert not log.with_suffix(".jsonl.1").exists()  # nothing to roll yet

                assert client.post("/quiz/generate", json=body).status_code == 200
                assert log.with_suffix(".jsonl.1").exists(), "log never rotated"
                assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
        finally:
            _server.RUNS_LOG_PATH = original


def test_run_log_rotation_disabled_at_zero() -> None:
    import tempfile
    from pathlib import Path as _Path

    from src.api import server as _server

    original = _server.RUNS_LOG_PATH
    with tempfile.TemporaryDirectory() as td:
        log = _Path(td) / "runs.jsonl"
        _server.RUNS_LOG_PATH = log
        try:
            with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0",
                      LOG_RUNS="1", RUNS_LOG_MAX_BYTES="0"):
                client = _make_client(_FakePipeline())
                body = {"topic": "x", "language": "en", "count": 1}
                for _ in range(3):
                    client.post("/quiz/generate", json=body)
                assert not log.with_suffix(".jsonl.1").exists()
                assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 3
        finally:
            _server.RUNS_LOG_PATH = original



# ---------------------------------------------------------------------------
# Phase 3 hardening — readiness, metrics, CORS configuration
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self, n: int = 5) -> None:
        self._n = n

    def count(self) -> int:
        return self._n


def _ready_pipeline(*, docs: int = 5, payload: bool = True, llm: bool = True):
    """A fake whose retriever looks like a loaded one."""
    pipeline = _FakePipeline()
    pipeline.retriever._collection = _FakeCollection(docs)
    pipeline.retriever._payload = {"doc-1": {}} if payload else {}
    pipeline.llm_client = object() if llm else None
    return pipeline


def test_ready_returns_200_when_everything_is_loaded() -> None:
    with _env(API_KEYS=None):
        client = _make_client(_ready_pipeline())
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        assert all(r.json()["checks"].values())


def test_ready_reports_503_when_vector_store_is_empty() -> None:
    """The failure /health cannot see: pipeline allocated, index empty.
    Every request would 502; readiness must say so."""
    with _env(API_KEYS=None):
        client = _make_client(_ready_pipeline(docs=0))
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"
        assert r.json()["checks"]["pipeline_loaded"] is True
        assert r.json()["checks"]["vector_store"] is False


def test_ready_reports_503_when_payload_missing() -> None:
    with _env(API_KEYS=None):
        client = _make_client(_ready_pipeline(payload=False))
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["payload"] is False


def test_ready_survives_a_throwing_collection() -> None:
    """A probe that raises must degrade, not 500."""
    class _Exploding:
        def count(self):
            raise RuntimeError("chroma is gone")

    with _env(API_KEYS=None):
        pipeline = _ready_pipeline()
        pipeline.retriever._collection = _Exploding()
        client = _make_client(pipeline)
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["vector_store"] is False


def test_ready_is_open_without_a_key() -> None:
    """The container healthcheck calls it with no credentials."""
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_ready_pipeline())
        assert client.get("/ready").status_code == 200


def test_metrics_requires_the_api_key() -> None:
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        assert client.get("/metrics").status_code == 401
        r = client.get("/metrics", headers={"X-API-Key": "sekret-1"})
        assert r.status_code == 200
        assert "quiz_api_up 1" in r.text


def test_metrics_counts_requests_and_excludes_noise() -> None:
    from src.api.observability import reset_metrics

    reset_metrics()
    try:
        with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"):
            client = _make_client(_ready_pipeline())
            client.post("/retrieve", json={"query": "x", "language": "en"})
            client.get("/health")            # excluded — healthcheck noise
            body = client.get("/metrics").text

        assert 'path="/retrieve",status="200"' in body
        assert 'quiz_api_request_duration_seconds_count{path="/retrieve"} 1' in body
        assert '"/health"' not in body       # healthcheck must not dominate
        assert '"/metrics"' not in body      # a scrape must not count itself
    finally:
        reset_metrics()


def test_metrics_records_rate_limit_events() -> None:
    from src.api.observability import reset_metrics

    reset_metrics()
    reset_rate_limits()
    try:
        with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="1"):
            client = _make_client(_FakePipeline())
            body = {"query": "x", "language": "en"}
            client.post("/retrieve", json=body)
            assert client.post("/retrieve", json=body).status_code == 429
            text = client.get("/metrics").text
        assert 'quiz_api_events_total{reason="rate_limited"} 1' in text
    finally:
        reset_metrics()
        reset_rate_limits()


def test_cors_origins_parsing() -> None:
    """CORS middleware is installed at import time, so the unit under test is
    the configuration reader."""
    from src.api.security import configured_cors_origins

    with _env(CORS_ALLOW_ORIGINS=None):
        assert configured_cors_origins() == []          # server-to-server default
    with _env(CORS_ALLOW_ORIGINS="https://school.tn, https://admin.school.tn"):
        assert configured_cors_origins() == ["https://school.tn",
                                             "https://admin.school.tn"]
    with _env(CORS_ALLOW_ORIGINS="  "):
        assert configured_cors_origins() == []



# ---------------------------------------------------------------------------
# Teacher-facing UI
# ---------------------------------------------------------------------------

def test_ui_is_served_without_a_key() -> None:
    """A browser navigating to /ui cannot send X-API-Key. If this ever needs
    auth the page becomes unreachable and the whole UI is dead."""
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        r = client.get("/ui")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "<title>" in r.text


def test_ui_page_carries_no_secrets() -> None:
    """The page is inert — it must never be built with a key baked in."""
    with _env(API_KEYS="sekret-1"):
        client = _make_client(_FakePipeline())
        body = client.get("/ui").text
        assert "sekret-1" not in body
        # And it must not persist a key the user types.
        assert "localStorage" not in body
        assert "sessionStorage" not in body


def test_root_redirects_to_the_ui() -> None:
    with _env(API_KEYS=None):
        client = _make_client(_FakePipeline())
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (302, 307)
        assert r.headers["location"] == "/ui"


def test_ui_is_absent_from_the_openapi_schema() -> None:
    """/ui and / are page routes, not API surface — they'd be noise in the
    contract the platform team reads."""
    with _env(API_KEYS=None):
        client = _make_client(_FakePipeline())
        paths = client.get("/openapi.json").json()["paths"]
        assert "/ui" not in paths
        assert "/" not in paths
        assert "/quiz/generate" in paths



# ---------------------------------------------------------------------------
# Feedback — the labelled generation set
# ---------------------------------------------------------------------------

import contextlib as _ctx2


@_ctx2.contextmanager
def _feedback_log():
    """Point FEEDBACK_LOG_PATH at a temp file for the duration."""
    import tempfile
    from pathlib import Path as _Path

    from src.api import server as _server

    original = _server.FEEDBACK_LOG_PATH
    with tempfile.TemporaryDirectory() as td:
        path = _Path(td) / "feedback.jsonl"
        _server.FEEDBACK_LOG_PATH = path
        try:
            yield path
        finally:
            _server.FEEDBACK_LOG_PATH = original


def test_feedback_appends_one_row_per_judgement() -> None:
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"), _feedback_log() as log:
        client = _make_client(_FakePipeline())
        r = client.post("/feedback", json={
            "verdict": "down",
            "question_text": "Which sentence is passive?",
            "question_index": 2,
            "request_id": "req-abc123",
            "topic": "passive voice",
            "language": "en",
            "subject": "ENGLISH",
            "school_phase": "HIGH",
            "note": "all three choices are active",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["verdict"] == "down"
        assert row["question_index"] == 2
        assert row["note"] == "all three choices are active"
        assert "timestamp" in row
        # The join key back to the full run (filters, retrieval, timings).
        assert row["request_id"] == "req-abc123"


def test_feedback_falls_back_to_the_current_request_id() -> None:
    """If the client forgets to send one, still record something joinable."""
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"), _feedback_log() as log:
        client = _make_client(_FakePipeline())
        client.post("/feedback", json={"verdict": "up", "question_text": "Q?"})
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert row["request_id"]


def test_feedback_rejects_a_bad_verdict() -> None:
    with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="0"), _feedback_log():
        client = _make_client(_FakePipeline())
        r = client.post("/feedback", json={"verdict": "maybe", "question_text": "Q?"})
        assert r.status_code == 422


def test_feedback_requires_the_api_key() -> None:
    with _env(API_KEYS="sekret-1"), _feedback_log():
        client = _make_client(_FakePipeline())
        r = client.post("/feedback", json={"verdict": "up", "question_text": "Q?"})
        assert r.status_code == 401


def test_feedback_is_not_rate_limited() -> None:
    """Throttling the one signal we want more of would be perverse."""
    reset_rate_limits()
    try:
        with _env(API_KEYS=None, RATE_LIMIT_PER_MINUTE="1"), _feedback_log() as log:
            client = _make_client(_FakePipeline())
            for i in range(4):
                r = client.post("/feedback",
                                json={"verdict": "up", "question_text": f"Q{i}"})
                assert r.status_code == 200
            assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 4
    finally:
        reset_rate_limits()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
           if n.startswith("test_")]
    for name, fn in fns:
        fn()
    print(f"All {len(fns)} API tests passed.")
