"""FastAPI app exposing the QuizPipeline over HTTP.

Endpoints:
    GET  /health         — liveness check (cheap, doesn't touch the LLM)
    GET  /taxonomy       — legal subjects/levels/languages (for UI dropdowns)
    POST /retrieve       — retrieval only, no LLM call
    POST /quiz/generate  — full retrieve + LLM + validate + retry

Pipeline lifetime: built ONCE at app startup via the lifespan context
manager (Decision 2a). First request after startup is fast; the 30s of
embedder + reranker + Ollama warmup happens once when you launch the
server.

Tests can short-circuit the heavy load by setting `app.state.pipeline`
*before* the lifespan runs — the lifespan only loads if no pipeline is
already attached.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    RetrieveRequest,
)


logger = logging.getLogger("quiz_api")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Lifespan — build the pipeline once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the QuizPipeline at startup, reuse for every request.

    If `app.state.pipeline` is already set (e.g. a test injected a fake),
    skip the heavy real load — this is the test-injection hook.
    """
    if getattr(app.state, "pipeline", None) is None:
        logger.info("Loading QuizPipeline (BGE-M3 + reranker + Ollama warmup)…")
        from src.pipeline import QuizPipeline
        app.state.pipeline = QuizPipeline()
        logger.info("Pipeline loaded.")
    else:
        logger.info("Using pre-injected pipeline (skipping real load).")

    yield

    # No teardown needed — process exit cleans up.


app = FastAPI(
    title="Quiz Generator API",
    description="HTTP surface over the multilingual RAG quiz pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieved_to_dict(c: Any) -> dict:
    """Serialise a RetrievedQuestion-like object into a JSON-safe dict."""
    return {
        "doc_id": c.doc_id,
        "quiz_id": c.quiz_id,
        "quiz_title": c.quiz_title,
        "language": c.language,
        "question_type": c.question_type,
        "question_text": c.question_text,
        "choices": list(c.choices or []),
        "correct_answers": list(c.correct_answers or []),
        "subjects": list(c.subjects or []),
        "levels": list(c.levels or []),
        "multiple_correct_answers": c.multiple_correct_answers,
        "author_name": c.author_name,
        "author_email": c.author_email,
        "distance": c.distance,
    }


def _get_pipeline(request: Request) -> Any:
    """Pull the loaded pipeline from app state, or 503 if not ready."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. The server is still starting.",
        )
    return pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    """Liveness probe. Cheap — does not call the LLM."""
    pipeline = getattr(request.app.state, "pipeline", None)
    return {
        "status": "ok" if pipeline is not None else "loading",
        "pipeline_loaded": pipeline is not None,
    }


@app.get("/taxonomy")
def taxonomy(request: Request) -> dict:
    """Return the legal values for the filterable fields. Used by the
    platform's UI to populate dropdowns without hard-coding values."""
    p = _get_pipeline(request)
    return {
        "languages": p.retriever.list_languages(),
        "question_types": p.retriever.list_question_types(),
        "subjects": p.retriever.list_subjects(),
        "levels": p.retriever.list_levels(),
    }


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, request: Request) -> dict:
    """Retrieval only — no LLM. Useful for the platform to debug whether a
    bad output is a retrieval problem or a generation problem."""
    p = _get_pipeline(request)
    try:
        results = p.retriever.retrieve(
            query=req.query,
            language=req.language,
            top_k=req.top_k,
            question_type=req.question_type,
            subject=req.subject,
            levels=req.levels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "query": req.query,
        "language": req.language,
        "count": len(results),
        "results": [_retrieved_to_dict(c) for c in results],
    }


@app.post("/quiz/generate")
def generate_quiz(req: GenerateRequest, request: Request) -> dict:
    """Full retrieve → generate → validate → retry pipeline.

    Returns the GeneratedQuiz JSON. Adds a `retrieval` field when the
    caller passes `include_retrieval=true`.
    """
    p = _get_pipeline(request)

    # Lazy import — keeps the module loadable in tests that don't have ML deps.
    from src.generation.generator import GenerationError

    # `temperature`, `max_attempts`, and `few_shot_count` come from
    # configs/models.yaml — not from the request. The pipeline reads the
    # config defaults when these are not passed.
    try:
        quiz = p.generate(
            topic=req.topic,
            language=req.language,
            count=req.count,
            question_type=req.question_type,
            subject=req.subject,
            levels=req.levels,
        )
    except ValueError as exc:
        # Bad inputs (e.g. unknown subject/level for taxonomy)
        raise HTTPException(status_code=400, detail=str(exc))
    except GenerationError as exc:
        # Retriever returned 0 examples, or LLM exhausted retries
        raise HTTPException(
            status_code=502,
            detail=f"Generation failed: {exc}",
        )

    response: dict = {
        "topic": req.topic,
        "language": quiz.language,
        "subject": quiz.subject,
        "level": quiz.level,
        "questions": [q.model_dump() for q in quiz.questions],
    }
    if req.include_retrieval:
        response["retrieval"] = [
            _retrieved_to_dict(c) for c in p.last_retrieval
        ]
    return response


# ---------------------------------------------------------------------------
# Generic exception fallback — anything we didn't anticipate
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler so the server never leaks stack traces.

    Logs the full traceback server-side; returns a uniform error envelope.
    """
    logger.exception("Unhandled exception while serving %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_error",
            detail=f"{type(exc).__name__}: {exc}",
        ).model_dump(),
    )
