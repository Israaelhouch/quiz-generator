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

import datetime as _dt
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from src.api.schemas import (
    ErrorResponse,
    FeedbackRequest,
    GenerateRequest,
    HealthResponse,
    RetrieveRequest,
)
from src.api.observability import record_event, record_request, render_prometheus
from src.api.security import (
    REQUEST_ID_HEADER,
    configured_cors_origins,
    enforce_rate_limit,
    log_security_posture,
    require_api_key,
)
from src.shared.logging_setup import request_id_ctx


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
    log_security_posture()
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


_cors_origins = configured_cors_origins()
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,          # we authenticate by header, not cookie
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization",
                       REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------
# Every incoming HTTP request gets a short unique ID, attached to:
#   - the `request_id_ctx` ContextVar so all log records picked up that
#     ID via the LogRecord factory in src/shared/logging_setup.py
#   - the `X-Request-Id` response header so the caller can echo it
#     back when reporting a problem ("call at 14:23 with X-Request-Id
#     'req-7a3f2b' failed")
#
# If the caller sets `X-Request-Id` on the request, we honour it instead
# of generating one. This lets the platform team correlate IDs across
# their layers and ours.


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Stamp every request with a short unique ID, plumb it to logs, and
    record metrics for it.

    Metrics ride along here so every route is covered without decorating each
    one. /metrics excludes itself so a scrape can't inflate its own numbers,
    and /health is excluded because the container healthcheck fires every 30s
    and would otherwise dominate every series.
    """
    import time as _t

    incoming = request.headers.get(REQUEST_ID_HEADER)
    rid = incoming if incoming else f"req-{uuid4().hex[:8]}"

    started = _t.perf_counter()
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)

    response.headers[REQUEST_ID_HEADER] = rid

    path = request.url.path
    if path not in ("/metrics", "/health"):
        record_request(
            method=request.method,
            path=path,
            status=response.status_code,
            duration=_t.perf_counter() - started,
        )
        if response.status_code == 429:
            record_event("rate_limited")
        elif response.status_code == 401:
            record_event("unauthorized")
        elif response.status_code == 502:
            record_event("generation_failed")
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _include_author_metadata() -> bool:
    """Whether to expose the original quiz author's name + email.

    These identify real teachers who authored the source corpus. They were
    handy while debugging retrieval, but they ship to the caller on every
    include_retrieval=true response AND into every runs.jsonl line, which is
    personal data accumulating in an unrotated file. Off by default; set
    INCLUDE_AUTHOR_METADATA=1 to restore the old behaviour.
    """
    return os.environ.get("INCLUDE_AUTHOR_METADATA", "0") == "1"


def _retrieved_to_dict(c: Any) -> dict:
    """Serialise a RetrievedQuestion-like object into a JSON-safe dict."""
    payload = {
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
        "distance": c.distance,
    }
    if _include_author_metadata():
        payload["author_name"] = c.author_name
        payload["author_email"] = c.author_email
    return payload


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
# Append-only run log
# ---------------------------------------------------------------------------
# Every successful /quiz/generate call appends one JSON line to this file.
# Useful for testing — you accumulate every query+result pair in one place.
# Set LOG_RUNS=0 in the environment to disable.

RUNS_LOG_PATH = Path(os.environ.get("RUNS_LOG_PATH", "/app/logs/runs.jsonl"))

# Size-based rotation. One /quiz/generate line with 12 retrieved chunks runs
# 20-40 KB, so an unrotated file fills a disk quietly over a few months.
# runs.jsonl -> runs.jsonl.1 -> ... -> runs.jsonl.<KEEP>, oldest discarded.
# Set RUNS_LOG_MAX_BYTES=0 to disable.
# Human judgements about generated questions. Separate file from runs.jsonl:
# runs are machine-generated and voluminous, feedback is scarce and precious.
FEEDBACK_LOG_PATH = Path(
    os.environ.get("FEEDBACK_LOG_PATH", "/app/logs/feedback.jsonl")
)

DEFAULT_RUNS_LOG_MAX_BYTES = 50 * 1024 * 1024
RUNS_LOG_KEEP = 3


def _runs_log_max_bytes() -> int:
    raw = os.environ.get("RUNS_LOG_MAX_BYTES")
    if raw is None:
        return DEFAULT_RUNS_LOG_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RUNS_LOG_MAX_BYTES


def _rotate_runs_log_if_needed(path: Path) -> None:
    """Roll the log when it outgrows the cap. Best-effort, never raises."""
    limit = _runs_log_max_bytes()
    if limit <= 0:
        return
    try:
        if not path.exists() or path.stat().st_size < limit:
            return
        oldest = path.with_suffix(path.suffix + f".{RUNS_LOG_KEEP}")
        if oldest.exists():
            oldest.unlink()
        for i in range(RUNS_LOG_KEEP - 1, 0, -1):
            src = path.with_suffix(path.suffix + f".{i}")
            if src.exists():
                src.replace(path.with_suffix(path.suffix + f".{i + 1}"))
        path.replace(path.with_suffix(path.suffix + ".1"))
        logger.info("Rotated run log at %d bytes", limit)
    except Exception as exc:                                # never break the API
        logger.warning("Run-log rotation failed: %s", exc)


def _append_run_log(*, request_dict: dict, response_dict: dict) -> None:
    """Append one timestamped run to the JSONL log file. Best-effort.

    Each entry includes the current `request_id` so a log line in
    stderr and a row in runs.jsonl can be cross-referenced — handy
    when a teacher reports a bad output and you only have the
    timestamp + request ID from the platform's records.
    """
    if os.environ.get("LOG_RUNS", "1") != "1":
        return
    try:
        RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _rotate_runs_log_if_needed(RUNS_LOG_PATH)
        entry = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id_ctx.get(),
            "request": request_dict,
            "response": response_dict,
        }
        with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:                                # don't break the API on logging failure
        logger.warning("Failed to append run log: %s", exc)


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


# ---------------------------------------------------------------------------
# Teacher-facing UI
# ---------------------------------------------------------------------------
# One self-contained page, served same-origin so no CORS is involved and the
# browser can reach the API with a relative path. It lives under src/ so the
# Dockerfile's existing `COPY src/` picks it up — no build step, no node.

_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Bare host lands on the UI rather than a 404."""
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the quiz UI.

    Unauthenticated ON PURPOSE: a browser navigating to a URL cannot attach an
    X-API-Key header, so requiring one here would make the page unreachable.
    The HTML itself is inert — it holds no key and no data. The API calls it
    makes are authenticated normally, and the page prompts for a key if one
    comes back 401.
    """
    if not _UI_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="UI not bundled in this build (src/api/static/index.html missing).",
        )
    # no-cache: the page is tiny and this avoids a stale UI after a rebuild.
    return FileResponse(
        _UI_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.post(
    "/feedback",
    dependencies=[Depends(require_api_key)],
)
def feedback(req: FeedbackRequest, request: Request) -> dict:
    """Record one human judgement about one generated question.

    This is the missing measurement. eval/RESULTS.md has real retrieval
    baselines but says outright that generation quality is judged manually and
    nothing is written down — which makes every prompt or threshold change
    unfalsifiable. Appending judgements as they happen turns normal use into a
    labelled set.

    Not rate-limited: it costs a file append, and throttling the one signal we
    want more of would be perverse.
    """
    entry = req.model_dump()
    entry["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
    # Fall back to THIS call's id only as a last resort — the useful one is the
    # generate call's id, which the client should send.
    entry["request_id"] = req.request_id or request_id_ctx.get()

    try:
        FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _rotate_runs_log_if_needed(FEEDBACK_LOG_PATH)
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to append feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not record feedback.")

    record_event(f"feedback_{req.verdict}")
    return {"ok": True, "request_id": entry["request_id"]}


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness probe — actually exercises the dependencies.

    /health answers "is the process alive". This answers "can it serve a
    request", which is a different question: the pipeline object can exist
    while the Chroma collection is empty or the payload failed to load, and
    every request would then 502. Returns 503 when degraded so an
    orchestrator can hold traffic back.
    """
    p = getattr(request.app.state, "pipeline", None)
    checks = {
        "pipeline_loaded": p is not None,
        "vector_store": False,
        "payload": False,
        "llm_client": False,
    }
    if p is not None:
        retriever = getattr(p, "retriever", None)
        try:
            collection = getattr(retriever, "_collection", None)
            checks["vector_store"] = bool(collection is not None and collection.count() > 0)
        except Exception as exc:
            logger.warning("Readiness: vector store unreachable: %s", exc)
        try:
            checks["payload"] = bool(getattr(retriever, "_payload", None))
        except Exception as exc:
            logger.warning("Readiness: payload unreadable: %s", exc)
        checks["llm_client"] = getattr(p, "llm_client", None) is not None

    ready_now = all(checks.values())
    return JSONResponse(
        status_code=200 if ready_now else 503,
        content={"status": "ready" if ready_now else "degraded", "checks": checks},
    )


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> Response:
    """Prometheus exposition. Behind the same API key as everything else —
    request counts and latencies are operational intelligence."""
    return Response(content=render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/taxonomy", dependencies=[Depends(require_api_key)])
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


def _validate_taxonomy_inputs(
    pipeline: Any,
    subject: str | None = None,
    levels: list[str] | None = None,
) -> None:
    """Reject inputs that don't exist in the corpus taxonomy BEFORE we spend
    any retrieval/LLM time. Two reasons:

    1. Catches placeholder values like 'string' (from the Swagger UI sample)
       so users see a clear, actionable error instead of "Retriever returned
       0 examples" 30 seconds later.
    2. Returns HTTP 400 — which Cloudflare lets pass through unchanged —
       instead of HTTP 502 from a downstream-detected match failure, which
       Cloudflare replaces with its own generic error page.

    Languages, question types, and school_phase are already validated at
    the schema level via Pydantic Literal types; this function covers the
    config-driven fields (subject, levels) that can't be Literal-typed.
    """
    if subject is not None:
        known_subjects = set(pipeline.retriever.list_subjects())
        if subject.strip().upper() not in known_subjects:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown subject {subject!r}. Valid subjects: "
                    f"{sorted(known_subjects)}."
                ),
            )

    if levels:
        known_levels = set(pipeline.retriever.list_levels())
        bad = [lv for lv in levels if lv not in known_levels]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown level(s) {bad!r}. Use /taxonomy to list valid "
                    "levels for the current index."
                ),
            )


@app.post(
    "/retrieve",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def retrieve(req: RetrieveRequest, request: Request) -> dict:
    """Retrieval only — no LLM. Useful for the platform to debug whether a
    bad output is a retrieval problem or a generation problem.

    By default no distance threshold is applied (raw debug view). Pass
    `max_distance` in the request to mirror /quiz/generate, which applies
    `llm.default_max_distance` from configs/models.yaml.
    """
    p = _get_pipeline(request)
    _validate_taxonomy_inputs(p, subject=req.subject, levels=req.levels)
    try:
        results = p.retriever.retrieve(
            query=req.query,
            language=req.language,
            top_k=req.top_k,
            question_type=req.question_type,
            subject=req.subject,
            school_phase=req.school_phase,
            levels=req.levels,
            max_distance=req.max_distance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "query": req.query,
        "language": req.language,
        "max_distance_applied": req.max_distance,
        "count": len(results),
        "results": [_retrieved_to_dict(c) for c in results],
    }


@app.post(
    "/quiz/generate",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def generate_quiz(req: GenerateRequest, request: Request) -> dict:
    """Full retrieve → generate → validate → retry pipeline.

    Returns the GeneratedQuiz JSON. Adds a `retrieval` field when the
    caller passes `include_retrieval=true`.
    """
    p = _get_pipeline(request)
    _validate_taxonomy_inputs(p, subject=req.subject, levels=req.levels)

    # Lazy import — keeps the module loadable in tests that don't have ML deps.
    from src.generation.generator import GenerationError

    import time as _time
    _t0 = _time.perf_counter()

    # `temperature`, `max_attempts`, and `few_shot_count` come from
    # configs/models.yaml — not from the request. The pipeline reads the
    # config defaults when these are not passed.
    try:
        result = p.generate_detailed(
            topic=req.topic,
            language=req.language,
            count=req.count,
            question_type=req.question_type,
            subject=req.subject,
            school_phase=req.school_phase,
            levels=req.levels,
        )
    except ValueError as exc:
        # Bad inputs (e.g. unknown subject/level for taxonomy)
        raise HTTPException(status_code=400, detail=str(exc))
    except GenerationError as exc:
        # GenerationError covers two distinct cases, and they are NOT the
        # same fault:
        #   (1) Retriever returned 0 examples for these filters → the caller
        #       asked for something we don't have. 400, and Cloudflare passes
        #       4xx bodies through unchanged so the hint actually arrives.
        #   (2) LLM exhausted its retries despite having context → genuine
        #       upstream failure. 502.
        # Either way the body is sanitised: str(exc) embeds diagnose_empty()
        # output — corpus size, per-subject language counts — which is our
        # intelligence, not the caller's. Full text goes to the log under the
        # request ID.
        msg = str(exc)
        rid = request_id_ctx.get()
        logger.warning("generation failed: %s", msg)
        if (
            "Retriever returned 0 examples" in msg
            or "Cannot build a few-shot prompt" in msg
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "No matching content for the requested filters. Try a "
                    "broader topic, or relax subject / school_phase / levels. "
                    f"(request_id={rid})"
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=(
                "Generation failed: the model could not produce a valid quiz "
                f"for this topic. (request_id={rid})"
            ),
        )

    quiz = result.quiz

    response: dict = {
        "topic": req.topic,
        "language": quiz.language,
        "subject": quiz.subject,
        "level": quiz.level,
        "questions": [q.model_dump() for q in quiz.questions],
    }
    if req.include_retrieval:
        # include_retrieval is already the "show me your working" flag, so the
        # per-stage timings ride along with it rather than needing a second
        # switch. Opt-in only — the default response shape is unchanged for
        # the platform.
        response["retrieval"] = [
            _retrieved_to_dict(c) for c in result.retrieval
        ]
        if result.timings:
            response["timings"] = result.timings

    duration = round(_time.perf_counter() - _t0, 2)

    # Append the run to the JSONL log (best-effort). Always includes the
    # retrieval, per-stage timing, and total duration — useful for offline
    # review even if the caller didn't ask for retrieval in the response.
    log_response = dict(response)
    if "retrieval" not in log_response:
        log_response["retrieval"] = [
            _retrieved_to_dict(c) for c in result.retrieval
        ]
    log_response["duration_seconds"] = duration
    log_response["request_id"] = request_id_ctx.get()
    # Per-stage timings captured by QuizPipeline.generate() (retrieve vs LLM).
    # Lets offline analysis distinguish "retrieval was slow" from "the LLM was slow."
    if result.timings:
        log_response["timings"] = result.timings
    _append_run_log(request_dict=req.model_dump(), response_dict=log_response)

    return response


# ---------------------------------------------------------------------------
# Generic exception fallback — anything we didn't anticipate
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler so the server never leaks stack traces.

    Logs the full traceback server-side; returns a uniform error envelope.
    """
    rid = request_id_ctx.get()
    logger.exception("Unhandled exception while serving %s", request.url.path)
    # Exception messages routinely carry file paths, config values and SDK
    # internals. The client gets an ID; the detail stays in the log.
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_error",
            detail=f"An internal error occurred. (request_id={rid})",
        ).model_dump(),
        headers={REQUEST_ID_HEADER: rid},
    )
