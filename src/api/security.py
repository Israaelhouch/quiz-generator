"""Edge concerns for the HTTP API: identity, request rate, correlation IDs.

Everything here is deliberately dependency-free (stdlib + FastAPI only) and
process-local. That is a real limitation, stated up front:

  * Rate limits are counted PER PROCESS. Run N uvicorn workers and the
    effective limit is N × RATE_LIMIT_PER_MINUTE. For a single-container
    deployment behind one Nginx that is fine; if you scale out, move the
    counter to Redis or enforce it at Nginx instead.
  * API keys are compared against an env var, not a database. There is no
    per-key metadata, no rotation workflow, no revocation list. This is the
    right amount of machinery for a backend-to-backend integration with one
    known caller; it is NOT an end-user auth system.

Configuration (all optional, all read per request so tests can vary them):

  API_KEYS                 comma-separated list of accepted keys.
                           UNSET => authentication DISABLED (dev mode).
  RATE_LIMIT_PER_MINUTE    max /quiz/generate + /retrieve calls per caller
                           per rolling 60s. Default 30. Set 0 to disable.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader


logger = logging.getLogger("quiz_api.security")


API_KEY_HEADER = "X-API-Key"

# Declaring the scheme (rather than just reading the header by hand) is what
# puts the "Authorize" button in the Swagger UI at /docs and marks the guarded
# routes as secured in the OpenAPI schema. auto_error=False because we raise
# our own 401 below — FastAPI's default message is less useful, and we still
# want to accept `Authorization: Bearer` as an alternative.
api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description=(
        "API key issued by the AI team. Alternatively send it as "
        "`Authorization: Bearer <key>`."
    ),
)
REQUEST_ID_HEADER = "X-Request-ID"

DEFAULT_RATE_LIMIT_PER_MINUTE = 30
_RATE_WINDOW_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def configured_api_keys() -> set[str]:
    """Parse API_KEYS. Empty set means authentication is disabled."""
    raw = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def auth_enabled() -> bool:
    return bool(configured_api_keys())


def configured_rate_limit() -> int:
    """Requests per minute per caller. 0 disables limiting."""
    raw = os.environ.get("RATE_LIMIT_PER_MINUTE")
    if raw is None:
        return DEFAULT_RATE_LIMIT_PER_MINUTE
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "RATE_LIMIT_PER_MINUTE=%r is not an integer; using default %d",
            raw, DEFAULT_RATE_LIMIT_PER_MINUTE,
        )
        return DEFAULT_RATE_LIMIT_PER_MINUTE
    return max(0, value)


def configured_cors_origins() -> list[str]:
    """Browser origins allowed to call this API.

    UNSET means no CORS headers are emitted at all — correct when the caller
    is the school platform's BACKEND (server-to-server calls aren't subject
    to CORS). Set it only if a browser talks to this API directly, and list
    exact origins: "*" plus an API key in a header is a combination you do
    not want, since any page on the internet could then spend your quota
    using a key it scraped from your own frontend bundle.
    """
    raw = os.environ.get("CORS_ALLOW_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def log_security_posture() -> None:
    """Emit the effective posture once, at startup. Loud when wide open."""
    if auth_enabled():
        logger.info(
            "API authentication ENABLED (%d key(s) configured)",
            len(configured_api_keys()),
        )
    else:
        logger.warning(
            "API authentication is DISABLED — API_KEYS is unset. Every caller "
            "that can reach this port may spend LLM quota. Do not expose this "
            "process to an untrusted network."
        )
    origins = configured_cors_origins()
    if origins == ["*"]:
        logger.warning(
            "CORS_ALLOW_ORIGINS=* — any web page may call this API from a "
            "browser. Only sane when authentication is also disabled and the "
            "port is unreachable from outside."
        )
    elif origins:
        logger.info("CORS enabled for origins: %s", origins)
    else:
        logger.info(
            "CORS disabled (server-to-server callers only; browsers cannot "
            "call this API directly)."
        )

    limit = configured_rate_limit()
    if limit:
        logger.info("Rate limit: %d requests/minute per caller (per process)", limit)
    else:
        logger.warning("Rate limiting is DISABLED (RATE_LIMIT_PER_MINUTE=0)")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _presented_key(request: Request) -> str | None:
    """Read the key from X-API-Key, or from an Authorization: Bearer header."""
    header = request.headers.get(API_KEY_HEADER)
    if header and header.strip():
        return header.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate:
            return candidate
    return None


def _matches_any(presented: str, accepted: set[str]) -> bool:
    """Constant-time comparison against every accepted key.

    Deliberately does NOT short-circuit on the first match: comparing against
    all of them keeps the work independent of which key was presented.
    """
    ok = False
    for key in accepted:
        if hmac.compare_digest(presented, key):
            ok = True
    return ok


def caller_id(request: Request) -> str:
    """Stable identifier for rate-limit accounting.

    Uses the API key when present (never the raw key — a short digest, so the
    secret can't leak through a log line), otherwise the client IP.
    """
    presented = _presented_key(request)
    if presented:
        import hashlib
        return "key:" + hashlib.sha256(presented.encode()).hexdigest()[:12]
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


async def require_api_key(
    request: Request,
    _scheme_key: str | None = Security(api_key_scheme),
) -> None:
    """FastAPI dependency. No-op when API_KEYS is unset (dev mode).

    `_scheme_key` is unused — the key is read from the raw request by
    `_presented_key` so both X-API-Key and Bearer work. It exists so the
    route is documented as secured and /docs renders an Authorize button.
    """
    accepted = configured_api_keys()
    if not accepted:
        return

    presented = _presented_key(request)
    if not presented:
        raise HTTPException(
            status_code=401,
            detail=f"Missing API key. Send it in the {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    if not _matches_any(presented, accepted):
        logger.warning(
            "Rejected request to %s from %s: invalid API key",
            request.url.path, caller_id(request),
        )
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# Rate limiting — in-process sliding window
# ---------------------------------------------------------------------------

_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def reset_rate_limits() -> None:
    """Drop all counters. For tests, and for an operator poking the process."""
    with _hits_lock:
        _hits.clear()


async def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency. 429s a caller over the per-minute budget.

    Sliding window rather than fixed buckets: a fixed bucket lets a caller
    send 2× the limit across a bucket boundary, which for an endpoint that
    costs three LLM calls is not a rounding error.
    """
    limit = configured_rate_limit()
    if limit <= 0:
        return

    who = caller_id(request)
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_SECONDS

    with _hits_lock:
        window = _hits[who]
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, int(window[0] + _RATE_WINDOW_SECONDS - now) + 1)
            logger.warning(
                "Rate limit hit by %s on %s (%d/%d in window)",
                who, request.url.path, len(window), limit,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit} requests/minute). "
                       f"Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)


# ---------------------------------------------------------------------------
# Correlation IDs
# ---------------------------------------------------------------------------


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def request_id_of(request: Request) -> str:
    """The current request's ID, assigned by the middleware in server.py."""
    return getattr(request.state, "request_id", "unassigned")
