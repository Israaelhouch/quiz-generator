"""Central logging configuration for the quiz generator.

Call `setup_logging()` ONCE at process start (from CLI / API entry points).
Other modules just do `logger = logging.getLogger(__name__)` and emit
log records — they don't configure anything themselves.

Conventions:
  - INFO   : one-line milestone per pipeline stage (retrieve start/end,
             LLM call start/end, total). Always on. Cheap. Operational.
  - DEBUG  : verbose internals (distance values, prompt previews, retry
             internals). Off by default; enable with LOG_LEVEL=DEBUG.
  - WARNING: low-pool retrieval, validation retries, deprecation.
  - ERROR  : permanent failure (caught at API boundary).

Format: ISO-ish timestamp + level + module + request_id + message.
Readable in a terminal, greppable in a log file, parseable by line.

Request IDs
-----------
`request_id_ctx` is a ContextVar that holds the current request's ID
(e.g., "req-7a3f2b"). Set by the API middleware at the start of every
HTTP request; reset at the end. Every log record automatically picks
up the current value via the LogRecord factory below.

Outside an HTTP request (CLI runs, pipeline jobs, startup hooks) the
value defaults to "-" so the log format stays consistent. Greppable
either way:
    grep "req-7a3f2b" logs/api.log   # one request's full timeline
    grep -v "\\[-\\]" logs/api.log   # everything inside a request scope
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys


# Set by the API middleware on every request. Defaults to "-" outside
# of a request scope so non-API code (CLI, pipeline) still logs cleanly.
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


_FORMAT  = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] [%(request_id)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _install_request_id_factory() -> None:
    """Make every LogRecord carry the current request_id.

    The format string above references `%(request_id)s`. Without this
    factory, log records emitted outside our middleware (e.g., from a
    third-party library) would raise KeyError because they lack the
    attribute. Setting it via the factory guarantees it's always present.
    """
    old_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        record = old_factory(*args, **kwargs)
        record.request_id = request_id_ctx.get()
        return record

    logging.setLogRecordFactory(factory)


# Install the factory at import time — safe to do multiple times because
# each call wraps the previous factory, but setup_logging() guards against
# double-install by checking a sentinel attribute on the factory.

# Third-party loggers that are too chatty at INFO. We clamp them to WARNING
# so the user only sees our own pipeline timings, not every HTTP retry or
# tokenizer cache hit.
_NOISY_LOGGERS = (
    "urllib3",
    "httpx",
    "httpcore",
    "sentence_transformers",
    "transformers",
    "chromadb",
    "asyncio",
)


_FACTORY_INSTALLED_SENTINEL = "_quiz_generator_request_id_factory"


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
    """Configure root logging for this process.

    Args:
        level: log level override. If None, reads LOG_LEVEL env var; falls
            back to "INFO" if that's unset. Case-insensitive.
        force: re-apply the config even if logging was already initialized
            elsewhere (uvicorn does its own basicConfig). Default False so
            we don't override a parent app's config in shared deployments.
    """
    # Install the request_id-injecting LogRecord factory exactly once per
    # process. Repeated calls to setup_logging() are common (uvicorn +
    # our own bootstrap both call it) so we guard with a sentinel.
    factory = logging.getLogRecordFactory()
    if not getattr(factory, _FACTORY_INSTALLED_SENTINEL, False):
        _install_request_id_factory()
        # Mark the new factory so we don't wrap it again on the next call.
        new_factory = logging.getLogRecordFactory()
        setattr(new_factory, _FACTORY_INSTALLED_SENTINEL, True)

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()

    logging.basicConfig(
        level=resolved,
        format=_FORMAT,
        datefmt=_DATEFMT,
        stream=sys.stderr,
        force=force,
    )

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
