"""In-process metrics for the HTTP API, in Prometheus exposition format.

No dependency, no client library, no push gateway — just counters behind a
lock and a `/metrics` endpoint that renders them as text.

Scope and limits, stated plainly:

  * Counters live in THIS process and reset when it restarts. With multiple
    uvicorn workers each one reports its own slice, so a scraper sees N
    separate series unless you aggregate. Fine for a single-container
    deployment; if you scale out, either scrape each worker or move to a
    real client library with a shared registry.
  * Latency is tracked as sum + count (enough for a rolling average), not as
    a histogram. `scripts/analyze_runs.py` already computes true percentiles
    offline from runs.jsonl; this is the live "is it slow right now" signal,
    not a replacement for that.
  * Paths are recorded verbatim. This API has four fixed routes and no path
    parameters, so there is no cardinality risk. Add a route with an ID in
    the path and you MUST template it here first.
"""

from __future__ import annotations

import threading
from collections import defaultdict


_lock = threading.Lock()

# (method, path, status) -> count
_requests: dict[tuple[str, str, str], int] = defaultdict(int)
# path -> (total seconds, number of requests)
_latency_sum: dict[str, float] = defaultdict(float)
_latency_count: dict[str, int] = defaultdict(int)
# reason -> count  (rate_limited, unauthorized, generation_failed, ...)
_events: dict[str, int] = defaultdict(int)


def record_request(*, method: str, path: str, status: int, duration: float) -> None:
    """Record one completed HTTP request."""
    with _lock:
        _requests[(method, path, str(status))] += 1
        _latency_sum[path] += duration
        _latency_count[path] += 1


def record_event(name: str) -> None:
    """Bump a named counter (rate_limited, unauthorized, ...)."""
    with _lock:
        _events[name] += 1


def reset_metrics() -> None:
    """Clear every counter. For tests."""
    with _lock:
        _requests.clear()
        _latency_sum.clear()
        _latency_count.clear()
        _events.clear()


def snapshot() -> dict:
    """Plain-dict view of current counters (for tests and debugging)."""
    with _lock:
        return {
            "requests": {"|".join(k): v for k, v in _requests.items()},
            "latency_sum": dict(_latency_sum),
            "latency_count": dict(_latency_count),
            "events": dict(_events),
        }


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


def render_prometheus() -> str:
    """Render the current counters in Prometheus text exposition format."""
    lines: list[str] = []

    lines.append("# HELP quiz_api_up 1 when the API process is serving.")
    lines.append("# TYPE quiz_api_up gauge")
    lines.append("quiz_api_up 1")

    lines.append("# HELP quiz_api_requests_total HTTP requests by method, path and status.")
    lines.append("# TYPE quiz_api_requests_total counter")
    with _lock:
        requests = sorted(_requests.items())
        latency = sorted(_latency_sum.items())
        latency_counts = dict(_latency_count)
        events = sorted(_events.items())

    for (method, path, status), count in requests:
        lines.append(
            f'quiz_api_requests_total{{method="{_escape(method)}",'
            f'path="{_escape(path)}",status="{_escape(status)}"}} {count}'
        )

    lines.append("# HELP quiz_api_request_duration_seconds_sum Total time spent per path.")
    lines.append("# TYPE quiz_api_request_duration_seconds_sum counter")
    for path, total in latency:
        lines.append(
            f'quiz_api_request_duration_seconds_sum{{path="{_escape(path)}"}} {total:.6f}'
        )

    lines.append("# HELP quiz_api_request_duration_seconds_count Requests observed per path.")
    lines.append("# TYPE quiz_api_request_duration_seconds_count counter")
    for path, _ in latency:
        lines.append(
            f'quiz_api_request_duration_seconds_count{{path="{_escape(path)}"}} '
            f'{latency_counts.get(path, 0)}'
        )

    lines.append("# HELP quiz_api_events_total Notable events by reason.")
    lines.append("# TYPE quiz_api_events_total counter")
    for name, count in events:
        lines.append(f'quiz_api_events_total{{reason="{_escape(name)}"}} {count}')

    return "\n".join(lines) + "\n"
