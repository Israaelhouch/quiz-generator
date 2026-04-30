"""Entry point for `python -m src.api`.

Starts the FastAPI app under uvicorn. Defaults to localhost:8000 — change
with --host / --port. Add --reload during development to auto-reload on
code changes (don't use --reload in production: the BGE-M3 + reranker
warmup happens on every reload).

Usage:
    python -m src.api                          # localhost:8000
    python -m src.api --port 9000              # custom port
    python -m src.api --host 0.0.0.0           # bind to all interfaces
    python -m src.api --reload                 # dev mode (auto-reload)
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.api",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1 — localhost only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (dev only — slow)")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    args = parser.parse_args()

    # Lazy import so --help is snappy
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
