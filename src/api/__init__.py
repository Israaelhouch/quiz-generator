"""HTTP API surface for the Quiz Generator.

Wraps `src.pipeline.QuizPipeline` behind a FastAPI app so the school
platform can call us over HTTP instead of running the CLI.

To get the FastAPI instance:
    from src.api.server import app

We deliberately don't re-export it here — that keeps `python -m src.api
--help` working even when FastAPI isn't installed (e.g. in tests that
only need the CLI surface).
"""
