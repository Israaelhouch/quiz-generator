"""End-to-end QuizPipeline.

Public API:
    from src.pipeline import QuizPipeline
"""

from src.pipeline.orchestrator import GenerationResult, QuizPipeline

__all__ = ["GenerationResult", "QuizPipeline"]
