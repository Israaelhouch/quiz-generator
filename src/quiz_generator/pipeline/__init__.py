"""End-to-end QuizPipeline.

Public API:
    from quiz_generator.pipeline import QuizPipeline
"""

from quiz_generator.pipeline.orchestrator import GenerationResult, QuizPipeline

__all__ = ["GenerationResult", "QuizPipeline"]
