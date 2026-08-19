"""End-to-end QuizPipeline (the orchestrator class).

A thin facade over the existing modules. Hides the wiring of:

    Retriever  +  LLMClient  +  Generator

behind a single class:

    pipeline = QuizPipeline(config_path="configs/models.yaml")
    result = pipeline.generate_detailed(
        topic="past simple vs past continuous",
        language="en",
        subject="ENGLISH",
        levels=["HIGH_SCHOOL_4TH_GRADE_ENGLISH"],
        count=5,
    )
    result.quiz        # validated GeneratedQuiz
    result.retrieval   # the chunks THIS call sent to the LLM
    result.timings     # per-stage seconds for THIS call

Concurrency note
----------------
One QuizPipeline instance is shared by every request (FastAPI builds it once
at startup and serves requests from a thread pool). Anything a call needs
after `generate()` returns must therefore travel back on the RETURN VALUE,
never on `self` — otherwise a concurrent call overwrites it in between and
request A reports request B's retrieval.

`generate_detailed()` is the thread-safe entry point and touches no shared
state. `generate()` is the single-threaded convenience wrapper: it returns
just the quiz and mirrors the rest onto `self.last_retrieval` /
`self.last_timings` for the CLI. Do not read those attributes from the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """Everything one generate call produced, bound together.

    Frozen and per-call, so two concurrent requests can never see each
    other's data.
    """

    quiz: Any                                  # GeneratedQuiz
    retrieval: list[Any] = field(default_factory=list)   # list[RetrievedQuestion]
    timings: dict[str, Any] = field(default_factory=dict)


class QuizPipeline:
    """Single entry point: config_path → generate() → GeneratedQuiz.

    The constructor builds the Retriever and the LLMClient eagerly so the
    heavy ML loads (BGE-M3 embedder, BGE reranker, Ollama warmup) happen
    once. `generate()` can then be called many times.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/models.yaml"),
        ready_jsonl_path: Path = Path("data/processed/ready_phase1.jsonl"),
        *,
        # Test-injection hooks — pass these to skip the heavy real builds.
        _retriever: Any | None = None,
        _llm_client: Any | None = None,
    ) -> None:
        from src.indexing.config import load_models_config
        from src.generation.generator import Generator

        self.config_path = config_path
        self.ready_jsonl_path = ready_jsonl_path
        self.config = load_models_config(config_path)
        self.llm_config = self.config.llm

        # 1. Retriever (loads embedder + Chroma + payload + optional reranker)
        if _retriever is not None:
            self.retriever = _retriever
        else:
            from src.retrieval.retriever import Retriever
            self.retriever = Retriever(
                config_path=config_path,
                ready_jsonl_path=ready_jsonl_path,
            )

        # 2. LLM client — provider switch lives here. Add new branches when
        # you wire up OpenAI / Anthropic / vLLM.
        if _llm_client is not None:
            self.llm_client = _llm_client
        else:
            self.llm_client = self._build_llm_client(self.llm_config)

        # 3. Generator (retrieve + prompt + LLM + validate + retry)
        self.generator = Generator(retriever=self.retriever, llm_client=self.llm_client)

        # Cache of the most recent retrieval, for the CLI only.
        # NOT SAFE to read from a concurrent server — see the module docstring
        # and use generate_detailed() instead.
        self.last_retrieval: list[Any] = []
        self.last_timings: dict[str, Any] = {}

    @staticmethod
    def _build_llm_client(llm_cfg: Any) -> Any:
        """Construct the LLM client from config. Extension point for new providers.

        The `OLLAMA_HOST` environment variable, if set, overrides the host
        from config. This lets us point the API at Ollama-in-another-container
        when running under docker-compose (where the address is
        `http://ollama:11434`) without changing models.yaml.
        """
        provider = llm_cfg.provider
        if provider == "ollama":
            import os
            from src.generation.llm_client import OllamaClient
            host = os.environ.get("OLLAMA_HOST") or llm_cfg.host
            return OllamaClient(model=llm_cfg.model, host=host)
        if provider == "groq":
            from src.generation.llm_client import GroqClient
            # GROQ_API_KEY is read from the environment by GroqClient itself.
            return GroqClient(model=llm_cfg.model)
        if provider == "gemini":
            from src.generation.llm_client import GeminiClient
            # GEMINI_API_KEY is read from the environment by GeminiClient itself.
            return GeminiClient(model=llm_cfg.model)
        raise ValueError(
            f"Unsupported llm provider {provider!r}. "
            "Add a branch in QuizPipeline._build_llm_client to wire it up."
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_detailed(
        self,
        *,
        topic: str,
        language: str,
        count: int = 5,
        question_type: str = "MULTIPLE_CHOICE",
        subject: str | None = None,
        school_phase: str | None = None,
        levels: list[str] | None = None,
        few_shot_count: int | None = None,
        temperature: float | None = None,
        max_attempts: int | None = None,
    ) -> GenerationResult:
        """Run the full retrieve → generate → validate flow.

        THREAD-SAFE: every value this call produces is returned on the
        GenerationResult. No instance attribute is written, so concurrent
        callers cannot observe each other's state.

        Returns a GenerationResult. Raises GenerationError on permanent
        failure (after max_attempts), or ValueError on bad inputs.

        `levels` is a list because Tunisian rows can carry several level tags;
        only the first is forwarded to the existing GenerationRequest (which
        takes a single `level` for now).
        """
        from src.generation.schemas import GenerationRequest

        # Resolve config-driven defaults — these are tuning knobs for the
        # AI engineer, not user inputs from the platform.
        if temperature is None:
            temperature = self.llm_config.default_temperature
        if max_attempts is None:
            max_attempts = self.llm_config.max_attempts
        if few_shot_count is None:
            few_shot_count = self.llm_config.default_few_shot_count
        # Quality floor for retrieval — chunks with distance > this get dropped
        # before the few_shot_count ceiling is applied.
        max_distance = self.llm_config.default_max_distance

        first_level: str | None = None
        if levels:
            if len(levels) > 1:
                logger.warning(
                    "Multiple levels passed (%s); forwarding only the first "
                    "(%r) to GenerationRequest. The retriever still filters "
                    "on the full list internally.",
                    levels, levels[0],
                )
            first_level = levels[0]

        request = GenerationRequest(
            topic=topic,
            language=language,
            count=count,
            question_type=question_type,
            subject=subject,
            level=first_level,
            few_shot_count=few_shot_count,
            temperature=temperature,
        )

        # Per-stage timing — returned on the result so callers (API server,
        # CLI) can log structured timings without re-instrumenting.
        import time as _time
        t_total_start = _time.perf_counter()
        logger.info(
            "generate start  topic=%r language=%s count=%d question_type=%s "
            "subject=%s school_phase=%s levels=%s few_shot=%d max_distance=%s",
            topic, language, count, question_type, subject, school_phase,
            levels, few_shot_count, max_distance,
        )

        # Retrieve ONCE here. Two benefits:
        #   1. We can hand the chunks back on the result for inspection.
        #   2. Avoids the previous double-fetch (probe + Generator's retrieve).
        # The Generator below uses these examples directly via
        # generate_with_examples().
        #
        # max_distance is the QUALITY FLOOR (drops noisy chunks);
        # top_k is the CEILING (caps prompt size).
        t_retrieve_start = _time.perf_counter()
        examples = self.retriever.retrieve(
            query=topic,
            language=language,
            top_k=few_shot_count,
            question_type=question_type,
            subject=subject,
            school_phase=school_phase,
            levels=levels,
            max_distance=max_distance,
        )
        t_retrieve = _time.perf_counter() - t_retrieve_start
        logger.info(
            "retrieve done   %d examples in %.3fs (top_k=%d, threshold=%s)",
            len(examples), t_retrieve, few_shot_count, max_distance,
        )
        retrieval = list(examples)

        # Low-pool warning (Decision 2a). Generator handles 0 examples itself.
        if 0 < len(retrieval) < few_shot_count:
            logger.warning(
                "Low retrieval pool: requested few_shot_count=%d but retrieval "
                "returned %d example(s) for topic=%r, language=%r, subject=%r, "
                "levels=%r. Generation will proceed with what's available. "
                "Consider broadening filters or rephrasing the topic for "
                "richer few-shot context.",
                few_shot_count, len(retrieval), topic, language, subject, levels,
            )

        logger.info(
            "llm call start  provider=%s model=%s temperature=%.2f "
            "max_attempts=%d count=%d",
            self.llm_config.provider, self.llm_config.model, temperature,
            max_attempts, count,
        )
        t_generate_start = _time.perf_counter()
        quiz = self.generator.generate_with_examples(
            request, retrieval, max_attempts=max_attempts
        )
        t_generate = _time.perf_counter() - t_generate_start
        n_questions = len(quiz.questions) if hasattr(quiz, "questions") else 0
        logger.info(
            "llm call done   %d questions in %.3fs", n_questions, t_generate,
        )

        t_total = _time.perf_counter() - t_total_start
        timings = {
            "retrieve_seconds": round(t_retrieve, 3),
            "generate_seconds": round(t_generate, 3),
            "total_seconds":    round(t_total, 3),
            "n_examples_used":  len(retrieval),
        }
        logger.info(
            "generate done   total=%.3fs (retrieve=%.3fs llm=%.3fs)",
            t_total, t_retrieve, t_generate,
        )
        return GenerationResult(quiz=quiz, retrieval=retrieval, timings=timings)

    def generate(self, **kwargs: Any) -> Any:
        """Single-threaded convenience wrapper. Returns just the GeneratedQuiz.

        Also mirrors the retrieval and timings onto `self.last_retrieval` /
        `self.last_timings` so the CLI can display them.

        DO NOT use this from a concurrent server — those two attributes are
        shared instance state and a parallel call will overwrite them between
        your `generate()` returning and your reading them. Call
        `generate_detailed()` there and read the result object instead.
        """
        result = self.generate_detailed(**kwargs)
        self.last_retrieval = result.retrieval
        self.last_timings = result.timings
        return result.quiz
