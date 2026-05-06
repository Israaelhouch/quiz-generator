"""Stage 6 — End-to-end QuizPipeline (the orchestrator class).

A thin facade over the existing modules. Hides the wiring of:

    Retriever  +  LLMClient  +  Generator

behind a single class with one method:

    pipeline = QuizPipeline(config_path="configs/models.yaml")
    quiz = pipeline.generate(
        topic="primitives des fonctions",
        language="fr",
        subject="MATHEMATICS",
        levels=["HIGH_SCHOOL_4TH_GRADE_MATHEMATICS"],
        count=3,
    )

Returns a validated `GeneratedQuiz` (Pydantic model from
`src.generation.schemas`).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


class QuizPipeline:
    """Single entry point: config_path → generate() → GeneratedQuiz.

    The constructor builds the Retriever and the LLMClient eagerly so the
    heavy ML loads (BGE-M3 embedder, BGE reranker, Ollama warmup) happen
    once. `generate()` can then be called many times.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/models.yaml"),
        ready_jsonl_path: Path = Path("data/processed/ready.jsonl"),
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

        # Cache of the most recent retrieval — set by every generate() call.
        # The CLI uses this to display / save the chunks the LLM saw.
        self.last_retrieval: list[Any] = []

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

    def generate(
        self,
        *,
        topic: str,
        language: str,
        count: int = 5,
        question_type: str = "MULTIPLE_CHOICE",
        subject: str | None = None,
        levels: list[str] | None = None,
        few_shot_count: int | None = None,
        temperature: float | None = None,
        max_attempts: int | None = None,
    ) -> Any:
        """Run the full retrieve → generate → validate flow.

        Returns a GeneratedQuiz (typed). Raises GenerationError on permanent
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
                warnings.warn(
                    f"Multiple levels passed ({levels}); forwarding only the "
                    f"first ({levels[0]!r}) to GenerationRequest. "
                    "The retriever still filters on the full list internally."
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

        # Retrieve ONCE here. Two benefits:
        #   1. We can expose the chunks via self.last_retrieval for inspection.
        #   2. Avoids the previous double-fetch (probe + Generator's retrieve).
        # The Generator below uses these examples directly via
        # generate_with_examples().
        #
        # max_distance is the QUALITY FLOOR (drops noisy chunks);
        # top_k is the CEILING (caps prompt size).
        examples = self.retriever.retrieve(
            query=topic,
            language=language,
            top_k=few_shot_count,
            question_type=question_type,
            subject=subject,
            levels=levels,
            max_distance=max_distance,
        )
        self.last_retrieval = list(examples)

        # Low-pool warning (Decision 2a). Generator handles 0 examples itself.
        if 0 < len(examples) < few_shot_count:
            warnings.warn(
                f"Low retrieval pool: requested few_shot_count={few_shot_count} "
                f"but retrieval returned {len(examples)} example(s) for "
                f"topic={topic!r}, language={language!r}, subject={subject!r}, "
                f"levels={levels!r}. Generation will proceed with what's "
                "available. Consider broadening filters or rephrasing the topic "
                "for richer few-shot context."
            )

        return self.generator.generate_with_examples(
            request, examples, max_attempts=max_attempts
        )
