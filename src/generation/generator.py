"""Stage 5 orchestrator — retriever + prompt + LLM + validation, with retry.

Flow:
  1. retriever.retrieve(...)             — get few-shot examples
  2. build_prompt(...)                   — construct system + user messages
  3. llm.complete_json(...)              — call Qwen (or mock)
  4. parse JSON + Pydantic validate      — enforce invariants
     - on failure: build a retry prompt with the error feedback,
       call LLM again, validate again. Up to N attempts.
  5. return GeneratedQuiz

Retry pattern: when a Qwen response fails validation, we send Qwen the
error message and ask it to fix the issue. Most failures (slightly bad
JSON, mismatched correct_answers, etc.) resolve in 1-2 retries.
"""

from __future__ import annotations

import json
from typing import Any

from src.generation.llm_client import LLMClient
from src.generation.prompts.simple import build_prompt
from src.generation.schemas import (
    GeneratedQuestion,
    GeneratedQuiz,
    GenerationRequest,
)


class GenerationError(RuntimeError):
    """Raised when generation definitively fails after exhausting retries."""


# Default cap on retry attempts. Configurable per-call via Generator.generate.
DEFAULT_MAX_ATTEMPTS = 3


class Generator:
    """Retriever + LLM wrapper producing validated GeneratedQuiz objects."""

    def __init__(
        self,
        retriever: Any,             # src.retrieval.retriever.Retriever or any .retrieve()-capable object
        llm_client: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.llm = llm_client

    def generate(
        self,
        request: GenerationRequest,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> GeneratedQuiz:
        """Retrieve few-shot examples and generate a quiz.

        Thin wrapper that fetches examples then delegates. Use
        `generate_with_examples()` directly when you've already retrieved
        (e.g. the QuizPipeline retrieves once and reuses the result).
        """
        examples = self.retriever.retrieve(
            query=request.topic,
            language=request.language,
            top_k=request.few_shot_count,
            question_type=request.question_type,
            subject=request.subject,
            levels=[request.level] if request.level else None,
        )
        return self.generate_with_examples(
            request, examples, max_attempts=max_attempts
        )

    def generate_with_examples(
        self,
        request: GenerationRequest,
        examples: list,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> GeneratedQuiz:
        """Generate a quiz from already-retrieved few-shot examples.

        This is the core loop (build prompt → call LLM → validate → retry).
        Splitting it out lets callers (like QuizPipeline) retrieve once and
        also display/save the chunks they passed in — without doing a second
        retrieval round-trip.
        """
        if not examples:
            diagnostic = ""
            diagnose = getattr(self.retriever, "diagnose_empty", None)
            if callable(diagnose):
                try:
                    diagnostic = "\n" + diagnose(
                        language=request.language,
                        question_type=request.question_type,
                        subject=request.subject,
                        levels=[request.level] if request.level else None,
                    )
                except Exception:
                    diagnostic = ""
            raise GenerationError(
                f"Retriever returned 0 examples for topic={request.topic!r}, "
                f"subject={request.subject!r}, level={request.level!r}. "
                f"Cannot build a few-shot prompt without examples.{diagnostic}"
            )

        # Build base prompt (used on attempt 1; retries augment the user message)
        system, base_user = build_prompt(
            language=request.language,
            question_type=request.question_type,
            topic=request.topic,
            count=request.count,
            examples=examples,
            subject=request.subject,
            level=request.level,
        )

        last_error: str | None = None
        last_raw: str | None = None

        for attempt in range(1, max_attempts + 1):
            # Build the user message — base prompt for attempt 1, base + retry feedback after
            if attempt == 1:
                user_message = base_user
            else:
                user_message = self._build_retry_prompt(
                    base_user=base_user,
                    last_error=last_error or "(unknown)",
                    last_raw=last_raw or "",
                    attempt=attempt,
                    requested_count=request.count,
                )

            # 3. Call LLM
            raw = self.llm.complete_json(
                system=system, user=user_message, temperature=request.temperature
            )
            last_raw = raw

            # 4. Parse + validate
            quiz_or_error = self._parse_and_validate(raw, request)
            if isinstance(quiz_or_error, GeneratedQuiz):
                return quiz_or_error                       # success
            last_error = quiz_or_error                     # error string; retry

        # All attempts exhausted
        raise GenerationError(
            f"Generation failed after {max_attempts} attempts. "
            f"Last error: {last_error}\n"
            f"Last raw response (truncated): {(last_raw or '')[:500]}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: str,
        request: GenerationRequest,
    ) -> GeneratedQuiz | str:
        """Parse JSON + validate each question.

        Returns GeneratedQuiz on success, or an error string describing the first
        failure encountered. The error string is what we feed back to the LLM
        on retry.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return f"LLM returned invalid JSON: {exc}"

        questions_raw = data.get("questions") if isinstance(data, dict) else None
        if not isinstance(questions_raw, list) or not questions_raw:
            return (
                f"Response missing or empty 'questions' list. "
                f"Expected: {{\"questions\": [...]}}, got top-level type "
                f"{type(data).__name__}."
            )

        validated: list[GeneratedQuestion] = []
        for idx, q in enumerate(questions_raw):
            if not isinstance(q, dict):
                return (
                    f"Question {idx} is not a JSON object — "
                    f"got {type(q).__name__} instead."
                )
            q_with_type = {**q, "question_type": request.question_type}
            try:
                validated.append(GeneratedQuestion.model_validate(q_with_type))
            except Exception as exc:
                return f"Question {idx} failed validation: {exc}"

        # Count check — Qwen sometimes returns fewer (or more) questions than asked.
        if len(validated) != request.count:
            return (
                f"Wrong number of questions: requested {request.count} but got "
                f"{len(validated)}. Generate exactly {request.count} questions."
            )

        return GeneratedQuiz(
            language=request.language,
            subject=request.subject,
            level=request.level,
            questions=validated,
        )

    def _build_retry_prompt(
        self,
        *,
        base_user: str,
        last_error: str,
        last_raw: str,
        attempt: int,
        requested_count: int,
    ) -> str:
        """Build a retry user message that includes the error feedback."""
        # Keep last_raw bounded so we don't blow the context window
        truncated_raw = last_raw[:1500] + ("..." if len(last_raw) > 1500 else "")
        return (
            f"{base_user}\n\n"
            f"---\n"
            f"PREVIOUS ATTEMPT FAILED (this is attempt #{attempt}).\n"
            f"\n"
            f"Error: {last_error}\n"
            f"\n"
            f"Your previous JSON output (for reference):\n"
            f"{truncated_raw}\n"
            f"\n"
            f"Please regenerate the questions, fixing the issue above.\n"
            f"REQUIREMENTS:\n"
            f"- Generate EXACTLY {requested_count} questions in the JSON array.\n"
            f"- Each correct_answers entry must appear VERBATIM in the choices list.\n"
            f"- Output valid JSON only — no commentary, no markdown."
        )
