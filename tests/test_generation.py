"""Tests for Stage 5 generation v1 (English MCQ, simple).

Uses MockClient for the LLM and a fake retriever for Stage 4.
No real Ollama or vector store needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.generator import GenerationError, Generator
from src.generation.llm_client import MockClient
from src.generation.prompts.simple import (
    build_mcq_prompt_english,
    build_prompt,
    build_prompt_english,
)
from src.generation.schemas import (
    GeneratedQuestion,
    GeneratedQuiz,
    GenerationRequest,
)
from src.retrieval.schemas import RetrievedQuestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _retrieved(
    doc_id: str,
    question_text: str = "What is X?",
    choices: list[str] | None = None,
    correct: list[str] | None = None,
) -> RetrievedQuestion:
    return RetrievedQuestion(
        doc_id=doc_id,
        quiz_id="quiz-1",
        quiz_title="Test Quiz",
        language="en",
        question_type="MULTIPLE_CHOICE",
        question_text=question_text,
        choices=choices or ["A", "B", "C", "D"],
        correct_answers=correct or ["A"],
        subjects=["SCIENCE"],
        levels=["PRIMARY_SCHOOL_6TH_GRADE"],
        multiple_correct_answers=False,
        author_name=None,
        author_email=None,
        search_text="search text",
        metadata={},
        distance=0.2,
    )


class FakeRetriever:
    """Mimics the retriever interface — returns canned results."""

    def __init__(self, canned: list[RetrievedQuestion]) -> None:
        self.canned = canned
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> list[RetrievedQuestion]:
        self.calls.append(kwargs)
        return list(self.canned)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_generated_question_valid_mcq() -> None:
    q = GeneratedQuestion(
        question_type="MULTIPLE_CHOICE",
        question_text="What color is the sky?",
        choices=["Blue", "Red", "Green"],
        correct_answers=["Blue"],
        explanation="Sky appears blue due to Rayleigh scattering.",
        difficulty="easy",
    )
    assert q.correct_answers == ["Blue"]


def test_generated_question_rejects_answer_not_in_choices() -> None:
    try:
        GeneratedQuestion(
            question_type="MULTIPLE_CHOICE",
            question_text="X?",
            choices=["A", "B"],
            correct_answers=["C"],   # not in choices
        )
        raise AssertionError("expected ValidationError")
    except Exception as exc:
        assert "not found verbatim" in str(exc)


def test_generated_question_rejects_empty_correct_answers() -> None:
    try:
        GeneratedQuestion(
            question_type="MULTIPLE_CHOICE",
            question_text="X?",
            choices=["A", "B"],
            correct_answers=[],
        )
        raise AssertionError("expected ValidationError")
    except Exception as exc:
        assert "at least one correct_answer" in str(exc)


def test_generated_question_empty_question_text_rejected() -> None:
    try:
        GeneratedQuestion(
            question_type="MULTIPLE_CHOICE",
            question_text="   ",
            choices=["A", "B"],
            correct_answers=["A"],
        )
        raise AssertionError("expected ValidationError")
    except Exception as exc:
        assert "question_text" in str(exc)


def test_generated_question_multiple_correct_answers_derived_from_answer_count() -> None:
    """`multiple_correct_answers` is auto-derived from correct_answers count.

    LLMs often forget to set this flag; we trust the data (correct_answers list)
    over the flag — same pattern we used in Stage 2a for source data.
    """
    # Two correct answers → multiple_correct_answers should become True regardless of input
    q = GeneratedQuestion(
        question_type="MULTIPLE_CHOICE",
        question_text="X?",
        choices=["A", "B", "C"],
        correct_answers=["A", "B"],
        multiple_correct_answers=False,  # LLM forgot to flip it; we fix it
    )
    assert q.multiple_correct_answers is True

    # One correct answer → multiple_correct_answers should be False regardless of input
    q2 = GeneratedQuestion(
        question_type="MULTIPLE_CHOICE",
        question_text="X?",
        choices=["A", "B", "C"],
        correct_answers=["A"],
        multiple_correct_answers=True,   # overclaimed; we fix it
    )
    assert q2.multiple_correct_answers is False


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_build_prompt_contains_topic_and_count() -> None:
    examples = [_retrieved(f"ex-{i}") for i in range(3)]
    system, user = build_mcq_prompt_english(
        topic="photosynthesis", count=2, examples=examples,
    )
    assert "photosynthesis" in user
    assert "2 new multiple-choice" in user


def test_build_prompt_renders_all_examples() -> None:
    examples = [
        _retrieved("ex-1", question_text="Q one"),
        _retrieved("ex-2", question_text="Q two"),
        _retrieved("ex-3", question_text="Q three"),
    ]
    _, user = build_mcq_prompt_english(topic="x", count=1, examples=examples)
    assert "Example 1" in user and "Q one" in user
    assert "Example 2" in user and "Q two" in user
    assert "Example 3" in user and "Q three" in user


def test_build_prompt_includes_subject_and_level_when_provided() -> None:
    examples = [_retrieved("ex-1")]
    _, user = build_mcq_prompt_english(
        topic="x", count=1, examples=examples,
        subject="MATHEMATICS", level="HIGH_SCHOOL_4TH_GRADE_MATHEMATICS",
    )
    assert "MATHEMATICS" in user
    assert "HIGH_SCHOOL_4TH_GRADE_MATHEMATICS" in user


def test_system_message_states_english_only() -> None:
    examples = [_retrieved("ex-1")]
    system, _ = build_mcq_prompt_english(topic="x", count=1, examples=examples)
    assert "English" in system
    assert "JSON" in system


# ---------------------------------------------------------------------------
# Generator orchestrator
# ---------------------------------------------------------------------------


CANNED_GOOD_RESPONSE = json.dumps({
    "questions": [
        {
            "question_text": "What is the main source of energy for Earth?",
            "choices": ["The Sun", "The Moon", "Wind", "Geothermal"],
            "correct_answers": ["The Sun"],
            "explanation": "Solar radiation is the primary energy input to Earth.",
            "difficulty": "easy",
        },
        {
            "question_text": "Where does photosynthesis happen in a plant cell?",
            "choices": ["Chloroplast", "Mitochondria", "Nucleus", "Vacuole"],
            "correct_answers": ["Chloroplast"],
            "explanation": "Chloroplasts contain chlorophyll.",
            "difficulty": "medium",
        },
    ]
})


def test_generator_happy_path() -> None:
    examples = [_retrieved(f"ex-{i}") for i in range(3)]
    retriever = FakeRetriever(examples)
    llm = MockClient(canned_response=CANNED_GOOD_RESPONSE)
    gen = Generator(retriever=retriever, llm_client=llm)

    result = gen.generate(
        GenerationRequest(topic="photosynthesis", language="en", count=2)
    )
    assert isinstance(result, GeneratedQuiz)
    assert len(result.questions) == 2
    assert result.questions[0].question_text.startswith("What is")
    assert result.language == "en"


def test_generator_passes_filters_to_retriever() -> None:
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)
    llm = MockClient(canned_response=CANNED_GOOD_RESPONSE)
    gen = Generator(retriever=retriever, llm_client=llm)

    # count=2 to match CANNED_GOOD_RESPONSE (which contains 2 questions);
    # the new count-validation in _parse_and_validate would otherwise reject
    # a mismatch. This test is about *filter propagation*, so the count
    # value is incidental.
    gen.generate(GenerationRequest(
        topic="x", language="en", count=2,
        subject="SCIENCE", level="PRIMARY_SCHOOL_6TH_GRADE",
        few_shot_count=3,
    ))

    call = retriever.calls[0]
    assert call["query"] == "x"
    assert call["language"] == "en"
    assert call["subject"] == "SCIENCE"
    assert call["levels"] == ["PRIMARY_SCHOOL_6TH_GRADE"]
    assert call["top_k"] == 3
    assert call["question_type"] == "MULTIPLE_CHOICE"


def test_generator_raises_when_no_examples_retrieved() -> None:
    retriever = FakeRetriever([])   # empty
    llm = MockClient(canned_response=CANNED_GOOD_RESPONSE)
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1))
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "0 examples" in str(exc)


def test_generator_raises_on_invalid_json_from_llm() -> None:
    retriever = FakeRetriever([_retrieved("ex-1")])
    llm = MockClient(canned_response="not json at all {")
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1))
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "invalid JSON" in str(exc)


def test_generator_raises_when_llm_output_missing_questions_key() -> None:
    retriever = FakeRetriever([_retrieved("ex-1")])
    llm = MockClient(canned_response=json.dumps({"wrong_key": []}))
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1))
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "questions" in str(exc).lower()


def test_generator_raises_when_llm_output_has_answer_not_in_choices() -> None:
    bad = json.dumps({"questions": [{
        "question_text": "X?",
        "choices": ["A", "B"],
        "correct_answers": ["C"],     # oops
    }]})
    retriever = FakeRetriever([_retrieved("ex-1")])
    llm = MockClient(canned_response=bad)
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1))
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "not found verbatim" in str(exc)


def test_generator_now_accepts_french_and_arabic() -> None:
    """Step 2 of Stage 5 unlocked fr and ar — no more NotImplementedError."""
    examples = [_retrieved("ex-1", question_text="Q1")]
    retriever = FakeRetriever(examples)
    llm = MockClient(canned_response=CANNED_GOOD_RESPONSE)
    gen = Generator(retriever=retriever, llm_client=llm)

    # French request — should succeed (mock LLM returns the canned response)
    quiz_fr = gen.generate(GenerationRequest(
        topic="primitives", language="fr", count=2,
        question_type="MULTIPLE_CHOICE",
    ))
    assert quiz_fr.language == "fr"
    assert len(quiz_fr.questions) == 2

    # Arabic request — should also succeed
    quiz_ar = gen.generate(GenerationRequest(
        topic="الرياضيات", language="ar", count=2,
        question_type="MULTIPLE_CHOICE",
    ))
    assert quiz_ar.language == "ar"


def test_build_prompt_french_uses_french_strings() -> None:
    examples = [_retrieved("ex-1")]
    system, user = build_prompt(
        language="fr",
        question_type="MULTIPLE_CHOICE",
        topic="primitives", count=2, examples=examples,
    )
    # French-specific phrases
    assert "Vous êtes un expert" in system
    assert "TÂCHE" in user
    assert "RÈGLES POUR MULTIPLE_CHOICE" in user
    assert "à choix multiples" in user   # type display
    # Must NOT contain English equivalents
    assert "TASK:" not in user
    assert "RULES FOR" not in user


def test_build_prompt_arabic_uses_arabic_strings() -> None:
    examples = [_retrieved("ex-1")]
    system, user = build_prompt(
        language="ar",
        question_type="FILL_IN_THE_BLANKS",
        topic="القواعد", count=1, examples=examples,
    )
    # Arabic-specific phrases (key tokens)
    assert "أنت خبير" in system
    assert "المهمة" in user
    assert "ملء الفراغات" in user           # type display for FITB
    assert "قواعد FILL_IN_THE_BLANKS" in user
    # JSON keys stay English (machine-readable contract)
    assert "question_text" in user
    assert "correct_answers" in user


def test_build_prompt_rejects_unsupported_language() -> None:
    examples = [_retrieved("ex-1")]
    try:
        build_prompt(
            language="es",   # Spanish — not supported
            question_type="MULTIPLE_CHOICE",
            topic="x", count=1, examples=examples,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Unsupported language" in str(exc)


def test_inline_choice_warning_present_in_all_languages() -> None:
    """The 'do not mimic label-only choices' warning must be in every language."""
    examples = [_retrieved("ex-1")]
    for lang in ("en", "fr", "ar"):
        _, user = build_prompt(
            language=lang, question_type="MULTIPLE_CHOICE",
            topic="x", count=1, examples=examples,
        )
        # Each language's warning has its own keyword we can grep for
        if lang == "en":
            assert "placeholder labels" in user
        elif lang == "fr":
            assert "étiquettes" in user
        elif lang == "ar":
            assert "رموز نائبة" in user


def test_build_prompt_english_fitb_has_fitb_rules() -> None:
    """FITB prompt instructs empty choices and the ___ marker."""
    examples = [_retrieved("ex-1")]
    system, user = build_prompt_english(
        question_type="FILL_IN_THE_BLANKS",
        topic="past tense", count=2, examples=examples,
    )
    assert "FILL_IN_THE_BLANKS" in user
    assert "___" in user   # the blank marker rule
    assert "empty list" in user   # the choices-must-be-empty rule


def test_build_prompt_english_rejects_unknown_type() -> None:
    examples = [_retrieved("ex-1")]
    try:
        build_prompt_english(
            question_type="UNSUPPORTED_TYPE",
            topic="x", count=1, examples=examples,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Unsupported question_type" in str(exc)


# ---------------------------------------------------------------------------
# End-to-end generator tests for FITB and TMC
# ---------------------------------------------------------------------------


CANNED_FITB_RESPONSE = json.dumps({
    "questions": [
        {
            "question_text": "The sun rises in the ___.",
            "choices": [],
            "correct_answers": ["east", "East", "EAST"],
            "explanation": "The sun rises in the east due to Earth's west-to-east rotation.",
            "difficulty": "easy",
        },
        {
            "question_text": "Water boils at ___ degrees Celsius at sea level.",
            "choices": [],
            "correct_answers": ["100"],
            "explanation": "At standard atmospheric pressure, water boils at 100°C.",
            "difficulty": "easy",
        },
    ]
})


CANNED_TMC_RESPONSE = json.dumps({
    "questions": [
        {
            "question_text": "Which sentence uses the past perfect tense correctly?",
            "choices": [
                "I had already eaten when she arrived.",
                "I have already eaten when she arrived.",
                "I eaten already when she arrived.",
                "I already eat when she arrived.",
            ],
            "correct_answers": ["I had already eaten when she arrived."],
            "explanation": "Past perfect uses 'had' + past participle to describe an action completed before another past action.",
            "difficulty": "medium",
        },
    ]
})


def _retrieved_fitb(doc_id: str, question_text: str = "He ___ to school.", correct: list[str] | None = None) -> RetrievedQuestion:
    return RetrievedQuestion(
        doc_id=doc_id,
        quiz_id="quiz-1",
        quiz_title="Grammar",
        language="en",
        question_type="FILL_IN_THE_BLANKS",
        question_text=question_text,
        choices=[],
        correct_answers=correct or ["went"],
        subjects=["ENGLISH"],
        levels=["MIDDLE_SCHOOL_2ND_GRADE"],
        multiple_correct_answers=False,
        author_name=None,
        author_email=None,
        search_text="grammar example",
        metadata={},
        distance=0.2,
    )


def test_generator_accepts_fitb_and_returns_valid_questions() -> None:
    """Generator now supports FILL_IN_THE_BLANKS — no NotImplementedError."""
    examples = [_retrieved_fitb(f"ex-{i}") for i in range(3)]
    retriever = FakeRetriever(examples)
    llm = MockClient(canned_response=CANNED_FITB_RESPONSE)
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(
        topic="past tense", language="en", count=2,
        question_type="FILL_IN_THE_BLANKS",
    ))
    assert len(quiz.questions) == 2
    # Validation accepts empty choices for FITB
    for q in quiz.questions:
        assert q.question_type == "FILL_IN_THE_BLANKS"
        assert q.choices == []
        assert q.correct_answers


class _SequencedMockClient:
    """LLM mock that returns a different canned response per call.

    Simulates Qwen behaviour: first response fails validation, second
    response is correct. Lets us verify the retry loop in the generator.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, temperature: float = 0.75) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self.responses:
            raise RuntimeError("Mock ran out of canned responses")
        return self.responses.pop(0)


def test_generator_retries_after_validation_failure_and_succeeds() -> None:
    """When the first response fails validation, generator retries with error
    feedback. The second response is good → final output is valid."""
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)

    # First response: correct_answer not in choices → validation fails
    bad_response = json.dumps({"questions": [{
        "question_text": "What is 2+2?",
        "choices": ["a) 4", "b) 5", "c) 6"],
        "correct_answers": ["a)"],     # ← bug: just the label, not in choices
        "explanation": "Basic math.",
        "difficulty": "easy",
    }]})
    # Second response: correct, full choice in correct_answers
    good_response = json.dumps({"questions": [{
        "question_text": "What is 2+2?",
        "choices": ["a) 4", "b) 5", "c) 6"],
        "correct_answers": ["a) 4"],
        "explanation": "Basic math.",
        "difficulty": "easy",
    }]})

    llm = _SequencedMockClient([bad_response, good_response])
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(topic="x", language="en", count=1))
    assert len(quiz.questions) == 1
    assert quiz.questions[0].correct_answers == ["a) 4"]
    assert len(llm.calls) == 2  # confirms retry happened

    # Second call's user message should contain the retry feedback
    second_call_user = llm.calls[1]["user"]
    assert "PREVIOUS ATTEMPT FAILED" in second_call_user
    assert "not found verbatim" in second_call_user


def test_generator_exhausts_retries_and_raises() -> None:
    """When all retries fail, raises GenerationError with the last error."""
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)

    # All 3 responses bad — same broken format
    bad = json.dumps({"questions": [{
        "question_text": "X?",
        "choices": ["a", "b", "c"],
        "correct_answers": ["NotInChoices"],
        "explanation": "...",
    }]})
    llm = _SequencedMockClient([bad, bad, bad])
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1))
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "after 3 attempts" in str(exc)
        assert len(llm.calls) == 3


def test_generator_succeeds_first_try_no_retry() -> None:
    """Happy path: first response is valid, no retry needed."""
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)
    llm = _SequencedMockClient([CANNED_GOOD_RESPONSE])
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(topic="x", language="en", count=2))
    assert len(quiz.questions) == 2
    assert len(llm.calls) == 1  # no retry


def test_generator_retries_on_invalid_json_too() -> None:
    """Retry loop also handles json.loads failures (not just Pydantic ones)."""
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)

    bad_json = "this is not json at all {"
    llm = _SequencedMockClient([bad_json, CANNED_GOOD_RESPONSE])
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(topic="x", language="en", count=2))
    assert len(quiz.questions) == 2
    assert len(llm.calls) == 2


def test_generator_retries_when_qwen_returns_wrong_count() -> None:
    """Qwen sometimes returns fewer questions than requested. Treat that as
    a validation failure and retry with explicit count reminder.
    """
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)

    # First response: Qwen produced only 1 question even though we asked for 3
    short_response = json.dumps({"questions": [{
        "question_text": "Q1?",
        "choices": ["A", "B", "C"],
        "correct_answers": ["A"],
        "explanation": "...",
        "difficulty": "easy",
    }]})
    # Second response: 3 questions
    correct_response = json.dumps({"questions": [
        {"question_text": "Q1?", "choices": ["A", "B", "C"], "correct_answers": ["A"], "explanation": "x", "difficulty": "easy"},
        {"question_text": "Q2?", "choices": ["A", "B", "C"], "correct_answers": ["B"], "explanation": "x", "difficulty": "easy"},
        {"question_text": "Q3?", "choices": ["A", "B", "C"], "correct_answers": ["C"], "explanation": "x", "difficulty": "easy"},
    ]})

    llm = _SequencedMockClient([short_response, correct_response])
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(topic="x", language="en", count=3))
    assert len(quiz.questions) == 3
    assert len(llm.calls) == 2  # confirms retry happened
    # Retry message should mention the count requirement
    second_user = llm.calls[1]["user"]
    assert "EXACTLY 3 questions" in second_user
    assert "Wrong number of questions" in second_user


def test_generator_max_attempts_configurable() -> None:
    """Caller can set max_attempts to override default (3)."""
    examples = [_retrieved("ex-1")]
    retriever = FakeRetriever(examples)

    bad = "not json"
    llm = _SequencedMockClient([bad, bad, bad, bad, bad])
    gen = Generator(retriever=retriever, llm_client=llm)

    try:
        gen.generate(GenerationRequest(topic="x", language="en", count=1), max_attempts=2)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "after 2 attempts" in str(exc)
        assert len(llm.calls) == 2  # respected the override


def test_generator_handles_long_phrase_choices_in_mcq() -> None:
    """MCQ now absorbs what used to be TMC — phrase-length choices work fine."""
    examples = [_retrieved("ex-1", question_text="Q1")]
    retriever = FakeRetriever(examples)
    # The canned response uses full-sentence choices (former TMC style)
    long_phrase_mcq = CANNED_TMC_RESPONSE.replace(
        '"correct_answers"', '"correct_answers"'  # no-op, just reuses the canned data
    )
    llm = MockClient(canned_response=long_phrase_mcq)
    gen = Generator(retriever=retriever, llm_client=llm)

    quiz = gen.generate(GenerationRequest(
        topic="past perfect tense", language="en", count=1,
        question_type="MULTIPLE_CHOICE",
    ))
    assert len(quiz.questions) == 1
    q = quiz.questions[0]
    assert q.question_type == "MULTIPLE_CHOICE"
    # Choice can be a full-sentence phrase — our schema allows it
    assert q.correct_answers[0] in q.choices


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    passed = 0
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
            passed += 1
    print(f"All {passed} Stage 5 (v1) tests passed.")
