"""Stage 6 — CLI shim around QuizPipeline.

Usage:
    python -m src.pipeline "primitives des fonctions" \\
        --language fr --subject MATHEMATICS --count 3
    python -m src.pipeline "past tense" --language en --format json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any


# Single fixed location for --save-run. Sits at the project root (i.e. CWD
# when you invoke the CLI) so you can find it in Finder / `ls` without
# hunting through /tmp. Overwritten every run.
SAVED_RUN_PATH = Path("last_run.json")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("topic", help="Topic / free-text query for the quiz")
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--ready", type=Path, default=Path("data/processed/ready.jsonl"))
    p.add_argument("--language", required=True, choices=["en", "fr", "ar"])
    p.add_argument("--count", type=int, default=5,
                   help="How many new questions to generate (default: 5)")
    p.add_argument(
        "--question-type",
        default="MULTIPLE_CHOICE",
        choices=["MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"],
    )
    p.add_argument("--subject", help="Optional retrieval filter (uppercased)")
    p.add_argument(
        "--levels",
        help="Comma-separated levels for retrieval filter "
             "(e.g. HIGH_SCHOOL_4TH_GRADE_MATHEMATICS)",
    )
    p.add_argument("--few-shot", type=int, default=None,
                   help="Override how many retrieval examples to feed the LLM "
                        "(default: from llm.default_few_shot_count in config)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override LLM temperature (default: from config)")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Override retry budget (default: from config)")
    p.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format. 'human' is rendered text, 'json' is machine-readable.",
    )
    p.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Print the retrieved few-shot chunks (full text) before the quiz. "
             "Human format only — JSON format always includes retrieval.",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help=f"Save the full run (retrieval + generated quiz) to "
             f"./{SAVED_RUN_PATH} in the project folder, overwritten each run. "
             "Useful when stdout is too long to scroll.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_retrieval_human(retrieval: list, *, topic: str) -> str:
    """Render the retrieved few-shot chunks for terminal viewing.

    Full text by default — when you're inspecting a bad output, you want to
    see the entire question + every choice + every correct answer that the
    LLM saw, not a truncated preview.
    """
    lines: list[str] = []
    line = "=" * 90
    lines.append(line)
    lines.append(f"RETRIEVED CHUNKS — what BGE-M3 fed to the LLM as few-shot "
                 f"examples (topic: {topic!r})")
    lines.append(f"count: {len(retrieval)}")
    lines.append(line)

    if not retrieval:
        lines.append("\n  (no chunks retrieved — Generator will raise)")
        lines.append(line)
        return "\n".join(lines)

    for i, c in enumerate(retrieval, start=1):
        lines.append("")
        lines.append(f"[{i}] doc_id={c.doc_id}   distance={c.distance:+.4f}")
        lines.append(f"    quiz_title : {c.quiz_title}")
        lines.append(f"    language   : {c.language}   "
                     f"type: {c.question_type}   "
                     f"subjects: {c.subjects}")
        lines.append(f"    levels     : {c.levels}")
        if c.author_name:
            lines.append(f"    author     : {c.author_name}")
        lines.append(f"    Q: {c.question_text}")
        if c.choices:
            for ch in c.choices:
                mark = "*" if ch in (c.correct_answers or []) else " "
                lines.append(f"        {mark} {ch}")
        else:
            lines.append(f"    Answer(s): {', '.join(c.correct_answers or [])}")

    lines.append("")
    lines.append(line)
    return "\n".join(lines)


def retrieval_to_dict(retrieval: list) -> list[dict]:
    """Serialise a list of RetrievedQuestion into plain dicts."""
    out: list[dict] = []
    for c in retrieval:
        out.append({
            "doc_id": c.doc_id,
            "quiz_id": c.quiz_id,
            "quiz_title": c.quiz_title,
            "language": c.language,
            "question_type": c.question_type,
            "question_text": c.question_text,
            "choices": list(c.choices or []),
            "correct_answers": list(c.correct_answers or []),
            "subjects": list(c.subjects or []),
            "levels": list(c.levels or []),
            "multiple_correct_answers": c.multiple_correct_answers,
            "author_name": c.author_name,
            "author_email": c.author_email,
            "distance": c.distance,
        })
    return out


def save_run_to_file(
    *,
    quiz: Any,
    retrieval: list,
    topic: str,
    language: str,
    subject: str | None = None,
    level: str | None = None,
    path: Path = SAVED_RUN_PATH,
) -> Path:
    """Write a complete debugging snapshot of one pipeline run to disk.

    Contains BOTH the retrieved chunks (what BGE-M3 fed the LLM) and the
    generated quiz (what the LLM produced). One file, overwritten each run,
    so you can always answer both:
        - 'what did Qwen produce?'        → look at `quiz.questions`
        - 'what was Qwen looking at?'     → look at `retrieval`
    from the same artifact.
    """
    payload = {
        "topic": topic,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "subject": subject,
        "level": level,
        "retrieval": retrieval_to_dict(retrieval),
        "quiz": {
            "questions": [q.model_dump() for q in quiz.questions],
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def render_human(quiz: Any, *, topic: str) -> str:
    """Pretty-print a GeneratedQuiz for terminal viewing."""
    lines: list[str] = []
    line = "=" * 90
    lines.append(line)
    lines.append(f"QUIZ — topic: {topic!r}")
    lines.append(f"language: {quiz.language}   subject: {quiz.subject or '-'}   "
                 f"level: {quiz.level or '-'}")
    lines.append(f"questions: {len(quiz.questions)}")
    lines.append(line)

    for i, q in enumerate(quiz.questions, start=1):
        lines.append("")
        lines.append(f"[{i}]  ({q.question_type}"
                     + (f", {q.difficulty}" if q.difficulty else "")
                     + (", multi-correct" if q.multiple_correct_answers else "")
                     + ")")
        lines.append(f"     Q: {q.question_text}")
        if q.choices:
            for c in q.choices:
                mark = "*" if c in q.correct_answers else " "
                lines.append(f"        {mark} {c}")
        else:
            # FILL_IN_THE_BLANKS — show correct answers explicitly
            lines.append(f"     Answer(s): {', '.join(q.correct_answers)}")
        if q.explanation:
            lines.append(f"     Why: {q.explanation}")

    lines.append("")
    lines.append(line)
    return "\n".join(lines)


def render_json(quiz: Any, *, topic: str, retrieval: list | None = None) -> str:
    """Serialise GeneratedQuiz + topic context (and optionally retrieval).

    Stage 7 eval will need both the generated quiz and the retrieval used to
    produce it, so we include retrieval whenever it's provided. When the CLI
    is in JSON mode the pipeline always passes its `last_retrieval`.
    """
    payload = {
        "topic": topic,
        "language": quiz.language,
        "subject": quiz.subject,
        "level": quiz.level,
        "questions": [q.model_dump() for q in quiz.questions],
    }
    if retrieval is not None:
        payload["retrieval"] = retrieval_to_dict(retrieval)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Lazy import — keeps `--help` snappy by not loading pydantic / models.
    from src.pipeline.orchestrator import QuizPipeline

    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    print("Loading pipeline (this also loads the embedder and reranker)…",
          file=sys.stderr)

    pipeline = QuizPipeline(config_path=args.config, ready_jsonl_path=args.ready)

    quiz = pipeline.generate(
        topic=args.topic,
        language=args.language,
        count=args.count,
        question_type=args.question_type,
        subject=args.subject,
        levels=levels,
        few_shot_count=args.few_shot,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
    )

    retrieval = pipeline.last_retrieval

    # Optional save-to-disk (project folder) regardless of output format.
    if args.save_run:
        path = save_run_to_file(
            quiz=quiz,
            retrieval=retrieval,
            topic=args.topic,
            language=args.language,
            subject=args.subject,
            level=(levels[0] if levels else None),
        )
        # Print the absolute path so the user always knows exactly where
        # to find the file, regardless of where they invoked the CLI from.
        print(f"[run saved to: {path.resolve()}]", file=sys.stderr)

    if args.format == "json":
        # JSON always includes retrieval — Stage 7 eval needs it.
        print(render_json(quiz, topic=args.topic, retrieval=retrieval))
    else:
        if args.show_retrieval:
            print(render_retrieval_human(retrieval, topic=args.topic))
        print(render_human(quiz, topic=args.topic))
