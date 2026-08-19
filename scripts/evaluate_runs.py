"""Layer 2 evaluation — uses Gemini as a judge to score every run in
`logs/runs.jsonl` across 12 quality dimensions.

For each run, sends Gemini:
  - the requested topic + language
  - the retrieved chunks (text only)
  - the generated questions

Asks Gemini to rate each dimension 1-5 with a one-line justification.
Writes results to `logs/eval_report.csv`.

KNOWN BIAS: Gemini judging Gemini's own output is mildly self-flattering,
especially on linguistic / math correctness dimensions. Treat the scores as
a relative ranking (which runs are weakest?), not as absolute quality.

Usage:
    python -m scripts.evaluate_runs
    python -m scripts.evaluate_runs path/to/runs.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# Load .env at the project root so GEMINI_API_KEY (and friends) are
# available when running this script natively on the host. Best-effort —
# silently skipped if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for a Tunisian-school RAG quiz generator.

Score the GENERATED QUIZ below across 12 quality dimensions, on a 1-5 scale
(1 = very poor, 5 = excellent).

Respond with a single JSON object — no other text, no markdown — with this
exact shape:

{{
  "R1_retrieval_relevance": <int 1-5>,
  "R2_retrieval_coverage": <int 1-5>,
  "R3_level_appropriateness": <int 1-5>,
  "G1_linguistic_correctness": <int 1-5>,
  "G2_register_appropriate": <int 1-5>,
  "G3_formatting_clean": <int 1-5>,
  "C1_topic_faithfulness": <int 1-5>,
  "C2_conceptual_correctness": <int 1-5>,
  "C3_factual_accuracy": <int 1-5>,
  "C4_choice_distinctness": <int 1-5>,
  "C5_explanation_quality": <int 1-5>,
  "Q1_question_variety": <int 1-5>,
  "Q2_difficulty_distribution": <int 1-5>,
  "summary": "<one short sentence — main observation>"
}}

Scoring guide:
- R1 — Are the retrieved chunks actually about the topic? Look at chunk
  question_text vs the requested topic.
- R2 — Do retrieved chunks span the concept (different angles), or are
  they all clustered on one narrow aspect?
- R3 — Are the chunks at a reasonable curriculum level (school-grade)?
- G1 — Is the language grammatically correct, no made-up words?
- G2 — Is the register appropriate (MSA for Arabic, formal "vous" French,
  not slangy/dialect/too informal)?
- G3 — Are LaTeX symbols clean (no broken \\frac), choices well-formed?
- C1 — Does each question stay on the requested topic, or drift?
- C2 — Does each question test the right pedagogical concept (not a
  related-but-different one)? E.g. for "تمييز", does it actually test
  تمييز, not حال or نعت.
- C3 — Are the marked correct_answers actually correct? For math, do the
  numbers work out? For factual questions, is the fact right?
- C4 — Is exactly ONE choice truly correct? (Not multiple equally valid.)
  Are distractors plausible-but-wrong, not obviously wrong filler?
- C5 — Does the explanation actually justify the answer (vs being generic
  "this is correct because of grammar rules")?
- Q1 — Do the N questions test different aspects, or repeat the same idea
  with minor changes?
- Q2 — Is there a reasonable spread of easy/medium/hard, or all same level?

REQUEST:
  topic: {topic}
  language: {language}
  count requested: {count}

RETRIEVED CHUNKS ({n_chunks} of them, with distance — lower = closer match):
{retrieval_block}

GENERATED QUESTIONS ({n_questions}):
{questions_block}

Now produce the JSON. Output JSON only.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "R1_retrieval_relevance",
    "R2_retrieval_coverage",
    "R3_level_appropriateness",
    "G1_linguistic_correctness",
    "G2_register_appropriate",
    "G3_formatting_clean",
    "C1_topic_faithfulness",
    "C2_conceptual_correctness",
    "C3_factual_accuracy",
    "C4_choice_distinctness",
    "C5_explanation_quality",
    "Q1_question_variety",
    "Q2_difficulty_distribution",
]


def _format_retrieval(retrieval: list) -> str:
    if not retrieval:
        return "(none)"
    out: list[str] = []
    for i, c in enumerate(retrieval[:15], 1):
        d = c.get("distance", "?")
        d_str = f"{d:.3f}" if isinstance(d, (int, float)) else str(d)
        title = c.get("quiz_title", "")
        text = (c.get("question_text") or "")[:200]
        out.append(f"[{i}] dist={d_str}  {title}\n    Q: {text}")
    return "\n".join(out)


def _format_questions(questions: list) -> str:
    if not questions:
        return "(none)"
    out: list[str] = []
    for i, q in enumerate(questions, 1):
        text = q.get("question_text", "")
        choices = q.get("choices", []) or []
        correct = q.get("correct_answers", []) or []
        difficulty = q.get("difficulty", "?")
        explanation = q.get("explanation", "")
        out.append(
            f"[{i}] (difficulty={difficulty})\n"
            f"   Q: {text}\n"
            f"   Choices: {choices}\n"
            f"   Correct: {correct}\n"
            f"   Explanation: {explanation}"
        )
    return "\n\n".join(out)


def _build_prompt(run: dict) -> str:
    req = run["request"]
    resp = run["response"]
    return JUDGE_PROMPT_TEMPLATE.format(
        topic=req["topic"],
        language=req["language"],
        count=req["count"],
        n_chunks=len(resp.get("retrieval") or []),
        n_questions=len(resp.get("questions") or []),
        retrieval_block=_format_retrieval(resp.get("retrieval") or []),
        questions_block=_format_questions(resp.get("questions") or []),
    )


def _judge_one(run: dict, model: str) -> dict | None:
    """Send one run to Gemini, get back the score JSON."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(run)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,                              # judging — be consistent
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        print(f"⚠ Judge call failed for topic '{run['request']['topic'][:40]}...': {exc}",
              file=sys.stderr)
        return None


def evaluate(log_path: Path, csv_path: Path, model: str = "gemini-2.5-flash") -> None:
    """Read all runs, judge each, write CSV."""
    if not log_path.exists():
        print(f"❌ Log not found: {log_path}")
        sys.exit(1)

    with log_path.open(encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]

    if not runs:
        print("❌ Empty log.")
        sys.exit(1)

    print(f"⏳ Judging {len(runs)} runs with {model}...")
    print(f"   (~5-10s per run; total ~{len(runs)*8//60} min)")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["topic", "language", "duration_s", "n_questions", "n_retrieval"]
        + DIMENSIONS
        + ["total", "summary"]
    )

    rows: list[dict] = []

    for i, run in enumerate(runs, 1):
        topic = run["request"]["topic"][:50]
        print(f"  [{i}/{len(runs)}] {topic}", end="", flush=True)

        scores = _judge_one(run, model=model)
        row: dict = {
            "topic": run["request"]["topic"],
            "language": run["request"].get("language", ""),
            "duration_s": run["response"].get("duration_seconds", ""),
            "n_questions": len(run["response"].get("questions") or []),
            "n_retrieval": len(run["response"].get("retrieval") or []),
        }
        if scores is None:
            for d in DIMENSIONS:
                row[d] = ""
            row["total"] = ""
            row["summary"] = "JUDGE FAILED"
            print(" — ❌ failed")
        else:
            total = 0
            for d in DIMENSIONS:
                v = scores.get(d, "")
                row[d] = v
                if isinstance(v, int):
                    total += v
            row["total"] = total
            row["summary"] = scores.get("summary", "")
            print(f" — total {total}/{5*len(DIMENSIONS)}")
        rows.append(row)

    # Write CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Wrote {len(rows)} rows to {csv_path}")

    # ---- Summary ----
    print(f"\n{'─' * 70}")
    print("  SUMMARY")
    print("─" * 70)

    valid_rows = [r for r in rows if isinstance(r["total"], int)]
    if valid_rows:
        totals = [r["total"] for r in valid_rows]
        max_total = 5 * len(DIMENSIONS)
        print(f"\nOverall: {len(valid_rows)} judged successfully")
        print(f"  avg total:  {sum(totals)/len(totals):.1f} / {max_total}")
        print(f"  min total:  {min(totals)} / {max_total}")
        print(f"  max total:  {max(totals)} / {max_total}")

        # Per-dimension averages
        print("\nAvg score per dimension (max 5):")
        for d in DIMENSIONS:
            vals = [r[d] for r in valid_rows if isinstance(r[d], int)]
            if vals:
                avg = sum(vals) / len(vals)
                bar = "█" * int(avg)
                short = d.split("_", 1)[1].replace("_", " ")[:30]
                print(f"  {d[:3]} {short:<32s} {bar:<5s} {avg:.2f}")

        # Bottom 5
        print("\n🔻 Bottom 5 runs (review these manually):")
        bottom = sorted(valid_rows, key=lambda r: r["total"])[:5]
        for r in bottom:
            print(f"  {r['total']:>3}  | {r['topic'][:60]}")
            print(f"       summary: {r['summary'][:90]}")

        # Top 5
        print("\n🔺 Top 5 runs:")
        top = sorted(valid_rows, key=lambda r: r["total"], reverse=True)[:5]
        for r in top:
            print(f"  {r['total']:>3}  | {r['topic'][:60]}")

    failures = [r for r in rows if r["summary"] == "JUDGE FAILED"]
    if failures:
        print(f"\n⚠ {len(failures)} runs failed to judge — see CSV.")

    print(f"\n💡 Open the CSV: open {csv_path}")
    print()


if __name__ == "__main__":
    log_arg = sys.argv[1] if len(sys.argv) > 1 else "logs/runs.jsonl"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "logs/eval_report.csv"
    model_arg = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
    evaluate(Path(log_arg), Path(out_arg), model=model_arg)
