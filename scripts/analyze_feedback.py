"""Layer 3 evaluation — human judgements joined to the runs that produced them.

`scripts/analyze_runs.py` answers "how fast, how close" from machine data.
This answers "was it any good", which is the question eval/RESULTS.md says is
currently unanswered:

    "LLM generation quality ... is currently judged manually."

Feedback rows (logs/feedback.jsonl, written by POST /feedback) carry a
`request_id`. Runs (logs/runs.jsonl, written by every /quiz/generate) carry
the same id plus the filters, the retrieved chunks and their distances. Joining
them turns a pile of thumbs into evidence about WHY a question was bad.

The headline output is the distance comparison: if downvoted questions were
generated from chunks that sat near the quality floor and upvoted ones came
from close chunks, then `llm.default_max_distance` is too loose — and you can
finally set it per language with evidence instead of the notebook estimate
recorded in configs/models.yaml.

Usage:
    python -m scripts.analyze_feedback
    python -m scripts.analyze_feedback --feedback logs/feedback.jsonl \\
                                       --runs logs/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! skipping malformed line {n} of {path}")
    return rows


def _index_runs(runs: list[dict]) -> dict[str, dict]:
    """request_id -> run entry. Later entries win (a retried id is rare)."""
    index: dict[str, dict] = {}
    for run in runs:
        response = run.get("response") or {}
        rid = response.get("request_id")
        if rid:
            index[rid] = run
    return index


def _distances(run: dict) -> list[float]:
    retrieval = (run.get("response") or {}).get("retrieval") or []
    out: list[float] = []
    for chunk in retrieval:
        try:
            out.append(float(chunk["distance"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _fmt(values: list[float]) -> str:
    if not values:
        return "no data"
    return (
        f"n={len(values):<5} best={min(values):.3f}  "
        f"mean={statistics.mean(values):.3f}  worst={max(values):.3f}"
    )


def analyze(feedback_path: Path, runs_path: Path) -> int:
    feedback = _read_jsonl(feedback_path)
    if not feedback:
        print(f"No feedback yet in {feedback_path}.")
        print("Generate a quiz at /ui and click 👍 / 👎 on a few questions.")
        return 1

    runs = _index_runs(_read_jsonl(runs_path))
    line = "─" * 68

    print(f"\n{line}\nFEEDBACK — {len(feedback)} judgement(s)\n{line}")
    verdicts = Counter(f.get("verdict", "?") for f in feedback)
    up, down = verdicts.get("up", 0), verdicts.get("down", 0)
    total = up + down
    rate = f"{100 * down / total:.0f}%" if total else "—"
    print(f"  up: {up}   down: {down}   down-rate: {rate}")

    # ---- per cell ---------------------------------------------------------
    print(f"\n{line}\nBY CELL (language × subject)\n{line}")
    cells: dict[tuple, Counter] = defaultdict(Counter)
    for f in feedback:
        cells[(f.get("language") or "?", f.get("subject") or "?")][f.get("verdict")] += 1
    print(f"  {'cell':<24} {'up':>4} {'down':>5} {'down-rate':>10}")
    for (lang, subject), counts in sorted(
        cells.items(), key=lambda kv: -(kv[1]["down"])
    ):
        n = counts["up"] + counts["down"]
        pct = f"{100 * counts['down'] / n:.0f}%" if n else "—"
        print(f"  {lang + ' × ' + subject:<24} {counts['up']:>4} "
              f"{counts['down']:>5} {pct:>10}")

    # ---- the question the whole file exists to answer ---------------------
    print(f"\n{line}\nRETRIEVAL DISTANCE vs VERDICT\n{line}")
    joined = 0
    by_verdict: dict[str, list[float]] = {"up": [], "down": []}
    worst_by_verdict: dict[str, list[float]] = {"up": [], "down": []}
    for f in feedback:
        run = runs.get(f.get("request_id") or "")
        if run is None:
            continue
        joined += 1
        dists = _distances(run)
        if not dists:
            continue
        by_verdict.setdefault(f.get("verdict"), []).extend(dists)
        worst_by_verdict.setdefault(f.get("verdict"), []).append(max(dists))

    if not joined:
        print("  No feedback row joined to a run.")
        print("  Either runs.jsonl predates the feedback, or the UI didn't send")
        print("  request_id. Check that logs/runs.jsonl is being written")
        print("  (RUNS_LOG_PATH must be writable — it defaults to a container path).")
    else:
        print(f"  joined {joined}/{len(feedback)} judgements to their run\n")
        for verdict in ("up", "down"):
            print(f"  all chunks   [{verdict:<4}] {_fmt(by_verdict.get(verdict, []))}")
        print()
        for verdict in ("up", "down"):
            print(f"  worst chunk  [{verdict:<4}] {_fmt(worst_by_verdict.get(verdict, []))}")

        up_worst = worst_by_verdict.get("up") or []
        down_worst = worst_by_verdict.get("down") or []
        if len(up_worst) >= 5 and len(down_worst) >= 5:
            gap = statistics.mean(down_worst) - statistics.mean(up_worst)
            print()
            if gap > 0.03:
                print(f"  → Downvoted questions were built from chunks {gap:.3f} farther")
                print("    away on average. That is the signature of a distance floor")
                print("    set too loose: tighten llm.default_max_distance and re-measure.")
            elif gap < -0.03:
                print("  → Downvoted questions had CLOSER chunks than upvoted ones.")
                print("    Retrieval isn't the problem here; look at the prompt.")
            else:
                print("  → No meaningful distance gap. Quality is not being decided")
                print("    by retrieval distance — look at the prompt or the model.")
        else:
            need = 5 - min(len(up_worst), len(down_worst))
            print(f"\n  (need ~{max(need, 1)} more judgement(s) per verdict "
                  "before the comparison means anything)")

    # ---- what people actually said ---------------------------------------
    notes = [f for f in feedback if f.get("verdict") == "down" and f.get("note")]
    if notes:
        print(f"\n{line}\nWHY THEY WERE BAD ({len(notes)} note(s))\n{line}")
        for f in notes[-15:]:
            cell = f"{f.get('language') or '?'} × {f.get('subject') or '?'}"
            print(f"\n  [{cell}] {f.get('topic') or '(no topic)'}")
            print(f"    Q: {(f.get('question_text') or '')[:100]}")
            print(f"    → {f['note']}")

    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=Path("logs/feedback.jsonl"))
    parser.add_argument("--runs", type=Path, default=Path("logs/runs.jsonl"))
    args = parser.parse_args()
    raise SystemExit(analyze(args.feedback, args.runs))


if __name__ == "__main__":
    main()
