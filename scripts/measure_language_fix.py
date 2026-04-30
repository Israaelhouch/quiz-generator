"""Diagnostic: simulate Stage 2b language detection with and without the
LaTeX-strip fix, on your current flat.jsonl. Report how many rows CHANGE
label as a result of the fix.

Run this before re-running the real pipeline, so you know what the fix
actually does on YOUR data.

Usage:
    python -m scripts.measure_language_fix
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.filters import strip_html_to_plain
from src.data.language import resolve_language
from src.data.latex import normalize_latex
from src.data.normalize import clean_quiz_title


def main() -> None:
    flat_path = Path("data/interim/flat.jsonl")
    if not flat_path.exists():
        raise SystemExit(f"{flat_path} missing — run Stage 2a first.")

    before_label: Counter[str] = Counter()
    after_label: Counter[str] = Counter()
    changed: list[dict] = []

    with flat_path.open("r", encoding="utf-8") as f:
        for line in f:
            flat = json.loads(line)
            question_text = strip_html_to_plain(flat.get("question_text_raw"))
            quiz_title = clean_quiz_title(flat.get("quiz_title_raw"))
            if not question_text:
                continue

            # Current behaviour: detect on LaTeX-laden text
            before_lang, before_src = resolve_language(
                flat.get("language_raw"), question_text, quiz_title,
            )
            # Fixed behaviour: detect on LaTeX-stripped text
            after_lang, after_src = resolve_language(
                flat.get("language_raw"),
                normalize_latex(question_text),
                normalize_latex(quiz_title),
            )

            before_label[str(before_lang)] += 1
            after_label[str(after_lang)] += 1

            if before_lang != after_lang:
                changed.append({
                    "doc_id": flat.get("doc_id", ""),
                    "raw": flat.get("language_raw"),
                    "before": before_lang,
                    "after": after_lang,
                    "before_source": before_src,
                    "after_source": after_src,
                    "text_sample": question_text[:120],
                })

    print(f"Rows processed: {sum(before_label.values())}")
    print("\nLanguage label distribution:")
    print(f"  BEFORE fix: {dict(before_label)}")
    print(f"  AFTER  fix: {dict(after_label)}")
    print(f"\nRows that CHANGE label with the fix: {len(changed)}")

    if changed:
        # Tally how labels moved
        transitions: Counter[str] = Counter()
        for c in changed:
            transitions[f"{c['before']} -> {c['after']}"] += 1
        print("\nTransitions (before -> after):")
        for k, v in transitions.most_common():
            print(f"  {k:25s} {v}")

        print("\nFirst 5 changed rows:")
        for c in changed[:5]:
            print(f"  doc_id={c['doc_id']}")
            print(f"    raw={c['raw']!r}  {c['before']} -> {c['after']}  ({c['before_source']} -> {c['after_source']})")
            print(f"    text: {c['text_sample']}")


if __name__ == "__main__":
    main()
