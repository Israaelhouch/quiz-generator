"""Tests for src/generation/latex_validity.py.

Stdlib-only (no Pydantic, no pylatexenc) — runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.latex_validity import check_latex_validity


# ---------------------------------------------------------------------------
# Valid cases — common patterns the LLM produces correctly
# ---------------------------------------------------------------------------


def test_empty_or_none_is_valid() -> None:
    assert check_latex_validity("") == (True, "")
    assert check_latex_validity(None) == (True, "")


def test_plain_text_no_math_is_valid() -> None:
    assert check_latex_validity("What is 2 plus 3?") == (True, "")


def test_simple_inline_math_is_valid() -> None:
    assert check_latex_validity(r"\(x^2\)") == (True, "")
    assert check_latex_validity(r"\(\frac{1}{2}\)") == (True, "")
    assert check_latex_validity(r"\(\sqrt{x+1}\)") == (True, "")


def test_simple_display_math_is_valid() -> None:
    assert check_latex_validity(r"\[\int_0^1 x \, dx\]") == (True, "")


def test_multiple_math_blocks_in_one_string() -> None:
    text = r"Solve \(x^2 = 4\), then check \(x = \pm 2\)."
    assert check_latex_validity(text) == (True, "")


def test_nested_balanced_braces() -> None:
    assert check_latex_validity(r"\(\frac{\sqrt{x}}{y^2}\)") == (True, "")


def test_escaped_braces_are_literal() -> None:
    """`\\{` and `\\}` are LITERAL braces, not grouping. They shouldn't
    count toward brace balance."""
    # The literal set { x } in math mode
    assert check_latex_validity(r"\(\{x, y\}\)") == (True, "")


# ---------------------------------------------------------------------------
# Invalid cases — what the LLM occasionally produces wrong
# ---------------------------------------------------------------------------


def test_unclosed_inline_math_fails() -> None:
    """Missing \\) is truly broken — the math block stays open and may
    break the rendering of everything after it on the page."""
    ok, reason = check_latex_validity(r"What is \(x^2 ?")
    assert not ok
    assert "unclosed inline math" in reason


def test_extra_close_inline_is_tolerated() -> None:
    """An extra \\) renders as literal text — ugly but local. Don't lose
    a whole 10-question quiz over one cosmetic blemish."""
    ok, _ = check_latex_validity(r"Solve \(x = 3\)\) and check")
    assert ok


def test_unclosed_display_math_fails() -> None:
    ok, reason = check_latex_validity(r"\[\int_0^1 x \, dx")
    assert not ok
    assert "unclosed display math" in reason


def test_extra_close_display_is_tolerated() -> None:
    ok, _ = check_latex_validity(r"\[\int x\,dx\]\] extra")
    assert ok


# ---------------------------------------------------------------------------
# Over-escaped delimiters — the canonical LLM bug we hit in production
# ---------------------------------------------------------------------------


def test_over_escaped_close_inline_is_caught() -> None:
    """The LLM sometimes emits `\\\\)` (two backslashes + paren) instead
    of `\\)` (one + paren). MathJax tokenizes `\\\\)` as `\\\\` (line break)
    + literal `)`, never closing the math block. The walk-based scanner
    must recognise this as 'unclosed math', not as 'balanced delimiters'.
    """
    # `\(\sqrt{2}\\)` — one opener, but the closer is over-escaped, so
    # the scan should count 1 open / 0 close inline.
    ok, reason = check_latex_validity(r"\(\sqrt{2}\\)")
    assert not ok
    assert "unclosed inline math" in reason


def test_over_escaped_open_inline_is_tolerated() -> None:
    """The mirror case: `\\\\(` at the start. Then there's no math at all
    (just literal `\\\\` followed by `(`). The user's text contains literal
    backslashes + parens, not a math expression. Not broken — just looks
    like literal text on the page."""
    # `\\(x^2\)` — no math opens; the `\)` at the end is an "extra close"
    # which we tolerate cosmetically.
    ok, _ = check_latex_validity(r"\\(x^2\)")
    assert ok


def test_mixed_correct_and_over_escaped_in_same_text() -> None:
    """If even ONE math block has an over-escaped closer, we fail —
    the unclosed block would corrupt downstream rendering."""
    text = r"Correct \(x^2\) and broken \(y^2\\)"
    ok, reason = check_latex_validity(text)
    assert not ok
    assert "unclosed" in reason


def test_missing_close_brace_inside_math() -> None:
    ok, reason = check_latex_validity(r"\(\frac{1}{2\)")
    assert not ok
    assert "missing" in reason and "brace" in reason


def test_extra_close_brace_inside_math() -> None:
    ok, reason = check_latex_validity(r"\(x^2}\)")
    assert not ok
    assert "extra closing brace" in reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_text_with_braces_outside_math_does_not_trigger() -> None:
    """Curly-brace balance is only checked INSIDE math blocks. A stray
    { in prose (e.g., quoting code) shouldn't fail the check."""
    text = "The function {x: x>0} is positive."
    assert check_latex_validity(text) == (True, "")


def test_arabic_text_with_correct_math() -> None:
    """Sanity: non-Latin scripts surrounding math expressions are fine."""
    text = r"الكسر \(\frac{1}{3}\) يساوي تقريبا 0,33"
    assert check_latex_validity(text) == (True, "")


def test_arabic_text_with_broken_math_still_caught() -> None:
    text = r"الكسر \(\frac{1}{3 يساوي تقريبا 0,33"
    ok, reason = check_latex_validity(text)
    assert not ok


if __name__ == "__main__":
    # Simple runner so this file can be executed without pytest.
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            obj()
    print("All latex_validity tests passed.")
