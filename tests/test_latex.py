"""Tests for LaTeX normalization (regex fallback path).

These tests target the zero-dependency regex normalizer so they pass
in any environment. When pylatexenc is installed locally it will be
used in production — and may produce slightly different (often better)
output, but the regex fallback covers the patterns we care about most.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.latex import _regex_normalize, contains_latex


def test_inline_math_delimiters_are_unwrapped() -> None:
    out = _regex_normalize(r"Calculer \(x^2\) ici")
    # Result should not contain the \( \) markers.
    assert "\\(" not in out and "\\)" not in out
    assert "x^2" in out
    assert "Calculer" in out


def test_frac_becomes_division() -> None:
    out = _regex_normalize(r"\(\frac{1}{2}\)")
    assert "1" in out and "2" in out
    assert "\\frac" not in out


def test_sqrt_becomes_function_form() -> None:
    out = _regex_normalize(r"\(\sqrt{x}\)")
    assert "sqrt" in out
    assert "x" in out
    assert "\\sqrt" not in out


def test_known_functions_lose_backslash() -> None:
    out = _regex_normalize(r"\(\sin(x) + \cos(x)\)")
    assert "sin" in out and "cos" in out
    assert "\\sin" not in out and "\\cos" not in out


def test_left_right_keywords_dropped_keeping_delimiters() -> None:
    out = _regex_normalize(r"\(\left(x+1\right)^2\)")
    assert "(x+1)" in out
    assert "left" not in out and "right" not in out


def test_special_symbols_become_words() -> None:
    out = _regex_normalize(r"\(\int f(x) dx\)")
    assert "integral" in out
    out2 = _regex_normalize(r"\(\pi r^2\)")
    assert "pi" in out2
    out3 = _regex_normalize(r"\(\infty\)")
    assert "infinity" in out3


def test_remaining_backslash_commands_stripped() -> None:
    out = _regex_normalize(r"\(\partial f / \partial x\)")
    # \partial isn't in our special-symbols list — should be stripped.
    assert "\\partial" not in out
    assert "f" in out and "x" in out


def test_nested_frac_simple_case() -> None:
    out = _regex_normalize(r"\(\frac{1}{x^2}\)")
    assert "\\frac" not in out
    assert "1" in out and "x^2" in out


def test_real_french_math_sample() -> None:
    raw = (
        r"Si la fonction F définie sur ] \(0,+\infty\) [ par "
        r"\(F\left(x\right)=x^3+6x^2+\frac{1}{2x^2}+3\) est une primitive de \(f\)"
    )
    out = _regex_normalize(raw)
    # Crucial: French words preserved, LaTeX gone.
    assert "fonction" in out
    assert "primitive" in out
    assert "infinity" in out  # \infty became infinity
    assert "F(x)" in out or "F (x)" in out
    assert "\\frac" not in out
    assert "\\left" not in out and "\\right" not in out


def test_arabic_math_sample() -> None:
    raw = r"هي قوة للعدد \(2^5\)"
    out = _regex_normalize(raw)
    # Arabic preserved, LaTeX cleaned.
    assert "هي قوة للعدد" in out
    assert "2^5" in out
    assert "\\(" not in out


def test_text_without_latex_is_unchanged_modulo_whitespace() -> None:
    raw = "What is the capital of France?"
    out = _regex_normalize(raw)
    assert out == raw


def test_empty_inputs() -> None:
    assert _regex_normalize("") == ""


def test_contains_latex_detector() -> None:
    assert contains_latex(r"\(x^2\)")
    assert contains_latex(r"\frac{1}{2}")
    assert contains_latex(r"\sin(x)")
    assert not contains_latex("just plain text")
    assert not contains_latex(None)
    assert not contains_latex("")


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All LaTeX normalizer tests passed.")
