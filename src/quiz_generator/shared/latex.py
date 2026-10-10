"""LaTeX-to-plain-text normalization for embedding input.

Why: embedding models don't understand LaTeX markup. A question like
"Calculer la dérivée de \\(\\sin(x)\\)" looks to the embedder like the word
"dérivée" followed by a pile of meaningless backslash-tokens. Normalizing
to "Calculer la dérivée de sin(x)" lets the embedder see three real words.

Two engines:
  - pylatexenc if installed (handles nested structures, complex LaTeX)
  - regex fallback (simple, zero-dependency, covers common patterns)
"""

from __future__ import annotations

import re

try:
    from pylatexenc.latex2text import LatexNodes2Text

    _PYLATEX_CONVERTER = LatexNodes2Text(math_mode="text", strict_latex_spaces=False)
    _HAS_PYLATEX = True
except ImportError:
    _PYLATEX_CONVERTER = None
    _HAS_PYLATEX = False


_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*?)\}\s*\{([^{}]*?)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*?)\}")
_LEFT_RIGHT_RE = re.compile(r"\\(left|right)\s*")
_KNOWN_FUNCTIONS_RE = re.compile(
    r"\\(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan|sinh|cosh|tanh|"
    r"log|ln|exp|lim|max|min|det|gcd|mod|Pr)\b"
)
_REMAINING_BACKSLASH_CMD_RE = re.compile(r"\\[a-zA-Z]+\s*")
_WHITESPACE_RE = re.compile(r"\s+")

_SPECIAL_SYMBOLS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\infty\b"), "infinity"),
    (re.compile(r"\\pi\b"), "pi"),
    (re.compile(r"\\theta\b"), "theta"),
    (re.compile(r"\\alpha\b"), "alpha"),
    (re.compile(r"\\beta\b"), "beta"),
    (re.compile(r"\\gamma\b"), "gamma"),
    (re.compile(r"\\delta\b"), "delta"),
    (re.compile(r"\\epsilon\b"), "epsilon"),
    (re.compile(r"\\sigma\b"), "sigma"),
    (re.compile(r"\\mu\b"), "mu"),
    (re.compile(r"\\lambda\b"), "lambda"),
    (re.compile(r"\\omega\b"), "omega"),
    (re.compile(r"\\int\b"), "integral"),
    (re.compile(r"\\sum\b"), "sum"),
    (re.compile(r"\\prod\b"), "product"),
    (re.compile(r"\\cdot\b"), "*"),
    (re.compile(r"\\times\b"), "x"),
    (re.compile(r"\\div\b"), "/"),
    (re.compile(r"\\leq?\b"), "<="),
    (re.compile(r"\\geq?\b"), ">="),
    (re.compile(r"\\neq?\b"), "!="),
    (re.compile(r"\\pm\b"), "+/-"),
    (re.compile(r"\\to\b"), "to"),
    (re.compile(r"\\rightarrow\b"), "to"),
    (re.compile(r"\\in\b"), "in"),
    (re.compile(r"\\forall\b"), "for all"),
    (re.compile(r"\\exists\b"), "exists"),
]


def normalize_latex(text: str | None) -> str:
    """Convert LaTeX-laden text to plain text suitable for embedding.

    Uses pylatexenc when available (better accuracy on complex LaTeX);
    otherwise falls back to regex rules covering common patterns.
    Empty/None input returns empty string.
    """
    if not text:
        return ""
    if _HAS_PYLATEX:
        try:
            result = _PYLATEX_CONVERTER.latex_to_text(text)
            return " ".join(result.split()).strip()
        except Exception:
            # Fall through to regex path on any pylatexenc failure.
            pass
    return _regex_normalize(text)


def _regex_normalize(text: str) -> str:
    """Zero-dependency LaTeX normalizer covering common patterns."""
    # Unwrap inline and display math delimiters — keep the content.
    text = _INLINE_MATH_RE.sub(lambda m: " " + m.group(1) + " ", text)
    text = _DISPLAY_MATH_RE.sub(lambda m: " " + m.group(1) + " ", text)

    # \frac{a}{b} → (a)/(b). Apply twice to catch simple nesting.
    for _ in range(2):
        text = _FRAC_RE.sub(r"(\1)/(\2)", text)

    # \sqrt{x} → sqrt(x)
    text = _SQRT_RE.sub(r"sqrt(\1)", text)

    # \left( \right) etc → drop the keyword, keep the delimiter
    text = _LEFT_RIGHT_RE.sub("", text)

    # Known function commands: keep the name, drop the backslash
    text = _KNOWN_FUNCTIONS_RE.sub(r"\1", text)

    # Special symbols map to words
    for pattern, replacement in _SPECIAL_SYMBOLS:
        text = pattern.sub(replacement, text)

    # Strip any remaining \cmd occurrences
    text = _REMAINING_BACKSLASH_CMD_RE.sub(" ", text)

    # Strip remaining braces (they're structural noise without content)
    text = text.replace("{", " ").replace("}", " ")

    # Collapse whitespace
    return _WHITESPACE_RE.sub(" ", text).strip()


def contains_latex(text: str | None) -> bool:
    """Quick check whether a string contains LaTeX-like markup."""
    if not text:
        return False
    return bool(re.search(r"\\[a-zA-Z]+|\\\(|\\\)|\\\[|\\\]|\\frac|\\sqrt", text))


def strip_latex_for_detection(text: str | None) -> str:
    """LaTeX stripper tuned for language detection.

    Unlike `normalize_latex`, this does NOT substitute LaTeX commands with
    English words (e.g. `\\to` → "to", `\\infty` → "infinity"). Those
    substitutions help the EMBEDDING model understand math, but they pollute
    STOPWORD-based language detection by injecting fake English tokens.

    This stripper removes all backslash commands entirely, unwraps math
    delimiters, and strips braces — leaving just the natural-language prose
    (if any) plus bare variables and numbers.
    """
    if not text:
        return ""
    # Unwrap inline and display math delimiters — keep content.
    text = _INLINE_MATH_RE.sub(lambda m: " " + m.group(1) + " ", text)
    text = _DISPLAY_MATH_RE.sub(lambda m: " " + m.group(1) + " ", text)
    # Strip ALL backslash commands (no substitution)
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    # Strip leftover structural braces
    text = text.replace("{", " ").replace("}", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()
