"""Lightweight LaTeX validity check for LLM-generated math content.

The LLM (Gemini Flash) usually emits well-formed LaTeX, but occasionally
produces output that won't render on the frontend (MathJax/KaTeX):

  - Missing closing brace:        \\(\\frac{1}{2\\)
  - Unclosed inline math:         \\(x^2
  - Over-escaped close delimiter: \\(...\\\\) ← `\\\\)` is `\\\\` + `)`,
                                                NOT a closing `\\)`. The
                                                math block stays open.

Why a walk-based parser instead of regex?
  A naive regex `\\\\\\)` matches the substring `\\)` even when it's
  preceded by another backslash. But LaTeX tokenizes `\\\\)` as `\\\\`
  (a line break) followed by literal `)`. So a regex-based counter
  over-counts and misses real bugs.

  The walk below treats `\\\\` as an escape pair and `\\(`/`\\)`/`\\[`/`\\]`
  as real delimiters, matching how MathJax actually parses input.

Pure stdlib, no Pydantic, no pylatexenc dependency.
"""

from __future__ import annotations


def _check_braces(s: str) -> tuple[bool, str]:
    """Check that `{` and `}` are balanced inside a math block. Literal
    braces `\\{` and `\\}` are escape pairs and don't count."""
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            # Any backslash-escape pair — skip both characters. Covers \{, \},
            # \\, \frac, \sqrt, etc. None of them count toward brace depth.
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False, "extra closing brace }"
        i += 1
    if depth != 0:
        return False, f"missing {depth} closing brace(s)"
    return True, ""


def _scan_math(text: str) -> tuple[list[tuple[str, str]], int, int, int, int]:
    """Walk through `text` and find all UN-ESCAPED math delimiters and the
    content of each math block.

    A backslash followed by another backslash (`\\\\`) is treated as a
    LaTeX literal/escape — the next char is NOT part of a delimiter. This
    is what makes the over-escaped case `\\(...\\\\)` correctly register
    as "open but never closed."

    Returns:
        (blocks, open_inline, close_inline, open_display, close_display)
        - blocks: list of (kind, content) where kind is "inline" or "display"
          and content is the text BETWEEN the delimiters of a properly
          closed block. Unclosed blocks contribute to the counts but no
          block content.
    """
    blocks: list[tuple[str, str]] = []
    open_i = close_i = open_d = close_d = 0
    i = 0
    n = len(text)
    # While inside a math block, remember its kind + content-start index.
    current: tuple[str, int] | None = None

    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            # \\\\ (two backslashes) — escape pair. Skip both. Any
            # following ( ) [ ] is then a LITERAL paren/bracket, not a
            # math delimiter.
            if nxt == "\\":
                i += 2
                continue
            # Real math delimiters
            if nxt == "(":
                open_i += 1
                if current is None:
                    current = ("inline", i + 2)
                i += 2
                continue
            if nxt == ")":
                close_i += 1
                if current is not None and current[0] == "inline":
                    blocks.append(("inline", text[current[1]:i]))
                    current = None
                i += 2
                continue
            if nxt == "[":
                open_d += 1
                if current is None:
                    current = ("display", i + 2)
                i += 2
                continue
            if nxt == "]":
                close_d += 1
                if current is not None and current[0] == "display":
                    blocks.append(("display", text[current[1]:i]))
                    current = None
                i += 2
                continue
            # Any other backslash-command (\frac, \sqrt, \pi, ...) — skip
            # the escape pair, it's content not a delimiter.
            i += 2
            continue
        i += 1

    return blocks, open_i, close_i, open_d, close_d


def check_latex_validity(text: str | None) -> tuple[bool, str]:
    """Check whether the LaTeX in `text` is syntactically renderable.

    Returns:
        (True, "")          — valid (or no LaTeX present)
        (False, "<reason>") — TRULY broken; caller should reject / retry.

    Lenient on cosmetic issues (extra `\\)` or `\\]` at the end of text
    render as literal characters, not page-breaking). Strict on truly
    broken cases:

      - Unclosed inline/display math   → math block stays open, may
                                          break rendering of the rest
                                          of the page.
      - Unbalanced braces inside math  → that math expression fails.
      - Over-escaped delimiters
        (`\\\\)` instead of `\\)`)    → silently leaves math unclosed;
                                          the walk-based scan in
                                          `_scan_math` catches it
                                          because `\\\\` is treated as
                                          an escape pair and the
                                          following `)` is just a
                                          literal char.
    """
    if not text:
        return True, ""

    blocks, open_i, close_i, open_d, close_d = _scan_math(text)

    # Hard fail: unclosed math (more openings than closings). The block
    # stays open and may corrupt downstream rendering.
    if open_i > close_i:
        return False, (
            f"unclosed inline math: {open_i} opening \\( vs only "
            f"{close_i} closing \\). Close every \\( with a \\). "
            "Make sure to write \\) (one backslash + paren), not \\\\) "
            "(two backslashes + paren)."
        )
    if open_d > close_d:
        return False, (
            f"unclosed display math: {open_d} opening \\[ vs only "
            f"{close_d} closing \\]. Close every \\[ with a \\]."
        )

    # Hard fail: brace imbalance inside a math block. The renderer will
    # show an error indicator on that one expression.
    for kind, content in blocks:
        ok, reason = _check_braces(content)
        if not ok:
            delim = "\\(...\\)" if kind == "inline" else "\\[...\\]"
            return False, f"in {delim} block: {reason}: {content[:80]!r}"

    # Soft pass: extra closing delimiters render as literal text. Not
    # ideal but not page-breaking. Leave them through.
    return True, ""
