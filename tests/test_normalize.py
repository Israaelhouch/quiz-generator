"""Tests for Stage 2b — normalization.

Covers language helpers + normalization logic. Stdlib-only so these
run even when Pydantic is unavailable in the sandbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.language import (
    SUPPORTED_LANGUAGES,
    detect_language,
    normalize_language_label,
    resolve_language,
)
from src.data.normalize import (
    _union_preserving_order,
    apply_subject_aliases,
    classify_empty_text_reason,
    clean_quiz_title,
    dedup_key,
    dedup_rows,
    split_choices,
)


# ----- language label normalization -----

def test_normalize_language_label_variants() -> None:
    assert normalize_language_label("english") == "en"
    assert normalize_language_label("English") == "en"
    assert normalize_language_label("EN") == "en"
    assert normalize_language_label("fr") == "fr"
    assert normalize_language_label("français") == "fr"
    assert normalize_language_label("Arabic") == "ar"
    assert normalize_language_label(None) is None
    assert normalize_language_label("spanish") is None
    assert normalize_language_label("") is None


def test_detect_language_arabic_dominance() -> None:
    lang, conf = detect_language("ما هو النظام الشمسي؟ كوكب المريخ")
    assert lang == "ar"
    assert conf > 0.7


def test_detect_language_french() -> None:
    lang, conf = detect_language("Quelle est la capitale de la France? C'est Paris.")
    assert lang == "fr"
    assert conf > 0.4


def test_detect_language_english() -> None:
    lang, conf = detect_language("What is the capital of France? It is Paris.")
    assert lang == "en"
    assert conf > 0.4


def test_detect_language_empty_and_unknown() -> None:
    assert detect_language("")[0] is None
    assert detect_language(None)[0] is None
    # Pure digits — no letters to evaluate.
    assert detect_language("2+2=?")[0] is None


# ----- resolve_language -----

def test_resolve_language_trust_raw_when_consistent() -> None:
    lang, source = resolve_language("english", "What is a pathogen?", "Quiz: Immunity")
    assert lang == "en"
    assert source == "raw"


def test_resolve_language_override_on_script_conflict() -> None:
    # Raw says English but content is pure Arabic — detector must override.
    lang, source = resolve_language(
        "english",
        "ما هي وظيفة الجهاز المناعي في الجسم البشري؟",
        "Quiz: Something",
    )
    assert lang == "ar"
    assert source == "raw_overridden_by_script"


def test_resolve_language_rescue_null_from_text() -> None:
    lang, source = resolve_language(
        None,
        "Quelle est la différence entre mitose et méiose?",
        "Quiz",
    )
    assert lang == "fr"
    assert source == "detected_text"


def test_resolve_language_unknown_when_nothing_works() -> None:
    lang, source = resolve_language(None, "", "")
    assert lang is None
    assert source == "unknown"


# ----- text cleaning -----

def test_clean_quiz_title_strips_prefix_and_html() -> None:
    assert clean_quiz_title("Quiz: <b>Immunity 1</b>") == "Immunity 1"
    assert clean_quiz_title("QUIZ:   Math Basics  ") == "Math Basics"
    assert clean_quiz_title(None) == ""


def test_split_choices_cleans_and_selects_correct() -> None:
    raw = [
        {"answer": "<p>Paris</p>", "isTrue": True, "media": None},
        {"answer": "<p>London</p>", "isTrue": False, "media": None},
        {"answer": "", "isTrue": True, "media": "img.png"},  # media-only correct
    ]
    texts, correct, media = split_choices(raw)
    assert texts == ["Paris", "London", ""]
    assert correct == ["Paris", ""]
    assert media == [None, None, "img.png"]


# ----- subject aliases -----

def test_apply_subject_aliases_maps_and_dedups() -> None:
    aliases = {"PHYSICS_1-MECHANICS": "PHYSICS", "MECHANIC": "PHYSICS"}
    assert apply_subject_aliases(["PHYSICS_1-MECHANICS", "MECHANIC", "PHYSICS"], aliases) == ["PHYSICS"]
    assert apply_subject_aliases(["MATHEMATICS"], aliases) == ["MATHEMATICS"]
    assert apply_subject_aliases([], aliases) == []


# ----- dedup -----

def _mk(doc_id: str, *, text: str, choices: list[str], subjects: list[str] | None = None,
        levels: list[str] | None = None, lang: str = "en", qtype: str = "MULTIPLE_CHOICE") -> dict:
    return {
        "doc_id": doc_id,
        "quiz_id": f"quiz-{doc_id}",
        "quiz_title": "T",
        "language": lang,
        "subjects": subjects or ["MATHEMATICS"],
        "levels": levels or ["L1"],
        "question_type": qtype,
        "multiple_correct_answers": False,
        "question_text": text,
        "choices_text": choices,
        "correct_choices_text": [choices[0]] if choices else [],
        "choices_media": [None] * len(choices),
        "points": 1.0,
        "time": 30,
        "author_name": "x",
        "author_email": "x@y.z",
    }


def test_dedup_key_case_and_order_insensitive_on_choices() -> None:
    a = _mk("a", text="What is 2+2?", choices=["Four", "Five"])
    b = _mk("b", text="what is 2+2?", choices=["five", "FOUR"])
    assert dedup_key(a) == dedup_key(b)


def test_dedup_key_different_when_choices_differ() -> None:
    a = _mk("a", text="What is 2+2?", choices=["Four", "Five"])
    c = _mk("c", text="What is 2+2?", choices=["Four", "Six"])
    assert dedup_key(a) != dedup_key(c)


def test_dedup_rows_merges_subjects_and_levels_across_dupes() -> None:
    a = _mk("a", text="Same Q", choices=["X", "Y"], subjects=["PHYSICS"], levels=["L1"])
    b = _mk("b", text="Same Q", choices=["Y", "X"], subjects=["CHEMISTRY"], levels=["L2"])
    c = _mk("c", text="Different Q", choices=["X", "Y"], subjects=["HISTORY"], levels=["L3"])

    kept, groups, dropped = dedup_rows([a, b, c])
    assert dropped == 1
    assert groups == 1
    kept_by_id = {row["doc_id"]: row for row in kept}
    # Row 'a' is first-seen winner; its subjects+levels should include b's.
    assert "a" in kept_by_id
    assert set(kept_by_id["a"]["subjects"]) == {"PHYSICS", "CHEMISTRY"}
    assert set(kept_by_id["a"]["levels"]) == {"L1", "L2"}
    assert "c" in kept_by_id  # non-duplicate passes through untouched


def test_union_preserving_order_drops_falsy() -> None:
    rows = [{"subjects": ["A", "B"]}, {"subjects": [None, "", "B", "C"]}]
    assert _union_preserving_order(rows, "subjects") == ["A", "B", "C"]


# ----- empty-text classification -----

def test_split_choices_deduplicates_repeated_answers() -> None:
    """Source data has ~31 rows with duplicate choice texts (copy-paste errors).
    split_choices() should keep only the first occurrence of each unique answer.
    """
    raw = [
        {"answer": "<p>Answer A</p>", "isTrue": True, "media": None},
        {"answer": "<p>Answer B</p>", "isTrue": False, "media": None},
        {"answer": "<p>Answer B</p>", "isTrue": False, "media": None},   # duplicate
        {"answer": "<p>Answer A</p>", "isTrue": True, "media": None},    # duplicate (already correct)
    ]
    texts, correct, media = split_choices(raw)
    assert texts == ["Answer A", "Answer B"]
    assert correct == ["Answer A"]
    assert len(media) == 2


def test_split_choices_keeps_empty_strings_separately() -> None:
    """Empty-string choices are NOT dedup'd against each other (they're handled
    by the ghost-row filter). Two empty choices stay as two — preserves position.
    """
    raw = [
        {"answer": "", "isTrue": False, "media": None},
        {"answer": "", "isTrue": False, "media": None},
        {"answer": "Real", "isTrue": True, "media": None},
    ]
    texts, correct, _ = split_choices(raw)
    assert texts == ["", "", "Real"]
    assert correct == ["Real"]


def test_normalize_row_drops_when_dedup_leaves_too_few_choices() -> None:
    """Source row with choices=['2', '2'] dedups to ['2'] → MCQ with 1 choice
    isn't really an MCQ; drop it with reason 'too_few_choices_after_dedup'.
    """
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "dup",
        "quiz_id": "q",
        "quiz_title_raw": "Math",
        "language_raw": "english",
        "subjects": ["MATHEMATICS"],
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": "<p>What is 1+1?</p>",
        "choices_raw": [
            {"answer": "<p>2</p>", "isTrue": True, "media": None},
            {"answer": "<p>2</p>", "isTrue": False, "media": None},   # duplicate of "2"
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }
    normalized, reason, audit = normalize_row(flat, aliases={})
    assert normalized is None
    assert reason == "too_few_choices_after_dedup"


def test_classify_empty_text_reason_image_only() -> None:
    raw = '<p><img src="https://x/img.png" /></p>'
    assert classify_empty_text_reason(raw) == "description_is_image_only"


def test_classify_empty_text_reason_whitespace_plus_image() -> None:
    raw = '<p>&nbsp;</p><p>&nbsp;</p><p><img src="x.png"/></p>'
    assert classify_empty_text_reason(raw) == "description_is_image_only"


def test_mathematics_subject_mislabeled_english_gets_overridden_to_fr() -> None:
    """The domain rule catches the canonical Tunisian-corpus mislabel:
    subject=MATHEMATICS with language_raw="english" forces away from en.
    Even the "Limites et comportement asymptotique" case works now —
    doesn't need langdetect or strong stopword density; subject alone
    triggers the override.
    """
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "m",
        "quiz_id": "q",
        "quiz_title_raw": "Quiz: Limites et comportement asymptotique",
        "language_raw": "english",
        "subjects": ["MATHEMATICS"],    # ← the key signal
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": (
            r"Calculer: \(\lim_{x\to+\infty}\sqrt{x^2-5}-\sqrt{x-3}\) "
            r"a) \(0\) b) \(1\) c) \(+\infty\)"
        ),
        "choices_raw": [
            {"answer": "a", "isTrue": True, "media": None},
            {"answer": "b", "isTrue": False, "media": None},
            {"answer": "c", "isTrue": False, "media": None},
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }
    normalized, reason, audit = normalize_row(flat, aliases={})
    assert reason == ""
    assert normalized is not None
    assert normalized["language"] == "fr"
    # Audit trail should mention the domain rule
    assert "MATHEMATICS" in audit["language_source"]


def test_detect_language_uses_langdetect_when_available() -> None:
    """When langdetect is importable, it should be the primary detector for
    Latin-script text (better accuracy than stopword heuristic).
    This test stubs out _ld_detect_langs to verify the integration plumbing
    regardless of whether the real library is installed in this env.
    """
    from src.data import language as lang_mod
    from unittest.mock import MagicMock

    # Create a fake candidate (mimics langdetect's return shape)
    class _FakeCandidate:
        def __init__(self, lang, prob):
            self.lang = lang
            self.prob = prob

    real_has = lang_mod._HAS_LANGDETECT
    real_fn = lang_mod._ld_detect_langs
    try:
        # Force the code path that calls langdetect
        lang_mod._HAS_LANGDETECT = True
        lang_mod._ld_detect_langs = MagicMock(return_value=[_FakeCandidate("fr", 0.97)])

        result_lang, result_conf = lang_mod.detect_language(
            "Limites et comportement asymptotique des fonctions"
        )
        assert result_lang == "fr"
        assert result_conf > 0.9
    finally:
        lang_mod._HAS_LANGDETECT = real_has
        lang_mod._ld_detect_langs = real_fn


def test_detect_language_falls_back_to_stopwords_when_langdetect_unavailable() -> None:
    """If langdetect is NOT installed, detection falls through to the
    stopword heuristic (existing behaviour preserved).
    """
    from src.data import language as lang_mod

    real_has = lang_mod._HAS_LANGDETECT
    try:
        lang_mod._HAS_LANGDETECT = False
        # French text with enough stopwords for the heuristic
        result_lang, _ = lang_mod.detect_language(
            "Les primitives de la fonction sur R sont calculées"
        )
        assert result_lang == "fr"
    finally:
        lang_mod._HAS_LANGDETECT = real_has


def test_french_title_boosts_detection_when_title_has_stopword_density() -> None:
    """Title + question combination catches math-heavy French when the title
    itself carries enough French function words. Titles with sparse stopwords
    (e.g. "Limites et comportement asymptotique" — only 1 stopword) still
    slip through — that's a fundamental limitation of stopword heuristics
    and requires langdetect to fix.
    """
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "t",
        "quiz_id": "q",
        # Title has "Les", "sur", "les" — multiple stopwords
        "quiz_title_raw": "Quiz: Exercices sur les primitives des fonctions",
        "language_raw": "english",
        "subjects": [],
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": (
            r"Calculer: \(\lim_{x\to+\infty}\sqrt{x^2-5}\) "
            r"a) \(0\) b) \(1\) c) \(+\infty\)"
        ),
        "choices_raw": [
            {"answer": "a", "isTrue": True, "media": None},
            {"answer": "b", "isTrue": False, "media": None},
            {"answer": "c", "isTrue": False, "media": None},
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }
    normalized, reason, audit = normalize_row(flat, aliases={})
    assert reason == ""
    assert normalized is not None
    assert normalized["language"] == "fr"


def test_all_empty_choices_are_dropped() -> None:
    """Ghost-row filter: MCQ with choices=['','',''] → dropped."""
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "ghost",
        "quiz_id": "q",
        "quiz_title_raw": "Quiz: Math",
        "language_raw": "english",
        "subjects": ["MATHEMATICS"],
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": "<p>Question 2</p>",    # placeholder text
        "choices_raw": [
            {"answer": "", "isTrue": True, "media": None},
            {"answer": "", "isTrue": False, "media": None},
            {"answer": "", "isTrue": False, "media": None},
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }
    normalized, reason, audit = normalize_row(flat, aliases={})
    assert normalized is None
    assert reason == "all_choices_empty"


def test_rows_with_real_choices_pass() -> None:
    """Sanity: the ghost filter must NOT reject rows with real choices."""
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "ok",
        "quiz_id": "q",
        "quiz_title_raw": "Quiz: Basic Math",
        "language_raw": "english",
        "subjects": ["MATHEMATICS"],
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": "<p>What is 2+2?</p>",
        "choices_raw": [
            {"answer": "4", "isTrue": True, "media": None},
            {"answer": "3", "isTrue": False, "media": None},
            {"answer": "5", "isTrue": False, "media": None},
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }
    normalized, reason, audit = normalize_row(flat, aliases={})
    assert normalized is not None
    assert reason == ""
    assert normalized["choices_text"] == ["4", "3", "5"]


def test_latex_heavy_french_content_detects_as_french() -> None:
    """Regression: before the LaTeX-strip-before-detection fix, math-heavy
    French rows would stay mislabeled as English because LaTeX tokens
    (left, right, frac, mathbb, ...) diluted the French stopword signal.
    After the fix, detection sees LaTeX-free text and catches the French.
    """
    from src.data.normalize import normalize_row

    flat = {
        "doc_id": "test-1",
        "quiz_id": "quiz-1",
        "quiz_title_raw": "Quiz: Primitives",
        "language_raw": "english",           # mislabeled in source
        "subjects": [],
        "levels": [],
        "question_type": "MULTIPLE_CHOICE",
        "multiple_correct_answers": False,
        "question_text_raw": (
            r"Les primitives de \(f\left(x\right)=\cos^2\left(x\right)"
            r"\times\sin^3\left(x\right)\) sur \(\mathbb{R}\) sont: A) "
            r"\(-\frac{1}{3}\cos^3\left(x\right)+\frac{1}{5}\cos^5\left(x\right)+k\)"
        ),
        "choices_raw": [
            {"answer": "A", "isTrue": True, "media": None},
            {"answer": "B", "isTrue": False, "media": None},
            {"answer": "C", "isTrue": False, "media": None},
        ],
        "points": 1, "time": 30,
        "author_name": None, "author_email": None,
    }

    normalized, reason, audit = normalize_row(flat, aliases={})

    # The row should survive cleaning.
    assert reason == ""
    assert normalized is not None

    # Critical: language gets overridden from 'english' (source label) to 'fr'.
    assert normalized["language"] == "fr"
    assert audit["language_source"] == "raw_overridden_by_script"

    # And the STORED question_text still contains LaTeX (not modified).
    assert "\\frac" in normalized["question_text"] or "\\left" in normalized["question_text"]


def test_classify_empty_text_reason_truly_empty() -> None:
    assert classify_empty_text_reason("") == "empty_description"
    assert classify_empty_text_reason(None) == "empty_description"
    assert classify_empty_text_reason(" ") == "empty_description"
    assert classify_empty_text_reason("<p></p>") == "empty_description"
    assert classify_empty_text_reason("<p>&nbsp;</p>") == "empty_description"


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All Stage 2b tests passed.")
