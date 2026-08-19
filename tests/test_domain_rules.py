"""Tests for src/data/domain_rules.py — Tunisian-corpus subject rules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.domain_rules import apply_subject_language_rule


def test_subject_english_locks_to_en() -> None:
    lang, rule = apply_subject_language_rule(["ENGLISH"], detected_language="fr")
    assert lang == "en"
    assert "locked_by_subject_ENGLISH" in rule


def test_subject_arabic_locks_to_ar() -> None:
    lang, rule = apply_subject_language_rule(["ARABIC"], detected_language="en")
    assert lang == "ar"
    assert "locked_by_subject_ARABIC" in rule


def test_subject_french_locks_to_fr() -> None:
    lang, rule = apply_subject_language_rule(["FRENCH"], detected_language="en")
    assert lang == "fr"
    assert "locked_by_subject_FRENCH" in rule


def test_subject_english_agrees_with_detection() -> None:
    lang, rule = apply_subject_language_rule(["ENGLISH"], detected_language="en")
    assert lang == "en"
    assert rule == "none"


def test_mathematics_forbids_en_latin_defaults_to_fr() -> None:
    lang, rule = apply_subject_language_rule(
        ["MATHEMATICS"],
        detected_language="en",
        text_sample="Les primitives de f(x) sur R",
    )
    assert lang == "fr"
    assert "forbids_en_using_fr" in rule


def test_mathematics_with_arabic_content_overrides_to_ar() -> None:
    lang, rule = apply_subject_language_rule(
        ["MATHEMATICS"],
        detected_language="en",
        text_sample="ما هو الجذر التربيعي للعدد",  # Arabic
    )
    assert lang == "ar"
    assert "forbids_en_using_ar" in rule


def test_physics_en_forbidden_fr_result() -> None:
    lang, rule = apply_subject_language_rule(["PHYSICS"], detected_language="en", text_sample="")
    assert lang == "fr"
    assert "forbids_en" in rule


def test_science_en_forbidden() -> None:
    lang, rule = apply_subject_language_rule(["SCIENCE"], detected_language="en", text_sample="x")
    assert lang == "fr"


def test_mathematics_already_fr_no_override() -> None:
    """Rule doesn't fire when detection already says fr."""
    lang, rule = apply_subject_language_rule(["MATHEMATICS"], detected_language="fr")
    assert lang == "fr"
    assert rule == "none"


def test_mathematics_already_ar_no_override() -> None:
    lang, rule = apply_subject_language_rule(["MATHEMATICS"], detected_language="ar")
    assert lang == "ar"
    assert rule == "none"


def test_no_subject_no_rule() -> None:
    lang, rule = apply_subject_language_rule([], detected_language="en")
    assert lang == "en"
    assert rule == "none"


def test_empty_subject_strings_ignored() -> None:
    lang, rule = apply_subject_language_rule(["", "   ", None], detected_language="en")
    assert lang == "en"
    assert rule == "none"


def test_subject_case_insensitive() -> None:
    """Subjects are matched case-insensitively (uppercased internally)."""
    lang, rule = apply_subject_language_rule(["english"], detected_language="fr")
    assert lang == "en"
    assert "locked" in rule


def test_unknown_subject_falls_through() -> None:
    """Subjects outside our defined rules don't force anything."""
    lang, rule = apply_subject_language_rule(["UNKNOWN_SUBJECT"], detected_language="en")
    assert lang == "en"
    assert rule == "none"


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All domain_rules tests passed.")
