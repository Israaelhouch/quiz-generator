"""Tests for src/data/curriculum_rules.py.

Stdlib-only — the curriculum_rules module has no Pydantic dependency, so
these tests run in minimal environments alongside test_ingest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.curriculum_rules import (
    EXPECTED_LANGUAGES,
    check_compliance,
)


# ---------------------------------------------------------------------------
# Rule table sanity
# ---------------------------------------------------------------------------


def test_rule_table_covers_all_in_scope_math_phases() -> None:
    """Math should have a rule for each of the three in-scope phases."""
    for phase in ("PRIMARY", "MIDDLE", "HIGH"):
        assert ("MATHEMATICS", phase) in EXPECTED_LANGUAGES, (
            f"missing curriculum rule for ('MATHEMATICS', {phase!r})"
        )


def test_math_curriculum_expected_languages() -> None:
    """Tunisian math: primary/middle in Arabic, high in French."""
    assert EXPECTED_LANGUAGES[("MATHEMATICS", "PRIMARY")] == frozenset({"ar"})
    assert EXPECTED_LANGUAGES[("MATHEMATICS", "MIDDLE")]  == frozenset({"ar"})
    assert EXPECTED_LANGUAGES[("MATHEMATICS", "HIGH")]    == frozenset({"fr"})


# ---------------------------------------------------------------------------
# check_compliance — happy paths (no rule applies, or rule satisfied)
# ---------------------------------------------------------------------------


def test_no_subjects_means_no_rule_can_apply() -> None:
    ok, reason = check_compliance(subjects=[], school_phase="PRIMARY", language="ar")
    assert ok and reason == ""

    ok, reason = check_compliance(subjects=None, school_phase="PRIMARY", language="ar")
    assert ok and reason == ""


def test_no_school_phase_means_no_rule_can_apply() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase=None, language="fr",
    )
    assert ok and reason == ""


def test_subject_without_a_rule_passes_through() -> None:
    """ENGLISH/ARABIC/FRENCH have no entry in EXPECTED_LANGUAGES — they're
    handled by domain_rules.py — so check_compliance should be a no-op."""
    ok, reason = check_compliance(
        subjects=["ENGLISH"], school_phase="HIGH", language="en",
    )
    assert ok and reason == ""


def test_math_primary_arabic_is_compliant() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="PRIMARY", language="ar",
    )
    assert ok and reason == ""


def test_math_middle_arabic_is_compliant() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="MIDDLE", language="ar",
    )
    assert ok and reason == ""


def test_math_high_french_is_compliant() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="HIGH", language="fr",
    )
    assert ok and reason == ""


# ---------------------------------------------------------------------------
# check_compliance — violations
# ---------------------------------------------------------------------------


def test_math_primary_french_is_a_violation() -> None:
    """The canonical violation we saw in the audit: high-school polynomial
    content mistagged as PRIMARY_SCHOOL_2ND_GRADE in the source."""
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="PRIMARY", language="fr",
    )
    assert not ok
    assert reason == "curriculum_MATHEMATICS_PRIMARY_violation"


def test_math_middle_french_is_a_violation() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="MIDDLE", language="fr",
    )
    assert not ok
    assert reason == "curriculum_MATHEMATICS_MIDDLE_violation"


def test_math_high_arabic_is_a_violation() -> None:
    ok, reason = check_compliance(
        subjects=["MATHEMATICS"], school_phase="HIGH", language="ar",
    )
    assert not ok
    assert reason == "curriculum_MATHEMATICS_HIGH_violation"


# ---------------------------------------------------------------------------
# check_compliance — edge cases
# ---------------------------------------------------------------------------


def test_subject_case_is_normalized_to_upper() -> None:
    """Source data sometimes stores subjects in non-canonical case
    ('Mathematics', 'mathematics'). The check should handle them."""
    ok, _ = check_compliance(
        subjects=["mathematics"], school_phase="HIGH", language="fr",
    )
    assert ok

    ok, _ = check_compliance(
        subjects=["Mathematics"], school_phase="HIGH", language="ar",
    )
    assert not ok


def test_none_in_subjects_list_is_tolerated() -> None:
    """Defensive: garbage source data can have null subjects in the list."""
    ok, _ = check_compliance(
        subjects=[None, "MATHEMATICS"], school_phase="HIGH", language="fr",
    )
    assert ok


def test_multiple_subjects_any_violation_drops_the_row() -> None:
    """If a row is multi-subject and ANY subject's rule is violated, drop.
    Note: this also catches cross-subject contamination cases where the
    'real' subject is the math co-tag and the primary subject is unrelated."""
    ok, reason = check_compliance(
        subjects=["SCIENCE", "MATHEMATICS"], school_phase="PRIMARY", language="fr",
    )
    assert not ok
    assert "MATHEMATICS" in reason


def test_multiple_subjects_no_rules_apply_means_compliant() -> None:
    ok, _ = check_compliance(
        subjects=["ENGLISH", "ARABIC"], school_phase="HIGH", language="en",
    )
    assert ok


if __name__ == "__main__":
    # Simple runner so this file can be executed without pytest.
    test_rule_table_covers_all_in_scope_math_phases()
    test_math_curriculum_expected_languages()
    test_no_subjects_means_no_rule_can_apply()
    test_no_school_phase_means_no_rule_can_apply()
    test_subject_without_a_rule_passes_through()
    test_math_primary_arabic_is_compliant()
    test_math_middle_arabic_is_compliant()
    test_math_high_french_is_compliant()
    test_math_primary_french_is_a_violation()
    test_math_middle_french_is_a_violation()
    test_math_high_arabic_is_a_violation()
    test_subject_case_is_normalized_to_upper()
    test_none_in_subjects_list_is_tolerated()
    test_multiple_subjects_any_violation_drops_the_row()
    test_multiple_subjects_no_rules_apply_means_compliant()
    print("All curriculum_rules tests passed.")
