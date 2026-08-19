"""Curriculum-compliance rules.

The Tunisian school curriculum constrains which language a subject is
taught in at each phase. Math is the canonical case: primary and middle
school math is taught in Arabic, high school math in French. A row whose
(subject, school_phase, language) doesn't match the curriculum is either
mistagged at the source or genuinely out-of-curriculum content (e.g., an
international-school question). Either way it doesn't belong in our
Tunisian-curriculum index.

This module exists separately from `domain_rules.py` because:
  - `domain_rules.py` handles subject→language locks where the subject
    ALONE determines the language (ENGLISH → en, ARABIC → ar, FRENCH → fr).
  - `curriculum_rules.py` handles subject+phase→language constraints,
    where the subject does NOT determine the language by itself —
    grade level matters. Math is the only subject in current scope with
    this property, but future subjects (PHYSICS, SCIENCE, CHEMISTRY) will
    follow the same pattern.

Pure stdlib — no Pydantic. Wired into `normalize.py` as a drop reason so
violations show up in the normalize stats audit.
"""

from __future__ import annotations


# Map (subject_upper, school_phase) → set of acceptable language codes.
#
# When a row's subject + phase matches a key here, its language MUST be in
# the value set, otherwise the row is dropped.
#
# When a row matches NONE of the keys (e.g., ENGLISH/ARABIC/FRENCH subjects),
# this module is a no-op — those subjects are already constrained by
# `domain_rules.py`.
#
# To add a new rule:
#   ("PHYSICS", "HIGH"): frozenset({"fr"}),
EXPECTED_LANGUAGES: dict[tuple[str, str], frozenset[str]] = {
    # Tunisian math curriculum:
    #   Primary & Middle math → Arabic
    #   High school math      → French
    # (Preparatory & Licence levels are out of current scope; their rule
    # would be added here when those phases enter scope.)
    ("MATHEMATICS", "PRIMARY"): frozenset({"ar"}),
    ("MATHEMATICS", "MIDDLE"):  frozenset({"ar"}),
    ("MATHEMATICS", "HIGH"):    frozenset({"fr"}),
}


def check_compliance(
    subjects: list[str] | None,
    school_phase: str | None,
    language: str | None,
) -> tuple[bool, str]:
    """Check whether (subjects, school_phase, language) satisfies the
    curriculum rules in EXPECTED_LANGUAGES.

    Returns:
        (True, "")   — no rule applies, or all applicable rules are satisfied
        (False, "curriculum_{SUBJECT}_{PHASE}_violation")
                     — at least one applicable rule was violated; row should
                       be dropped. The reason string is short and snake_case
                       so it slots cleanly into the normalize-stats `dropped`
                       counter alongside other drop reasons.

    Behaviour notes:
      - Empty subjects or missing school_phase → no rule can apply, returns
        compliant. The upstream pipeline is responsible for catching
        those as separate issues if it wants.
      - Multiple subjects: if any subject has a rule and that rule is
        violated, the row is dropped. The first violating subject wins
        the reason string (deterministic, depends on input order).
    """
    if not subjects or school_phase is None:
        return True, ""

    for subj in subjects:
        if subj is None:
            continue
        key = (str(subj).strip().upper(), school_phase)
        expected = EXPECTED_LANGUAGES.get(key)
        if expected is None:
            continue  # no rule for this (subject, phase) — skip
        if language in expected:
            continue  # rule satisfied — keep checking other subjects
        return False, f"curriculum_{key[0]}_{school_phase}_violation"

    return True, ""
