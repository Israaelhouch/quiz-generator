"""Domain-specific rules for the Tunisian educational corpus.

Language is highly predictable from subject in this dataset:

  ENGLISH   subject = English content (students learning English)
  ARABIC    subject = Arabic content   (students learning Arabic)
  FRENCH    subject = French content   (students learning French)

  MATHEMATICS, PHYSICS, CHEMISTRY, SCIENCE, COMPUTER_SCIENCE,
  HISTORY, TECHNIQUE, ECONOMICS = French OR Arabic, NEVER English

This prior is stronger than any general-purpose language detector for
this corpus. A MATHEMATICS question mislabeled 'en' is virtually always
wrong — subject tells us so, regardless of what the raw label says.

This module is intentionally Tunisian-specific. If you re-use this
codebase with a different educational system, redefine these mappings.
"""

from __future__ import annotations


# Subjects whose content language is fixed by the subject itself.
# Locking one of these overrides any detector or raw label disagreement.
SUBJECT_LANGUAGE_LOCKED: dict[str, str] = {
    "ENGLISH": "en",
    "ARABIC": "ar",
    "FRENCH": "fr",
}

# Subjects where English is impossible in our corpus.
# Everything else — math, science, etc. — is taught in fr or ar only.
SUBJECTS_NO_ENGLISH: frozenset[str] = frozenset({
    "MATHEMATICS",
    "PHYSICS",
    "CHEMISTRY",
    "SCIENCE",
    "COMPUTER_SCIENCE",
    "HISTORY",
    "TECHNIQUE",
    "ECONOMICS",
})


def apply_subject_language_rule(
    subjects: list[str] | None,
    detected_language: str | None,
    text_sample: str | None = None,
) -> tuple[str | None, str]:
    """Apply the domain subject→language rule to the detected language.

    Returns (final_language, rule_applied). `rule_applied` is "none" when
    no rule fired, or a short string naming which rule triggered (useful
    for audit stats).

    - If primary subject is language-locked (ENGLISH/ARABIC/FRENCH) and
      detected_language disagrees, override to the locked language.
    - If primary subject is in SUBJECTS_NO_ENGLISH and detected_language
      is "en", override: use "ar" if text contains Arabic script, else "fr"
      (French is the Tunisian default for scientific content).
    - Otherwise keep the detected_language as-is.
    """
    if not subjects:
        return detected_language, "none"

    # Use the first non-empty subject as primary
    primary: str | None = None
    for s in subjects:
        if s is None:
            continue
        candidate = str(s).strip().upper()
        if candidate:
            primary = candidate
            break
    if not primary:
        return detected_language, "none"

    # Rule 1: language-locked subjects
    if primary in SUBJECT_LANGUAGE_LOCKED:
        expected = SUBJECT_LANGUAGE_LOCKED[primary]
        if detected_language != expected:
            return expected, f"locked_by_subject_{primary}"
        return detected_language, "none"

    # Rule 2: subjects where English is impossible
    if primary in SUBJECTS_NO_ENGLISH and detected_language == "en":
        # Decide en→fr or en→ar based on whether content has Arabic script
        from src.data.language import ARABIC_SCRIPT_RE
        has_arabic = bool(ARABIC_SCRIPT_RE.search(text_sample or ""))
        forced = "ar" if has_arabic else "fr"
        return forced, f"subject_{primary}_forbids_en_using_{forced}"

    return detected_language, "none"
