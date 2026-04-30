"""Language normalization + detection — Stage 2b helper.

Kept separate so it can be unit-tested without Pydantic and reused
later by the retriever if we choose to consolidate detection code.

Detection strategy (in order):
  1. Arabic script dominance → Arabic (fast, always reliable)
  2. `langdetect` (char-n-gram ML detector) if installed — preferred for
     Latin-script en/fr disambiguation because it works even on sparse
     prose where our stopword heuristic struggles.
  3. Stopword-ratio heuristic as fallback when langdetect is unavailable
     or fails on very short text.
"""

from __future__ import annotations

import re


# langdetect is optional — detection falls back gracefully without it.
try:
    from langdetect import detect_langs as _ld_detect_langs
    from langdetect import DetectorFactory as _ld_DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException as _LangDetectException

    _ld_DetectorFactory.seed = 42   # deterministic output
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    _ld_detect_langs = None
    _LangDetectException = Exception


# Canonical language aliases. Keys are case-insensitive.
_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
    "french": "fr",
    "fr": "fr",
    "francais": "fr",
    "français": "fr",
    "arabic": "ar",
    "ar": "ar",
    "العربية": "ar",
}

SUPPORTED_LANGUAGES = {"en", "fr", "ar"}

ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_SCRIPT_RE = re.compile(r"[A-Za-zÀ-ÿ]")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ']+")

_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "are", "was", "were", "be",
    "a", "an", "on", "at", "by", "from", "as", "or", "for", "with",
    "what", "which", "who", "how", "when", "where", "why", "this",
    "that", "these", "those", "it", "its",
}
_FR_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "et", "ou", "est", "sont",
    "dans", "pour", "avec", "sur", "par", "qui", "que", "où", "un",
    "une", "au", "aux", "ce", "cette", "ces", "se", "son", "sa",
    "ses", "il", "elle", "ils", "elles", "nous", "vous", "mais",
    "comme", "quand", "quoi", "pas", "plus",
}
_FR_HINT_CHARS = set("àâçéèêëîïôùûüÿœæ")


def normalize_language_label(raw: str | None) -> str | None:
    """Map a raw label string to 'en'/'fr'/'ar' or None. Case-insensitive."""
    if not raw:
        return None
    return _LANGUAGE_ALIASES.get(raw.strip().lower())


def detect_language(text: str | None) -> tuple[str | None, float]:
    """Detector. Returns (language_code_or_None, confidence in [0, 1]).

    Order: Arabic-script dominance first (fast), then `langdetect` if
    available, then stopword heuristic as fallback.
    """
    if not text:
        return (None, 0.0)

    ar_count = len(ARABIC_SCRIPT_RE.findall(text))
    lat_count = len(LATIN_SCRIPT_RE.findall(text))

    # Arabic script dominates → Arabic. (faster + more accurate than langdetect
    # here because Arabic Unicode range is a decisive signal.)
    if ar_count >= 5 and ar_count > lat_count * 1.2:
        return ("ar", min(0.98, 0.70 + ar_count / 400.0))
    if ar_count > 0 and lat_count == 0:
        return ("ar", 0.85)

    # Try langdetect for Latin-script content before falling to stopwords.
    # Only accept en/fr/ar results; any other language is out-of-scope.
    if _HAS_LANGDETECT and lat_count >= 3:
        try:
            candidates = _ld_detect_langs(text)
            for candidate in candidates:
                if candidate.lang in {"en", "fr", "ar"}:
                    # langdetect confidence is already well-calibrated
                    return (candidate.lang, min(0.98, float(candidate.prob)))
        except _LangDetectException:
            pass  # fall through to stopword fallback

    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        if ar_count > 0:
            return ("ar", 0.5)
        return (None, 0.0)

    total = len(tokens)
    en_hits = sum(1 for t in tokens if t in _EN_STOPWORDS)
    fr_hits = sum(1 for t in tokens if t in _FR_STOPWORDS)
    en_score = en_hits / total
    fr_score = fr_hits / total
    has_fr_chars = any(ch in _FR_HINT_CHARS for ch in text.lower())

    if fr_score >= en_score * 1.2 and (fr_score >= 0.04 or has_fr_chars):
        confidence = min(0.95, 0.55 + fr_score * 2.0 + (0.10 if has_fr_chars else 0.0))
        return ("fr", confidence)
    if en_score > fr_score * 1.2 and en_score >= 0.04:
        confidence = min(0.95, 0.55 + en_score * 2.0)
        return ("en", confidence)

    # Latin text with no clear stopword signal — weak English guess.
    if lat_count > 0 and ar_count == 0:
        return ("en", 0.40)
    if ar_count > 0:
        return ("ar", 0.50)
    return (None, 0.0)


def resolve_language(
    raw_label: str | None,
    question_text: str,
    quiz_title: str,
) -> tuple[str | None, str]:
    """Return (final_language, source_tag).

    source_tag ∈ {"raw", "raw_overridden_by_script", "detected_text",
                  "detected_title", "unknown"}.

    Logic:
      1. Map raw label through alias table.
      2. If raw exists and content script disagrees with high confidence,
         override with detection.
      3. If raw is missing/unrecognized, try detecting from question_text,
         then quiz_title, then give up.
    """
    normalized = normalize_language_label(raw_label)
    detected_text, conf_text = detect_language(question_text)
    detected_title, conf_title = detect_language(quiz_title)

    if normalized:
        if detected_text and detected_text != normalized and conf_text >= 0.70:
            return (detected_text, "raw_overridden_by_script")
        return (normalized, "raw")

    if detected_text and conf_text >= 0.40:
        return (detected_text, "detected_text")
    if detected_title and conf_title >= 0.40:
        return (detected_title, "detected_title")
    return (None, "unknown")
