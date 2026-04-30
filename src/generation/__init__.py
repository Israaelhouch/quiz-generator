"""Stage 5 — Generation.

Turn retrieved examples (Stage 4) into NEW quiz questions via an LLM.

Minimal first version:
  - English only
  - MULTIPLE_CHOICE only
  - Single-shot generation (no retry)
  - No regurgitation check
Follow-ups add fr/ar, FITB/TMC, retry, dedup-check.
"""
