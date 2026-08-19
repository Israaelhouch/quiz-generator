"""Tests for the cross-encoder reranker (Stage 4 enhancement).

The CrossEncoder model is heavy (sentence-transformers + torch + 600 MB
download). These tests exercise the reranker logic with an injected mock
model — no actual ML stack needed in the sandbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.reranker import Reranker, RerankerConfig


# ---------------------------------------------------------------------------
# Mock model — predict() returns canned scores per pair
# ---------------------------------------------------------------------------


class _FakeCrossEncoder:
    """Minimal stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls: list = []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls.append({"pairs": pairs, "batch_size": batch_size})
        # Score each (query, doc) pair using the canned dict — keyed by doc text.
        return [self.scores_by_text.get(doc, 0.0) for _, doc in pairs]


def _make_reranker_with_fake_model(scores: dict[str, float]) -> Reranker:
    """Bypass the heavy __init__ and inject a fake model."""
    r = Reranker.__new__(Reranker)
    r.config = RerankerConfig()
    r.device = "cpu"
    r._model = _FakeCrossEncoder(scores)
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_returns_one_per_candidate() -> None:
    r = _make_reranker_with_fake_model({"doc A": 0.9, "doc B": 0.4, "doc C": 0.7})
    scores = r.score(query="anything", candidate_texts=["doc A", "doc B", "doc C"])
    assert scores == [0.9, 0.4, 0.7]


def test_score_empty_candidates_returns_empty_list() -> None:
    r = _make_reranker_with_fake_model({})
    assert r.score("query", []) == []


def test_rerank_orders_by_score_descending() -> None:
    """High-score candidates come first."""

    class _Cand:
        def __init__(self, doc_id: str, search_text: str) -> None:
            self.doc_id = doc_id
            self.search_text = search_text

    candidates = [
        _Cand("a", "doc A"),     # score 0.4
        _Cand("b", "doc B"),     # score 0.9 ← should win
        _Cand("c", "doc C"),     # score 0.7
    ]
    r = _make_reranker_with_fake_model({"doc A": 0.4, "doc B": 0.9, "doc C": 0.7})

    reranked = r.rerank(query="anything", candidates=candidates)
    assert [c.doc_id for c in reranked] == ["b", "c", "a"]


def test_rerank_empty_input() -> None:
    r = _make_reranker_with_fake_model({})
    assert r.rerank("q", []) == []


def test_rerank_uses_search_text_attribute_by_default() -> None:
    """Default text_attr is 'search_text'."""

    class _Cand:
        def __init__(self, search_text: str, distance: float = 0.0) -> None:
            self.search_text = search_text
            self.distance = distance

    candidates = [_Cand("doc A"), _Cand("doc B")]
    r = _make_reranker_with_fake_model({"doc A": 0.3, "doc B": 0.8})

    reranked = r.rerank("q", candidates)
    assert reranked[0].search_text == "doc B"
    assert reranked[1].search_text == "doc A"


def test_rerank_uses_custom_text_attribute() -> None:
    """text_attr can be overridden (useful for testing or unusual schemas)."""

    class _Cand:
        def __init__(self, qtext: str) -> None:
            self.qtext = qtext

    candidates = [_Cand("alpha"), _Cand("beta")]
    r = _make_reranker_with_fake_model({"alpha": 0.1, "beta": 0.9})

    reranked = r.rerank("q", candidates, text_attr="qtext")
    assert [c.qtext for c in reranked] == ["beta", "alpha"]


def test_rerank_does_not_mutate_input() -> None:
    """rerank returns a NEW list — original order preserved."""

    class _Cand:
        def __init__(self, doc_id: str, search_text: str) -> None:
            self.doc_id = doc_id
            self.search_text = search_text

    candidates = [_Cand("a", "x"), _Cand("b", "y")]
    original_order = list(candidates)
    r = _make_reranker_with_fake_model({"x": 0.1, "y": 0.9})

    _ = r.rerank("q", candidates)
    assert candidates == original_order  # input unchanged


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All reranker tests passed.")
