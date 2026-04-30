"""Cross-encoder reranker for Stage 4 retrieval.

Two-stage retrieval pattern:
  1. Chroma + BGE-M3 (bi-encoder) — fast cosine similarity, recall-focused
  2. Cross-encoder reranker         — slower but more accurate scoring,
                                      precision-focused on the candidate pool

The reranker reads (query, candidate) pairs as one input and produces a
relevance score per pair. This is more accurate than independent embeddings
because the model can attend to the query while reading each candidate.

Default model: BAAI/bge-reranker-v2-m3
  - Multilingual (incl. Arabic) — same family as our BGE-M3 embedder
  - ~600 MB download (cached after first run)
  - ~30-40ms per pair on Mac MPS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankerConfig:
    """Plain dataclass — no Pydantic dependency at the model layer."""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    batch_size: int = 16


def _resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Reranker:
    """Wraps a cross-encoder model for query×candidate relevance scoring.

    Lazy-imports sentence_transformers so unit tests can mock this layer
    without pulling the heavy ML stack.
    """

    def __init__(self, config: RerankerConfig) -> None:
        from sentence_transformers import CrossEncoder

        self.config = config
        self.device = _resolve_device(config.device)
        self._model = CrossEncoder(config.model_name, device=self.device)

    def score(self, query: str, candidate_texts: list[str]) -> list[float]:
        """Return a relevance score per candidate (higher = more relevant)."""
        if not candidate_texts:
            return []
        pairs = [[query, text] for text in candidate_texts]
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        # CrossEncoder returns numpy array; coerce to plain list of floats.
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        text_attr: str = "search_text",
    ) -> list[Any]:
        """Reorder candidates by reranker score, descending.

        Each candidate object must have a `text_attr` attribute (default:
        'search_text') used as the document side of the (query, doc) pair.
        Returns a NEW list — does not mutate input.
        """
        if not candidates:
            return []
        texts = [getattr(c, text_attr, "") for c in candidates]
        scores = self.score(query, texts)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [c for c, _ in ranked]
