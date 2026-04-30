"""Thin wrapper around sentence-transformers with prefix-aware encoding.

The indexer uses `encode_passages()` at build time.
The retriever (Stage 4) will use `encode_query()`.

Both sides must stay in sync on prefixes — which is why the prefix lives
in config (models.yaml), not hardcoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingModelConfig:
    """Plain dataclass so we don't force Pydantic on the embedding layer."""
    name: str
    embedding_dim: int
    batch_size: int = 16
    device: str = "auto"
    normalize_embeddings: bool = True
    passage_prefix: str = ""
    query_prefix: str = ""


def _resolve_device(preference: str) -> str:
    """Auto-detect the best available device if 'auto' is requested."""
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


class EmbeddingModel:
    """Lazy-loaded sentence-transformers wrapper.

    The heavy import happens only when the class is instantiated, so
    tests that mock the embedder can import this module without
    pulling in torch/transformers.
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        self.config = config
        self.device = _resolve_device(config.device)
        self._model = SentenceTransformer(config.name, device=self.device)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode_passages(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ):
        """Encode documents (corpus side). Applies passage_prefix."""
        prefixed = [self.config.passage_prefix + t for t in texts]
        return self._model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

    def encode_query(self, text: str):
        """Encode one query. Applies query_prefix. Used by the retriever."""
        return self._model.encode(
            self.config.query_prefix + text,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
