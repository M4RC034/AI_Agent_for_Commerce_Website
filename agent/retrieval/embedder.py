"""Sentence embedding, adapted from the source project's ``build_text_index.py``.

Reused from Multimodal Commerce Agent
    The call shape — ``SentenceTransformer.encode`` with
    ``normalize_embeddings=True`` and ``convert_to_numpy=True``, cast to float32
    so FAISS accepts it, with normalised vectors making inner product equal
    cosine similarity.

Changed for this agent
    The checkpoint. ``all-MiniLM-L6-v2`` has effectively no Chinese capability
    and would return near-arbitrary neighbours for the zh-TW knowledge base
    (FR22). Dimensionality is now read from the model instead of the hardcoded
    ``dimension = 384`` in the original script, which would silently mismatch
    after the swap.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from agent import config

#: e5-family models require asymmetric prefixes; bge-m3 does not. Keyed by
#: substring so a pinned revision of either still matches.
_E5_PREFIXES = {"query": "query: ", "passage": "passage: "}


class Embedder:
    """Lazily-loaded wrapper around a multilingual sentence encoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBED_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        # Loaded on first use, then held. bge-m3 is ~2.2GB and slow to load on
        # CPU — paying that cost at import time would make `--help` take a
        # minute and would make the eval harness unusable.
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        # sentence-transformers 6.x renamed this; keep both paths so the build
        # script works across the 5.x pinned in the source project's
        # requirements.txt and the 6.x installed for this agent.
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        return getter()

    @property
    def _needs_prefix(self) -> bool:
        return "e5" in self.model_name.lower()

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if not self._needs_prefix:
            return texts
        return [_E5_PREFIXES[kind] + t for t in texts]

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode KB chunks for indexing."""
        return self.model.encode(
            self._prefix(texts, "passage"),
            batch_size=16,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single search query, shaped ``(1, dim)`` for FAISS."""
        return self.model.encode(
            self._prefix([text], "query"),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
