"""CrossEncoder reranking, adapted from the source project's pipeline.

Reused from Multimodal Commerce Agent
    The whole two-stage shape and the rerank block itself. ``backend/main.py``
    built ``[query, doc]`` pairs, called ``CrossEncoder.predict`` on all of
    them, zipped scores back onto the candidates, sorted descending and sliced
    the top N. That is exactly what happens below.

Changed for this agent
    The checkpoint — ``ms-marco-MiniLM-L-6-v2`` scores English pairs only.
    And the scores no longer just *order* the candidates: they also feed the
    sufficiency verdict the agent has to act on (FR4), so they are returned
    rather than discarded after sorting.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from agent import config
from agent.retrieval.chunker import Chunk


class Reranker:
    """Lazily-loaded cross-encoder for second-stage precision."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.RERANK_MODEL
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rescore FAISS candidates against the query and keep the best ``top_n``.

        The returned scores are the cross-encoder's, not FAISS's — they are what
        the sufficiency threshold in ``config`` is calibrated against.
        """
        if not candidates:
            return []

        top_n = top_n or config.RERANK_TOP_N
        pairs = [[query, f"{c.title}｜{c.section}\n{c.text}"] for c, _ in candidates]
        scores = self.model.predict(pairs)

        scored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]
