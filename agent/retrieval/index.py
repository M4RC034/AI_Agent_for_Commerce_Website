"""FAISS vector store, adapted from the source project's search engine.

Reused from Multimodal Commerce Agent
    ``faiss.IndexFlatIP`` over L2-normalised vectors, so inner product *is*
    cosine similarity — the same choice ``build_text_index.py`` made, for the
    same reason. The ``_fetch_results`` pattern from ``search_engine.py`` is
    kept too: FAISS returns row offsets, and a parallel metadata list turns
    them back into records.

Changed for this agent
    The payload is a KB chunk rather than a product row, and the index
    dimension comes from the embedder instead of a hardcoded constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from agent.retrieval.chunker import Chunk


class KBIndex:
    """A FAISS index plus the chunk metadata its row offsets point into."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk]) -> None:
        self.index = index
        self.chunks = chunks

    # -- build -------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray) -> "KBIndex":
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} vs {embeddings.shape[0]}"
            )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, chunks)

    def save(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # -- load --------------------------------------------------------------

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "KBIndex":
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"知識庫索引尚未建立。請先執行：python scripts/build_kb_index.py\n"
                f"  缺少：{index_path if not index_path.exists() else chunks_path}"
            )
        index = faiss.read_index(str(index_path))
        chunks = [
            Chunk(**json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if index.ntotal != len(chunks):
            raise ValueError(
                f"索引與 chunk 檔不同步（{index.ntotal} vs {len(chunks)}）。請重建索引。"
            )
        return cls(index, chunks)

    # -- search ------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return up to ``k`` (chunk, score) pairs, best first."""
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        hits: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k neighbours exist
                continue
            hits.append((self.chunks[idx], float(score)))
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
