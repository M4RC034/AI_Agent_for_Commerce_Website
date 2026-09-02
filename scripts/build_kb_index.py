#!/usr/bin/env python3
"""Build the zh-TW knowledge base FAISS index.

Adapted from the source project's ``src/build_text_index.py`` — same encode →
normalise → ``IndexFlatIP`` → write flow. Two things changed: the corpus is the
customer-service knowledge base rather than the product catalog, and the index
dimension is read from the model instead of assuming 384.

    python scripts/build_kb_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from agent.retrieval.chunker import chunk_corpus  # noqa: E402
from agent.retrieval.embedder import Embedder  # noqa: E402
from agent.retrieval.index import KBIndex  # noqa: E402


def main() -> int:
    if not config.KB_DIR.exists():
        print(f"錯誤：找不到知識庫目錄 {config.KB_DIR}", file=sys.stderr)
        return 1

    print(f"讀取知識庫：{config.KB_DIR}")
    chunks = chunk_corpus(config.KB_DIR)
    if not chunks:
        print("錯誤：知識庫沒有任何可索引的段落。", file=sys.stderr)
        return 1

    docs = len({c.doc for c in chunks})
    print(f"  {docs} 份文件 → {len(chunks)} 個 chunk")

    print(f"載入編碼模型：{config.EMBED_MODEL}")
    embedder = Embedder()
    print(f"  維度 {embedder.dimension}")

    print("編碼中…")
    embeddings = embedder.encode_passages(
        [c.embed_text for c in chunks], show_progress=True
    )

    print("建立 FAISS 索引（IndexFlatIP，向量已正規化）…")
    index = KBIndex.build(chunks, embeddings)
    index.save(config.KB_INDEX_PATH, config.KB_CHUNKS_PATH)

    print(f"完成：{config.KB_INDEX_PATH}")
    print(f"      {config.KB_CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
