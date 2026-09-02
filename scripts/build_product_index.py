#!/usr/bin/env python3
"""Build the CLIP product index from the Kaggle Amazon catalog.

    python scripts/build_product_index.py

Pipeline, each step measured rather than assumed (see README):

    42,675 CSV rows
      -> dedupe on title          8,808   (79% of rows are duplicate listings;
                                           one battery appears 744 times)
      -> CLIP zero-shot filter    ~8,071  (drops protection plans, toys, pet
                                           supplies — 91.6% retained, vs 78.2%
                                           for the source project's keyword
                                           classifier, which also mislabels
                                           "ASURION Laptop Protection Plan" as
                                           a laptop)
      -> CLIP text index          ~50s

Data: Amazon Products Sales Dataset 42K+ Items (2025), CC BY-NC 4.0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from agent.retrieval.clip_store import (  # noqa: E402
    IN_DOMAIN,
    OUT_OF_DOMAIN,
    ClipEncoder,
    ProductIndex,
)

RAW_CSV = config.BASE_DIR / "data" / "raw" / "amazon_products_sales_data_uncleaned.csv"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"錯誤：找不到 {RAW_CSV}", file=sys.stderr)
        print("請先下載資料集：", file=sys.stderr)
        print("  kaggle datasets download -d ikramshah512/"
              "amazon-products-sales-dataset-42k-items-2025 -p data/raw --unzip",
              file=sys.stderr)
        return 1

    print(f"讀取 {RAW_CSV.name}")
    df = pd.read_csv(RAW_CSV).dropna(subset=["title", "image_url"])
    rows = len(df)

    # Dedupe first — indexing the raw rows would fill the index with duplicate
    # vectors and return the same product repeatedly in every result set.
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    print(f"  {rows:,} 列 → 去重後 {len(df):,} 項獨立商品")

    encoder = ClipEncoder()
    print(f"載入 OpenCLIP（device={encoder.device}）…")

    titles = df["title"].tolist()
    t0 = time.monotonic()
    embeddings = encoder.encode_texts(titles)
    print(f"  編碼 {len(titles):,} 筆標題，耗時 {time.monotonic() - t0:.0f}s")

    # Zero-shot filter, using the same gate vectors the query path uses.
    labels = IN_DOMAIN + OUT_OF_DOMAIN
    gate = encoder.encode_texts(
        [f"a product photo of {c}" for c in IN_DOMAIN]
        + [f"a listing for {c}" for c in OUT_OF_DOMAIN]
    )
    best = np.argmax(embeddings @ gate.T, axis=1)
    keep = best < len(IN_DOMAIN)

    # CLIP has one systematic blind spot: titles like "ASURION 2 Year
    # *Headphones* Protection Plan" carry a product noun that pulls the
    # classification into the electronics bucket. Measured leakage was 25 items
    # (0.31%), all of this shape. A keyword rule catches exactly that class —
    # CLIP for semantic recall, a regex for the one case it reliably misses.
    import re
    NON_PRODUCT_RE = re.compile(r"protection plan|service contract|asurion", re.I)
    leaked = np.array([bool(NON_PRODUCT_RE.search(str(t))) for t in titles])
    before = keep.sum()
    keep = keep & ~leaked
    print(f"  關鍵字後篩：再剔除 {before - keep.sum():,} 筆保固方案／服務合約")

    dropped = (~keep).sum()
    print(f"  電子商品篩選：保留 {keep.sum():,}，剔除 {dropped:,} "
          f"（{keep.mean():.1%} 保留率）")
    from collections import Counter
    for lab, n in Counter(labels[b] for b in best[~keep]).most_common(4):
        print(f"    剔除 {n:5,}  {lab}")

    products = [
        {
            "product_id": f"prod_{i}",
            "title": row["title"],
            "price": row.get("current/discounted_price"),
            "rating": row.get("rating"),
            "image_url": row["image_url"],
            "url": row["product_url"],
        }
        for i, (_, row) in enumerate(df[keep].iterrows())
    ]

    index = ProductIndex.build(products, embeddings[keep])
    index.save(config.PRODUCT_INDEX_PATH, config.PRODUCT_META_PATH)

    print(f"完成：{config.PRODUCT_INDEX_PATH}  ({len(index):,} 項)")
    print(f"      {config.PRODUCT_META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
