#!/usr/bin/env python3
"""Calibrate the FR4 sufficiency thresholds against real zh-TW queries.

Prints the rerank score of the top hit for three classes of probe: questions the
KB covers, the deliberate gaps recorded in data/kb_coverage.md, and out-of-scope
questions. The thresholds in agent/config.py should sit in the empty band
between the covered minimum and the gap maximum.

Re-run this after editing the knowledge base or swapping the reranker — a new
corpus shifts the bands, and a threshold tuned to the old one fails silently.

    python scripts/calibrate_thresholds.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
from agent import config
from agent.retrieval.embedder import Embedder
from agent.retrieval.index import KBIndex
from agent.retrieval.reranker import Reranker

COVERED = [
    "七天鑑賞期怎麼算？",
    "運費多少錢，滿多少免運？",
    "退款要多久才會到帳？",
    "耳機拆封了還能退嗎？",
    "筆電保固多久？",
    "可以改收件地址嗎？",
    "發票統編打錯了怎麼辦？",
    "會員等級怎麼升？",
    "可以寄到香港嗎？",
    "客服幾點上班？",
]
GAPS = [
    "企業採購大宗訂單有什麼折扣？",
    "你們的實體門市在哪裡？",
    "有沒有二手商品寄賣服務？",
    "員工優惠怎麼申請？",
    "這台筆電現在還剩幾台庫存？",
]
OFF_SCOPE = [
    "你們跟 momo 誰比較便宜？",
    "推薦台北好吃的餐廳",
    "幫我算一下這個微積分題目",
]

emb, idx, rr = Embedder(), KBIndex.load(config.KB_INDEX_PATH, config.KB_CHUNKS_PATH), Reranker()

def probe(label, queries):
    print(f"\n{'='*76}\n{label}\n{'='*76}")
    tops = []
    for q in queries:
        hits = idx.search(emb.encode_query(q), config.RETRIEVE_K)
        ranked = rr.rerank(q, hits, config.RERANK_TOP_N)
        top_chunk, top_score = ranked[0]
        tops.append(top_score)
        print(f"{top_score:8.4f}  {q}")
        print(f"          -> {top_chunk.title}｜{top_chunk.section}")
    print(f"\n  min={min(tops):.4f}  max={max(tops):.4f}  mean={sum(tops)/len(tops):.4f}")
    return tops

c = probe("COVERED — should be SUFFICIENT", COVERED)
g = probe("DELIBERATE GAPS (AC4) — should be INSUFFICIENT", GAPS)
o = probe("OUT OF SCOPE (AC6) — should be INSUFFICIENT", OFF_SCOPE)

print(f"\n{'='*76}\nSEPARATION\n{'='*76}")
print(f"  covered  min = {min(c):.4f}")
print(f"  gap      max = {max(g):.4f}")
print(f"  offscope max = {max(o):.4f}")
print(f"  margin (covered.min - max(gap,offscope)) = {min(c) - max(max(g), max(o)):+.4f}")
print(f"\n  current SUFFICIENT_THRESHOLD = {config.SUFFICIENT_THRESHOLD}")
print(f"  current PARTIAL_THRESHOLD    = {config.PARTIAL_THRESHOLD}")
