#!/usr/bin/env python3
"""Summarise the eval traces — the source for several README claims.

    python scripts/trace_stats.py

Reads every JSONL file in ``evals/traces/`` and reports the numbers the README
cites: how many tool calls the agent issues per step (the basis for the
"dependency chain, not a fan-out" argument against multi-agent), which tool
sequences occur, how often self-correcting retrieval fired, and the prompt-cache
read figures per mode.

Same spirit as scripts/calibrate_thresholds.py — every measured number in the
README should have a script that regenerates it.
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parent.parent / "evals" / "traces"


def main() -> int:
    files = sorted(TRACE_DIR.glob("*.jsonl"))
    if not files:
        print(f"找不到追蹤記錄：{TRACE_DIR}", file=sys.stderr)
        print("請先執行：python evals/run_evals.py", file=sys.stderr)
        return 1

    per_step = collections.Counter()
    sequences = collections.Counter()
    reformulations = 0
    cache = {"rag": [], "cag": []}
    steps_per_run = {"rag": [], "cag": []}

    for path in files:
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        mode = "cag" if "-cag" in path.name else "rag"

        by_step = collections.defaultdict(list)
        for r in records:
            if r["event"] == "tool_call":
                by_step[(r["turn"], r["step"])].append(r["tool"])
            elif r["event"] == "retrieval" and r.get("reformulation_of"):
                reformulations += 1
            elif r["event"] == "model_response":
                read = (r.get("usage") or {}).get("cache_read_input_tokens")
                if read:
                    cache[mode].append(read)

        for tools in by_step.values():
            per_step[len(tools)] += 1

        tools = [r["tool"] for r in records if r["event"] == "tool_call"]
        if len(tools) > 1:
            sequences[" → ".join(tools)] += 1
        steps_per_run[mode].append(max((r.get("step", 0) for r in records if "step" in r), default=0))

    total = sum(per_step.values())
    print(f"追蹤檔案 {len(files)} 個\n")

    print("每一步發出的工具呼叫數（多代理人論證的依據）")
    for n, c in sorted(per_step.items()):
        print(f"  {n} 個工具／步：{c:3d} 步   {c/total:5.1%}")
    print(f"  合計 {total} 步有工具呼叫\n")

    print("多工具序列")
    for seq, c in sequences.most_common(5):
        print(f"  {c:2d}x  {seq}")
    print()

    print(f"自我修正檢索（FR3）觸發次數：{reformulations}\n")

    print("每輪步數中位數 / 快取讀取中位數")
    for mode in ("rag", "cag"):
        s, c = steps_per_run[mode], cache[mode]
        if not s:
            continue
        med_cache = f"{int(st.median(c)):,}" if c else "—"
        print(f"  {mode.upper()}: {st.median(s):.0f} 步   cache_read {med_cache} tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
