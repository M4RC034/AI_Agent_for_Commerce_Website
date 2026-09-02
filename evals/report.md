# Eval results

執行時間：2026-09-01 21:38:42　模型：`claude-sonnet-5`　effort：`medium`

軌跡斷言讀自 `evals/traces/*.jsonl`，內容斷言讀自最終答案，`judge` 欄為 LLM 評分（僅用於難以字串比對的案例）。

| Case | Mode | 軌跡 | 內容 | Judge | 步數 | 秒 | 結果 |
|---|---|---|---|---|---|---|---|
| B1-b2b-pricing | rag | – | 3/3 | – | 2 | 7.36 | ✅ PASS |
| B1-b2b-pricing | cag | – | 3/3 | – | 2 | 8.29 | ✅ PASS |

**2/2 通過**
