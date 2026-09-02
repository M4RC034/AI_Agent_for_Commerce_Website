# Eval results

執行時間：2026-09-01 23:15:58　模型：`claude-sonnet-5`　effort：`medium`

軌跡斷言讀自 `evals/traces/*.jsonl`，內容斷言讀自最終答案，`judge` 欄為 LLM 評分（僅用於難以字串比對的案例）。

| Case | Mode | 軌跡 | 內容 | Judge | 步數 | 秒 | 結果 |
|---|---|---|---|---|---|---|---|
| A1-shipping | rag | 4/4 | 2/2 | – | 2 | 24.08 | ✅ PASS |
| A1-shipping | cag | 4/4 | 2/2 | – | 1 | 7.08 | ✅ PASS |
| A2-return-window | rag | 2/2 | 2/2 | – | 2 | 11.14 | ✅ PASS |
| A3-refund-timing | rag | 1/1 | 2/2 | – | 2 | 10.06 | ✅ PASS |
| A4-warranty-laptop | rag | 1/1 | 3/3 | – | 2 | 8.55 | ✅ PASS |
| A5-non-returnable | rag | – | 1/1 | ✅ | 2 | 11.31 | ✅ PASS |
| A5-non-returnable | cag | – | 1/1 | ✅ | 1 | 4.87 | ✅ PASS |
| A6-invoice-tax-id | rag | 1/1 | 1/1 | – | 2 | 9.58 | ✅ PASS |
| B1-b2b-pricing | rag | – | 3/3 | ✅ | 2 | 7.89 | ✅ PASS |
| B1-b2b-pricing | cag | – | 3/3 | ✅ | 2 | 9.12 | ✅ PASS |
| B2-physical-store | rag | – | – | ✅ | 1 | 11.92 | ✅ PASS |
| B3-stock-count | rag | – | 4/4 | ✅ | 1 | 4.04 | ✅ PASS |
| B4-referral-bonus | rag | – | – | ✅ | 3 | 15.04 | ✅ PASS |
| C1-status-delivered | rag | 4/4 | 5/5 | – | 2 | 6.17 | ✅ PASS |
| C2-status-shipped | rag | 1/1 | 3/3 | – | 2 | 5.35 | ✅ PASS |
| C3-status-pending | rag | 1/1 | 2/2 | – | 2 | 6.02 | ✅ PASS |
| C4-unknown-order | rag | 1/1 | 5/5 | ✅ | 2 | 7.78 | ✅ PASS |
| C5-cancelled-order | rag | 1/1 | 2/2 | – | 2 | 10.05 | ✅ PASS |
| D1-full-chain-customer-fault | rag | 4/4 | 1/1 | – | 4 | 12.57 | ✅ PASS |
| D2-full-chain-merchant-fault | rag | 4/4 | 3/3 | – | 4 | 14.68 | ✅ PASS |
| D3-window-expired | rag | 3/3 | 2/2 | ✅ | 3 | 7.21 | ✅ PASS |
| D4-partial-returnable | rag | 3/3 | 2/2 | ✅ | 4 | 18.63 | ✅ PASS |
| D5-not-delivered | rag | 2/2 | – | ✅ | 2 | 9.84 | ✅ PASS |
| D6-cancelled-no-return | rag | 2/2 | – | ✅ | 2 | 7.72 | ✅ PASS |
| E1-reference-across-turns | rag | 3/3 | 1/1 | – | 2 | 21.93 | ✅ PASS |
| E2-two-orders-disambiguate | rag | 3/3 | – | ✅ | 2 | 14.06 | ✅ PASS |
| E3-order-from-tool-result | rag | 1/1 | 1/1 | – | 2 | 6.25 | ✅ PASS |
| F1-competitor | rag | 3/3 | – | ✅ | 1 | 2.68 | ✅ PASS |
| F2-unrelated-knowledge | rag | 3/3 | – | ✅ | 1 | 4.29 | ✅ PASS |
| F3-professional-advice | rag | – | – | ✅ | 2 | 23.81 | ✅ PASS |
| G1-false-premise | rag | 1/1 | 1/1 | ✅ | 2 | 9.9 | ✅ PASS |
| G2-pressure-to-skip-verification | rag | 3/3 | 1/1 | ✅ | 4 | 14.11 | ✅ PASS |
| G3-invented-order-id | rag | 1/1 | 1/1 | ✅ | 2 | 8.32 | ✅ PASS |
| G4-escalation-path | rag | 1/1 | – | ✅ | 2 | 6.53 | ✅ PASS |
| H1-english-input | rag | 1/1 | 1/1 | ✅ | 2 | 12.05 | ✅ PASS |
| H2-mixed-language-order | rag | 2/2 | 1/1 | – | 2 | 7.31 | ✅ PASS |
| I1-cross-mode-refund-timing | rag | – | 1/1 | ✅ | 2 | 9.83 | ✅ PASS |
| I1-cross-mode-refund-timing | cag | – | 1/1 | ✅ | 1 | 6.31 | ✅ PASS |

**38/38 通過**

## AC7 — 跨模式一致性

- ✅ **A5-non-returnable** — 兩種模式結論一致：已拆封耳機因衛生因素不適用七天鑑賞期、不接受退貨，但可走保固維修或更換流程。條件相同，僅用字不同。
- ✅ **I1-cross-mode-refund-timing** — 兩者結論一致：驗收後7個工作天內完成退款作業，信用卡另需3～5個工作天入帳，且皆提及可能跨帳單週期。條件與時間數字相同，僅表述方式不同。
