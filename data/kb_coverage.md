# Knowledge base coverage and deliberate gaps

This file is **not indexed** — it lives outside `data/kb/` on purpose. It records
what the knowledge base covers and, more importantly, what it deliberately does
not, so AC4 tests a real gap rather than an accident of chunking.

## Covered

| Document | Topic |
|---|---|
| `returns-policy.md` | 七天鑑賞期、退貨條件、不適用鑑賞期的品項、部分退貨 |
| `refund-process.md` | 退款四階段、各付款方式到帳時間、退款金額計算、退款失敗 |
| `shipping-fees.md` | 標準運費、免運門檻、外島加價、大型商品、退貨運費歸屬 |
| `delivery-time.md` | 出貨與配送時效、查件、配送失敗、指定時段 |
| `warranty.md` | 各品類保固期、保固範圍內外、送修流程、與鑑賞期的差別 |
| `exchange.md` | 可換貨情形、流程、時間、費用、換貨後的保固起算 |
| `payment-methods.md` | 支援的付款方式、分期、ATM 期限、貨到付款、付款失敗 |
| `invoice.md` | 電子發票、紙本補印、統編、載具捐贈、退貨時的發票處理 |
| `membership.md` | 四級會員、各級權益、點數取得與使用、效期、退貨影響 |
| `stock-preorder.md` | 缺貨處理、預購、預購取消、混單出貨、補貨時間 |
| `order-changes.md` | 可修改項目、取消訂單、自動取消、出貨後改址、重複下單 |
| `service-hours.md` | 服務時間、聯絡方式、回覆時間、何時該轉真人 |
| `overseas-shipping.md` | 不提供國際配送、轉運注意事項、海外卡、海外會員 |
| `privacy-account.md` | 蒐集範圍、使用範圍、查詢刪除、取消訂閱、帳號安全、身分核對 |

## Deliberate gaps — the KB cannot answer these

These are the AC4 probes. The agent must say it does not know and offer
escalation, **not** answer from background knowledge:

1. **企業採購與大宗訂單合約條件** — no B2B pricing, no volume discount tiers, no
   contract terms anywhere in the corpus.
2. **實體門市與維修中心地址** — the shop is presented as online-only; no store
   locations, no walk-in repair counters.
3. **二手商品與寄賣服務** — trade-in, refurbished resale, and consignment are
   never mentioned.
4. **員工優惠與推薦獎金制度** — no staff discount policy, no referral program.
5. **各商品的具體庫存數量** — stock is described as a state (現貨 / 補貨中 / 預購)
   but never as a number.

## Out-of-scope probes — for AC6

These are not KB gaps; they are questions the agent should decline politely
because they fall outside a shop assistant's remit:

- Competitor comparisons ("你們跟 momo 誰比較便宜？")
- General knowledge unrelated to the shop ("推薦台北好吃的餐廳")
- Requests for advice the shop is not qualified to give (legal, medical, financial)

## Consistency requirement

The numbers in `returns-policy.md`, `refund-process.md`, and `shipping-fees.md`
must agree with the constants in `agent/config.py` — 7-day window, 5% restocking
fee, NT$1,000 free-shipping threshold, NT$80 standard shipping, 7 business days
to refund. The eval harness cross-checks the figures the agent states against
what the tools compute, so a drift between the KB prose and the config would
show up as a failing case rather than a silent inconsistency.
