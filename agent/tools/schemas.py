"""Claude native ``tool_use`` schemas for the customer-service agent.

Every schema here is declared ``strict: True`` with ``additionalProperties: False``
and a fully-populated ``required`` list, so Claude's tool inputs are guaranteed to
validate before they ever reach an executor (FR9).

Optional arguments are expressed as nullable types that stay in ``required``
rather than being omitted from it — strict mode wants every declared property
accounted for, and an explicit ``null`` is a stronger signal than an absent key.

Descriptions are Traditional Chinese on purpose. The tool surface is part of the
prompt; an English-only surface pulls the model's replies toward English
mid-conversation, which fights FR22.

The dependency chain between the three order tools is enforced by *data*, not by
prose in the system prompt:

    get_order_status ──delivered_at──▶ check_return_eligibility
                                              │
                                       eligibility_token
                                              │
                                              ▼
                                      calculate_refund

Each downstream tool re-validates the evidence it was handed against the fixture
store, so a fabricated date or an invented token is a hard error rather than a
plausible-looking answer (FR8, FR19).
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Retrieval (Mode A only — the CAG provider withholds this tool entirely)
# --------------------------------------------------------------------------

RETRIEVE_KB: dict[str, Any] = {
    "name": "retrieve_kb",
    "description": (
        "檢索本商店的客服知識庫，內容涵蓋退貨政策、退款流程、運費與配送時效、"
        "保固範圍、換貨、付款方式、發票、會員等級、缺貨與預購等規定。\n\n"
        "使用時機：當顧客詢問任何「本商店的規定或政策」時呼叫。這不是每一輪都必須執行的"
        "前置步驟——若問題純粹是查訂單、或明顯超出本商店服務範圍，就不需要檢索。\n\n"
        "回傳值中的 sufficiency 欄位會告訴你這次檢索是否足以回答："
        "verdict 為 sufficient 時可直接根據 results 作答；"
        "為 partial 或 insufficient 時，你必須改寫查詢重新檢索一次（並填入 reformulation_of），"
        "或明確告知顧客查無相關規定並提供轉真人客服的選項。"
        "任何情況下都不得以你自身的背景知識補足缺漏的政策細節。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "檢索用的查詢字串，建議使用顧客問題中的關鍵詞加上政策領域詞，"
                    "例如「七天鑑賞期 計算方式」或「大型家電 退貨 運費負擔」。"
                    "請使用繁體中文。"
                ),
            },
            "reformulation_of": {
                "type": ["string", "null"],
                "description": (
                    "若這次呼叫是因為前一次檢索的 sufficiency 不足而重新表述，"
                    "請填入前一次使用的 query 原文；這是第一次檢索則填 null。"
                    "系統會依此記錄自我修正的檢索軌跡。"
                ),
            },
        },
        "required": ["query", "reformulation_of"],
        "additionalProperties": False,
    },
}

# --------------------------------------------------------------------------
# Business tools — available in both modes
# --------------------------------------------------------------------------

GET_ORDER_STATUS: dict[str, Any] = {
    "name": "get_order_status",
    "description": (
        "查詢單一訂單的目前狀態、下單／出貨／送達日期、商品明細與金額。\n\n"
        "這是所有訂單相關流程的第一步。退貨資格判斷與退款試算都需要本工具回傳的"
        "delivered_at，因此在呼叫 check_return_eligibility 之前必須先呼叫本工具。\n\n"
        "回傳值中的 days_since_delivery 已由系統依當前日期計算完成，"
        "請直接使用，不要自行推算日期差。"
        "查無訂單時會回傳錯誤，此時應告知顧客查無此訂單編號並請其確認，不可自行編造訂單內容。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": (
                    "訂單編號，格式為 ORD- 加四位數字（例如 ORD-1001）。"
                    "必須是顧客在對話中明確提供過的編號，不可自行推測或補齊。"
                ),
            },
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}

CHECK_RETURN_ELIGIBILITY: dict[str, Any] = {
    "name": "check_return_eligibility",
    "description": (
        "依據送達日期與商品類別，判斷訂單是否仍在可退貨範圍內，"
        "並在符合資格時核發一組 eligibility_token 供後續退款試算使用。\n\n"
        "前置條件：必須先呼叫 get_order_status 取得該訂單的 delivered_at，"
        "並將該值原樣帶入本工具。系統會比對你提供的日期與訂單實際資料，"
        "不一致時會回傳錯誤——請勿自行推測或填寫未經查詢的日期。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "訂單編號，需與 get_order_status 查詢的訂單一致。",
            },
            "delivered_at": {
                "type": ["string", "null"],
                "description": (
                    "訂單送達日期，格式 YYYY-MM-DD。"
                    "此值必須取自 get_order_status 回傳的 delivered_at 欄位，原樣帶入。"
                    "若該訂單尚未送達（delivered_at 為 null）則填 null，"
                    "系統會據此回覆尚不適用退貨流程。"
                ),
            },
        },
        "required": ["order_id", "delivered_at"],
        "additionalProperties": False,
    },
}

CALCULATE_REFUND: dict[str, Any] = {
    "name": "calculate_refund",
    "description": (
        "試算退款金額，包含商品小計、運費是否退還、依退貨原因適用的手續費，"
        "以及預計退款到帳時間。\n\n"
        "前置條件：必須先呼叫 check_return_eligibility 並取得 eligibility_token。"
        "沒有有效憑證時本工具會拒絕執行——這是為了確保退款金額永遠建立在"
        "已驗證的退貨資格之上。若顧客不具退貨資格，請直接說明原因，不要嘗試呼叫本工具。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "訂單編號，需與前兩步查詢的訂單一致。",
            },
            "reason": {
                "type": "string",
                "enum": [
                    "商品瑕疵",
                    "與描述不符",
                    "運送破損",
                    "尺寸或規格不符",
                    "不再需要",
                    "其他",
                ],
                "description": (
                    "退貨原因。此欄位會影響運費是否退還與是否收取整備費："
                    "可歸責於商店的原因（商品瑕疵、與描述不符、運送破損）全額退還且免手續費；"
                    "顧客個人因素（不再需要、尺寸或規格不符）則依政策扣除。"
                    "請依顧客實際陳述選擇，不要為了對顧客有利而擅自歸類。"
                ),
            },
            "eligibility_token": {
                "type": "string",
                "description": (
                    "由 check_return_eligibility 回傳的資格憑證，必須原樣帶入。"
                    "此憑證與特定訂單綁定且無法自行構造。"
                ),
            },
        },
        "required": ["order_id", "reason", "eligibility_token"],
        "additionalProperties": False,
    },
}

ESCALATE_TO_HUMAN: dict[str, Any] = {
    "name": "escalate_to_human",
    "description": (
        "建立真人客服轉接單。\n\n"
        "使用時機：知識庫查無相關規定且改寫查詢後仍然不足、訂單資料出現異常、"
        "顧客明確要求真人客服，或問題超出自助服務可處理的範圍。\n\n"
        "這是「不知道就說不知道」的正當出口——當你無法從檢索結果或工具回傳值中"
        "找到答案時，轉接真人永遠優於猜測。呼叫後請向顧客說明已建立轉接單與預計回覆時間。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": [
                    "知識庫查無相關政策",
                    "訂單資料異常",
                    "客訴或情緒升溫",
                    "超出自助服務範圍",
                    "顧客明確要求真人客服",
                ],
                "description": "轉接原因分類，用於分流至對應的客服佇列。",
            },
            "summary": {
                "type": "string",
                "description": (
                    "給真人客服的交接摘要，繁體中文，兩到四句。"
                    "需包含：顧客的問題、已查證到的事實（訂單編號、狀態、已檢索到的政策），"
                    "以及卡在哪一步。請僅陳述工具回傳過的事實，不要加入推測。"
                ),
            },
        },
        "required": ["reason", "summary"],
        "additionalProperties": False,
    },
}

SEARCH_PRODUCTS_BY_IMAGE: dict[str, Any] = {
    "name": "search_products_by_image",
    "description": (
        "以顧客上傳的商品照片搜尋本商店目錄中外觀相似的商品。\n\n"
        "使用時機：顧客附上了照片並想找同款或類似的商品時。"
        "對話中若出現 image_ref（例如 image_ref=img_1），表示顧客有附圖可供搜尋。\n\n"
        "系統會先做影像領域判斷：若照片不是本商店販售的電子商品類別"
        "（例如水果、寵物、家具、人物），會直接回傳 out_of_domain 並附上判定類別，"
        "此時請告訴顧客這看起來不是本商店販售的商品，不要硬是推薦目錄中的東西。\n\n"
        "回傳的是「外觀相似的候選清單」而非確定答案。"
        "請把候選商品呈現給顧客並請其確認，不要斷言照片中的就是某一項商品。"
        "候選清單以外的商品一律不得提及。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "image_ref": {
                "type": "string",
                "description": (
                    "顧客所附照片的識別碼，格式為 img_N，必須取自對話中出現過的 "
                    "image_ref，不可自行編造。"
                ),
            },
            "note": {
                "type": ["string", "null"],
                "description": (
                    "顧客對這張照片的補充說明（例如「想找便宜一點的」）。"
                    "目前僅記錄於追蹤日誌，不影響檢索結果；沒有補充說明則填 null。"
                ),
            },
        },
        "required": ["image_ref", "note"],
        "additionalProperties": False,
    },
}

# --------------------------------------------------------------------------
# Groupings consumed by the context providers (agent/context.py)
# --------------------------------------------------------------------------

#: Tools available regardless of retrieval mode.
BUSINESS_TOOLS: list[dict[str, Any]] = [
    GET_ORDER_STATUS,
    CHECK_RETURN_ELIGIBILITY,
    CALCULATE_REFUND,
    ESCALATE_TO_HUMAN,
]

#: Mode A adds vector retrieval; Mode B (CAG) carries the KB in the cached
#: system prefix instead and withholds this tool.
RETRIEVAL_TOOLS: list[dict[str, Any]] = [RETRIEVE_KB]

#: Multimodal product search (FR23). Offered only when the product index has
#: been built — an advertised tool that always errors is worse than no tool.
MULTIMODAL_TOOLS: list[dict[str, Any]] = [SEARCH_PRODUCTS_BY_IMAGE]

ALL_SCHEMAS: list[dict[str, Any]] = RETRIEVAL_TOOLS + BUSINESS_TOOLS + MULTIMODAL_TOOLS

#: Human-readable zh-TW labels for the streaming status line (FR18).
TOOL_LABELS: dict[str, str] = {
    "retrieve_kb": "正在查詢客服規定…",
    "get_order_status": "正在查詢您的訂單…",
    "check_return_eligibility": "正在確認退貨資格…",
    "calculate_refund": "正在試算退款金額…",
    "escalate_to_human": "正在為您轉接真人客服…",
    "search_products_by_image": "正在以圖片搜尋相似商品…",
}
