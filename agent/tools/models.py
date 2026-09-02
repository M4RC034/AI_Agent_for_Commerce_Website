"""Pydantic models for every tool return value.

FR9 asks for structured constraints on tool *outputs* as well as inputs. Each
executor in this package builds one of these models and serialises it with
``model_dump_json()``; nothing hand-rolls a dict on the way back to Claude. A
malformed return is then a ``ValidationError`` at the executor boundary rather
than a subtly wrong string the model has to interpret.

The enums matter more than they look. ``SufficiencyVerdict`` is the structured
signal FR4 asks for, and ``ReturnVerdict`` is what the system prompt keys its
"may I proceed to refund" rule off — both are read by the eval harness straight
out of the trace log, so they need to be closed sets rather than free text.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Shared enums
# --------------------------------------------------------------------------


class OrderState(str, Enum):
    """Lifecycle states a fixture order can be in."""

    PENDING = "待出貨"
    SHIPPED = "已出貨"
    DELIVERED = "已送達"
    CANCELLED = "已取消"
    RETURNING = "退貨處理中"


class SufficiencyVerdict(str, Enum):
    """FR4's structured judgment, computed from rerank scores.

    ``PARTIAL`` and ``INSUFFICIENT`` both oblige the agent to either reformulate
    (FR3) or admit ignorance / escalate (FR20). They are kept distinct so the
    trace can tell "found something adjacent" apart from "found nothing".
    """

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ReturnVerdict(str, Enum):
    ELIGIBLE = "eligible"
    WINDOW_EXPIRED = "window_expired"
    NOT_DELIVERED = "not_delivered"
    NON_RETURNABLE = "non_returnable"
    ORDER_CANCELLED = "order_cancelled"


# --------------------------------------------------------------------------
# retrieve_kb
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc: str = Field(description="來源文件檔名，供答覆時引用")
    title: str
    text: str
    rerank_score: float = Field(description="CrossEncoder 重排分數，越高越相關")


class Sufficiency(BaseModel):
    """The FR4 signal. Deterministic — derived from scores, not from the model."""

    verdict: SufficiencyVerdict
    top_score: float | None
    threshold: float
    reason: str = Field(description="繁體中文說明，會一併寫入 trace 供評測讀取")


class RetrievalResult(BaseModel):
    query: str
    reformulation_of: str | None
    attempt: int = Field(description="本輪對話中第幾次檢索，用於封頂自我修正次數")
    results: list[RetrievedChunk]
    sufficiency: Sufficiency


# --------------------------------------------------------------------------
# get_order_status
# --------------------------------------------------------------------------


class OrderItem(BaseModel):
    sku: str
    name: str
    category: str
    unit_price: int
    quantity: int
    returnable: bool = Field(description="部分類別（如已拆封耗材）依政策不可退")


class OrderStatus(BaseModel):
    order_id: str
    status: OrderState
    placed_at: str
    shipped_at: str | None
    delivered_at: str | None
    days_since_delivery: int | None = Field(
        description="系統依當前日期預先算好，避免把今日日期放進被快取的系統提示詞"
    )
    carrier: str | None
    tracking_no: str | None
    items: list[OrderItem]
    items_subtotal: int
    shipping_fee: int
    total: int
    currency: Literal["TWD"] = "TWD"


# --------------------------------------------------------------------------
# check_return_eligibility
# --------------------------------------------------------------------------


class ReturnEligibility(BaseModel):
    order_id: str
    eligible: bool
    verdict: ReturnVerdict
    window_days: int = Field(description="政策規定的鑑賞期天數")
    days_elapsed: int | None
    days_remaining: int | None
    non_returnable_skus: list[str] = Field(
        default_factory=list, description="訂單中依政策不可退的品項"
    )
    explanation: str
    eligibility_token: str | None = Field(
        description="僅在 eligible 為 true 時核發；calculate_refund 需要此憑證"
    )


# --------------------------------------------------------------------------
# calculate_refund
# --------------------------------------------------------------------------


class RefundLine(BaseModel):
    sku: str
    name: str
    quantity: int
    line_total: int


class RefundCalculation(BaseModel):
    order_id: str
    reason: str
    fault: Literal["merchant", "customer"] = Field(
        description="歸責方，決定運費與整備費是否退還"
    )
    refundable_items: list[RefundLine]
    items_subtotal: int
    shipping_refund: int
    restocking_fee: int = Field(description="整備費，可歸責於商店時為 0")
    net_refund: int
    currency: Literal["TWD"] = "TWD"
    refund_method: str
    eta_business_days: int


# --------------------------------------------------------------------------
# escalate_to_human
# --------------------------------------------------------------------------


class EscalationTicket(BaseModel):
    ticket_id: str = Field(description="由 session_id 與輪次雜湊而成，確保可重現（FR10）")
    reason: str
    queue: str
    summary: str
    expected_response: str


# --------------------------------------------------------------------------
# Uniform error envelope
# --------------------------------------------------------------------------


class ToolError(BaseModel):
    """Returned with ``is_error: True`` on the tool_result block.

    Errors are surfaced to the model rather than swallowed — an unknown order ID
    must read as a hard failure, never as an empty success, or the model fills
    the silence (FR19).
    """

    error: str
    detail: str
    remedy: str = Field(description="給模型的下一步指示，繁體中文")


# --------------------------------------------------------------------------
# search_products_by_image (FR23)
# --------------------------------------------------------------------------


class ProductMatch(BaseModel):
    product_id: str
    title: str
    price: str | None
    rating: str | None
    url: str
    similarity: float = Field(description="CLIP 相似度，僅供排序，不可作為信心度")


class ImageSearchResult(BaseModel):
    """Gate verdict first, candidates second.

    ``in_domain`` is the load-bearing field: CLIP similarity alone cannot tell
    an in-catalog product from a banana (measured ranges overlap), so the
    zero-shot gate is what makes this tool groundable.
    """

    image_ref: str
    in_domain: bool
    detected_category: str = Field(description="零樣本判定的類別，供拒絕時說明用")
    candidates: list[ProductMatch]
    note: str = Field(description="繁體中文說明，指示模型如何呈現結果")
