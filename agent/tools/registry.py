"""Tool executors and dispatch (FR7–FR10).

Two ideas run through this module.

**Fixed fixtures, real dates.** FR10 wants reproducibility, but return-window
arithmetic needs a clock, and absolute fixture dates would drift out of the
window within a week and stop exercising the eligibility logic. The fixtures
therefore store day *offsets*, resolved against ``config.today()`` — which is
pinnable via ``AGENT_TODAY`` so eval runs are exactly reproducible.

**The chain is enforced by data, not by prose.** ``check_return_eligibility``
re-validates the ``delivered_at`` it was handed against the fixture, and
``calculate_refund`` verifies an HMAC token that only the eligibility check can
mint. A model that invents a delivery date or fabricates a token gets a hard
error rather than a plausible-looking answer, which is what turns FR19's
grounding rule from a request into a constraint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from agent import config
from agent.tools import models as m
from agent.trace import Tracer

#: Salt for the eligibility HMAC. This is a data-integrity check between two of
#: our own tools, not a security boundary — its job is to make the token
#: unguessable by a model that is pattern-matching, nothing more.
_TOKEN_SALT = b"maiagent-task1-eligibility-v1"


def _ok(payload: Any, receipt: str) -> tuple[dict[str, Any], str]:
    content = payload.model_dump_json() if hasattr(payload, "model_dump_json") else payload
    return {"type": "tool_result", "content": content}, receipt


def _err(error: str, detail: str, remedy: str, receipt: str) -> tuple[dict[str, Any], str]:
    payload = m.ToolError(error=error, detail=detail, remedy=remedy)
    return (
        {"type": "tool_result", "content": payload.model_dump_json(), "is_error": True},
        receipt,
    )


# ==========================================================================
# Fixture store
# ==========================================================================


class OrderStore:
    """Loads ``data/fixtures/orders.json`` and resolves offsets to real dates."""

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._orders: dict[str, dict] = raw["orders"]

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    @property
    def ids(self) -> list[str]:
        return sorted(self._orders)

    @staticmethod
    def _resolve(days_ago: int | None, today: date) -> str | None:
        return None if days_ago is None else (today - timedelta(days=days_ago)).isoformat()

    def get(self, order_id: str) -> m.OrderStatus | None:
        record = self._orders.get(order_id)
        if record is None:
            return None

        today = config.today()
        items = [m.OrderItem(**item) for item in record["items"]]
        items_subtotal = sum(i.unit_price * i.quantity for i in items)
        delivered_days_ago = record["delivered_days_ago"]

        return m.OrderStatus(
            order_id=order_id,
            status=m.OrderState(record["status"]),
            placed_at=self._resolve(record["placed_days_ago"], today),
            shipped_at=self._resolve(record["shipped_days_ago"], today),
            delivered_at=self._resolve(delivered_days_ago, today),
            # Precomputed so today's date never has to enter the prompt.
            days_since_delivery=delivered_days_ago,
            carrier=record["carrier"],
            tracking_no=record["tracking_no"],
            items=items,
            items_subtotal=items_subtotal,
            shipping_fee=record["shipping_fee"],
            total=items_subtotal + record["shipping_fee"],
        )


# ==========================================================================
# Eligibility token
# ==========================================================================


def _mint_token(order_id: str, delivered_at: str) -> str:
    digest = hmac.new(
        _TOKEN_SALT, f"{order_id}|{delivered_at}".encode(), hashlib.sha256
    ).hexdigest()
    return f"ELG-{digest[:16]}"


def _verify_token(token: str, order_id: str, delivered_at: str) -> bool:
    return hmac.compare_digest(token or "", _mint_token(order_id, delivered_at))


# ==========================================================================
# Registry
# ==========================================================================


class ToolRegistry:
    """Owns tool state: the fixture store, the retrieval stack, per-turn counters."""

    def __init__(
        self,
        mode: config.Mode,
        tracer: Tracer,
        image_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.mode = mode
        self.tracer = tracer
        self.orders = OrderStore(config.FIXTURES_PATH)

        # Resolves an image_ref like "img_1" to a PIL Image. Supplied by the
        # caller (server session or CLI), so the tool layer never touches
        # transport concerns.
        self.image_resolver = image_resolver
        self._clip: _ClipStack | None = None

        # Retrieval is built only in Mode A. Mode B never loads the embedding
        # model or the reranker at all — that asymmetry is most of its latency
        # advantage, and it would be invisible if the stack loaded eagerly.
        self._retrieval: _RetrievalStack | None = None
        if mode is config.Mode.RAG:
            self._retrieval = _get_retrieval_stack()

        #: FR3 guard, keyed by turn so each user turn gets a fresh budget.
        self._attempts: dict[int, int] = {}

    @property
    def kb_size(self) -> int | None:
        return len(self._retrieval.index) if self._retrieval else None

    @property
    def clip(self) -> "_ClipStack | None":
        """Lazily build the CLIP stack; None when the index has not been built."""
        if self._clip is None:
            try:
                self._clip = _ClipStack()
            except FileNotFoundError:
                return None
        return self._clip

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], str]:
        handler: Callable[..., tuple[dict[str, Any], str]] | None = {
            "retrieve_kb": self.retrieve_kb,
            "get_order_status": self.get_order_status,
            "check_return_eligibility": self.check_return_eligibility,
            "calculate_refund": self.calculate_refund,
            "escalate_to_human": self.escalate_to_human,
            "search_products_by_image": self.search_products_by_image,
        }.get(name)

        if handler is None:
            return _err(
                "unknown_tool",
                f"沒有名為 {name} 的工具。",
                "請改用可用的工具，或直接回覆顧客。",
                f"未知的工具：{name}",
            )

        try:
            return handler(**tool_input)
        except TypeError as exc:
            # Strict schemas make this near-impossible, but a tool that raises
            # must still produce a tool_result — a dropped block is a 400.
            return _err(
                "bad_arguments",
                f"{name} 的參數不正確：{exc}",
                "請檢查參數後重試。",
                f"{name} 參數錯誤",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return _err(
                "tool_failure",
                f"{name} 執行失敗：{exc}",
                "請告知顧客系統暫時無法處理，並建議轉接真人客服。",
                f"{name} 執行失敗",
            )

    # -- FR2 / FR3 / FR4 / FR6 --------------------------------------------

    def retrieve_kb(self, query: str, reformulation_of: str | None = None):
        if self._retrieval is None:
            return _err(
                "retrieval_unavailable",
                "目前為 CAG 模式，知識庫已直接附在系統提示詞中。",
                "請直接閱讀系統提示詞中的知識庫全文作答。",
                "CAG 模式不提供檢索工具",
            )

        turn = self.tracer.turn
        attempt = self._attempts.get(turn, 0) + 1
        self._attempts[turn] = attempt

        if attempt > config.MAX_RETRIEVAL_ATTEMPTS:
            self.tracer.guard_tripped(
                "max_retrieval_attempts", turn=turn, attempts=attempt
            )
            return _err(
                "retrieval_budget_exhausted",
                f"本輪已檢索 {config.MAX_RETRIEVAL_ATTEMPTS} 次，仍未找到足夠的資料。",
                "請停止重新檢索，明確告知顧客查無相關規定，並呼叫 escalate_to_human。",
                f"檢索已達 {config.MAX_RETRIEVAL_ATTEMPTS} 次上限",
            )

        ranked = self._retrieval.search(query)
        top_score = ranked[0][1] if ranked else None
        verdict, reason = _judge_sufficiency(top_score)

        result = m.RetrievalResult(
            query=query,
            reformulation_of=reformulation_of,
            attempt=attempt,
            results=[
                m.RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc=c.doc,
                    title=c.title,
                    text=f"{c.section}\n{c.text}",
                    rerank_score=round(score, 4),
                )
                for c, score in ranked
            ],
            sufficiency=m.Sufficiency(
                verdict=verdict,
                top_score=round(top_score, 4) if top_score is not None else None,
                threshold=config.SUFFICIENT_THRESHOLD,
                reason=reason,
            ),
        )

        self.tracer.retrieval(
            query=query,
            reformulation_of=reformulation_of,
            attempt=attempt,
            verdict=verdict.value,
            top_score=top_score,
            chunk_ids=[c.chunk_id for c, _ in ranked],
        )

        tag = {"sufficient": "資料充分", "partial": "資料部分相關", "insufficient": "查無相關規定"}
        receipt = f"檢索「{query}」→ {len(ranked)} 段，{tag[verdict.value]}"
        return _ok(result, receipt)

    # -- FR7 ---------------------------------------------------------------

    def get_order_status(self, order_id: str):
        order = self.orders.get(order_id)
        if order is None:
            return _err(
                "order_not_found",
                f"查無訂單 {order_id}。",
                "請告知顧客查無此訂單編號並請其確認，不要推測是哪一筆訂單。",
                f"查無訂單 {order_id}",
            )

        when = f"，{order.days_since_delivery} 天前送達" if order.days_since_delivery is not None else ""
        return _ok(order, f"訂單 {order_id}：{order.status.value}{when}")

    def check_return_eligibility(self, order_id: str, delivered_at: str | None = None):
        order = self.orders.get(order_id)
        if order is None:
            return _err(
                "order_not_found",
                f"查無訂單 {order_id}。",
                "請先確認訂單編號是否正確。",
                f"查無訂單 {order_id}",
            )

        # FR8 + FR19: the evidence the model passed in must match the record.
        # This is the hallucination trap — an invented date fails loudly here.
        if delivered_at != order.delivered_at:
            return _err(
                "delivered_at_mismatch",
                f"你提供的送達日期（{delivered_at}）與訂單實際資料（{order.delivered_at}）不符。",
                "請先呼叫 get_order_status，並將其回傳的 delivered_at 原樣帶入，不要自行推測日期。",
                "送達日期與訂單資料不符",
            )

        verdict, eligible, explanation = _judge_return(order)
        non_returnable = [i.sku for i in order.items if not i.returnable]

        result = m.ReturnEligibility(
            order_id=order_id,
            eligible=eligible,
            verdict=verdict,
            window_days=config.RETURN_WINDOW_DAYS,
            days_elapsed=order.days_since_delivery,
            days_remaining=(
                config.RETURN_WINDOW_DAYS - order.days_since_delivery
                if eligible and order.days_since_delivery is not None
                else None
            ),
            non_returnable_skus=non_returnable,
            explanation=explanation,
            eligibility_token=(
                _mint_token(order_id, order.delivered_at) if eligible else None
            ),
        )
        return _ok(result, f"訂單 {order_id} 退貨資格：{'符合' if eligible else '不符合'}")

    def calculate_refund(self, order_id: str, reason: str, eligibility_token: str):
        order = self.orders.get(order_id)
        if order is None:
            return _err(
                "order_not_found", f"查無訂單 {order_id}。", "請先確認訂單編號。", f"查無訂單 {order_id}"
            )

        # FR8: the chain cannot be short-circuited. Without a token minted by
        # check_return_eligibility for *this* order, there is no refund to quote.
        if not _verify_token(eligibility_token, order_id, order.delivered_at or ""):
            return _err(
                "invalid_eligibility_token",
                "eligibility_token 無效或與此訂單不符。",
                "請先呼叫 check_return_eligibility 取得憑證；若顧客不具退貨資格，請直接說明原因，不要試算退款。",
                "退貨資格憑證無效",
            )

        refundable = [i for i in order.items if i.returnable]
        lines = [
            m.RefundLine(
                sku=i.sku, name=i.name, quantity=i.quantity,
                line_total=i.unit_price * i.quantity,
            )
            for i in refundable
        ]
        items_subtotal = sum(line.line_total for line in lines)

        merchant_fault = reason in config.MERCHANT_FAULT_REASONS
        shipping_refund = order.shipping_fee if merchant_fault else 0
        restocking_fee = 0 if merchant_fault else round(items_subtotal * config.RESTOCKING_FEE_RATE)

        result = m.RefundCalculation(
            order_id=order_id,
            reason=reason,
            fault="merchant" if merchant_fault else "customer",
            refundable_items=lines,
            items_subtotal=items_subtotal,
            shipping_refund=shipping_refund,
            restocking_fee=restocking_fee,
            net_refund=items_subtotal + shipping_refund - restocking_fee,
            refund_method="退回原付款方式",
            eta_business_days=config.REFUND_ETA_BUSINESS_DAYS,
        )
        return _ok(result, f"訂單 {order_id} 預估退款 NT${result.net_refund:,}")

    def escalate_to_human(self, reason: str, summary: str):
        # Deterministic ticket ID (FR10) — no randomness, so evals are stable.
        seed = f"{self.tracer.session_id}|{self.tracer.turn}|{reason}".encode()
        ticket_id = f"CS-{hashlib.sha1(seed).hexdigest()[:6].upper()}"

        queue = {
            "知識庫查無相關政策": "政策諮詢組",
            "訂單資料異常": "訂單處理組",
            "客訴或情緒升溫": "客訴處理組",
            "超出自助服務範圍": "一般客服組",
            "顧客明確要求真人客服": "一般客服組",
        }.get(reason, "一般客服組")

        result = m.EscalationTicket(
            ticket_id=ticket_id,
            reason=reason,
            queue=queue,
            summary=summary,
            expected_response="服務時間內一個工作天回覆（週一至週五 09:00–18:00）",
        )
        return _ok(result, f"已建立轉接單 {ticket_id}（{queue}）")

    # -- FR23 --------------------------------------------------------------

    def search_products_by_image(self, image_ref: str, note: str | None = None):
        if self.clip is None:
            return _err(
                "product_index_unavailable",
                "商品圖片索引尚未建立，無法進行以圖搜圖。",
                "請告知顧客目前無法使用圖片搜尋，並改以文字描述協助，或轉接真人客服。",
                "商品索引未建立",
            )

        image = self.image_resolver(image_ref) if self.image_resolver else None
        if image is None:
            return _err(
                "image_not_found",
                f"找不到 image_ref={image_ref} 對應的圖片。",
                "請確認 image_ref 取自對話中顧客實際上傳的圖片，不可自行編造。",
                f"查無圖片 {image_ref}",
            )

        vector = self.clip.encoder.encode_image(image)
        category, in_domain = self.clip.encoder.classify(vector)

        # The gate runs BEFORE retrieval, not after. Similarity alone cannot
        # separate an in-catalog product from a banana — the measured score
        # ranges overlap — so an ungated result would be a confident fabrication.
        if not in_domain:
            result = m.ImageSearchResult(
                image_ref=image_ref, in_domain=False, detected_category=category,
                candidates=[],
                note=(
                    f"這張照片被判定為「{category}」，不屬於本商店販售的電子商品類別。"
                    "請告訴顧客這看起來不是本商店販售的商品，不要從目錄中挑選任何商品推薦。"
                ),
            )
            self.tracer.emit("image_search", image_ref=image_ref, in_domain=False,
                             category=category, candidates=0)
            return _ok(result, f"圖片判定為「{category}」，非本商店商品")

        hits = self.clip.index.search(vector, config.PRODUCT_TOP_K)
        candidates = [
            m.ProductMatch(
                product_id=p["product_id"], title=p["title"],
                price=None if p.get("price") is None else str(p["price"]),
                rating=None if p.get("rating") is None else str(p["rating"]),
                url=p["url"], similarity=round(score, 4),
            )
            for p, score in hits
        ]
        result = m.ImageSearchResult(
            image_ref=image_ref, in_domain=True, detected_category=category,
            candidates=candidates,
            note=(
                "以下是外觀相似的候選商品，並非確定的比對結果。"
                "請將候選清單呈現給顧客並請其確認是哪一項，不要斷言照片中就是某一項商品，"
                "也不得提及候選清單以外的商品。"
            ),
        )
        self.tracer.emit("image_search", image_ref=image_ref, in_domain=True,
                         category=category, candidates=len(candidates),
                         top=[c.product_id for c in candidates])
        return _ok(result, f"圖片判定為「{category}」，找到 {len(candidates)} 項相似商品")


# ==========================================================================
# Judgment helpers — deterministic, so the trace is scoreable
# ==========================================================================


def _judge_sufficiency(top_score: float | None) -> tuple[m.SufficiencyVerdict, str]:
    """FR4. Derived from the reranker, never from the model's self-assessment."""
    if top_score is None:
        return m.SufficiencyVerdict.INSUFFICIENT, "檢索沒有回傳任何段落。"
    if top_score >= config.SUFFICIENT_THRESHOLD:
        return (
            m.SufficiencyVerdict.SUFFICIENT,
            f"最高重排分數 {top_score:.3f} 高於門檻 {config.SUFFICIENT_THRESHOLD}，資料足以作答。",
        )
    if top_score >= config.PARTIAL_THRESHOLD:
        return (
            m.SufficiencyVerdict.PARTIAL,
            f"最高重排分數 {top_score:.3f} 低於門檻 {config.SUFFICIENT_THRESHOLD}，"
            "檢索到的段落只有部分相關。請改寫查詢重新檢索，或告知顧客查無明確規定。",
        )
    return (
        m.SufficiencyVerdict.INSUFFICIENT,
        f"最高重排分數 {top_score:.3f} 遠低於門檻 {config.SUFFICIENT_THRESHOLD}，"
        "知識庫沒有涵蓋這個主題。請勿以背景知識作答。",
    )


def _judge_return(order: m.OrderStatus) -> tuple[m.ReturnVerdict, bool, str]:
    if order.status is m.OrderState.CANCELLED:
        return m.ReturnVerdict.ORDER_CANCELLED, False, "此訂單已取消，不需辦理退貨。"

    if order.delivered_at is None or order.days_since_delivery is None:
        return (
            m.ReturnVerdict.NOT_DELIVERED,
            False,
            f"此訂單目前狀態為「{order.status.value}」，尚未送達，"
            "七天鑑賞期自送達次日起算，因此還無法申請退貨。",
        )

    if not any(i.returnable for i in order.items):
        return (
            m.ReturnVerdict.NON_RETURNABLE,
            False,
            "此訂單的商品全部屬於不適用七天鑑賞期的品項（例如已拆封的耳機類商品），恕無法退貨。",
        )

    if order.days_since_delivery > config.RETURN_WINDOW_DAYS:
        return (
            m.ReturnVerdict.WINDOW_EXPIRED,
            False,
            f"商品已於 {order.days_since_delivery} 天前送達，超過 "
            f"{config.RETURN_WINDOW_DAYS} 天鑑賞期，無法以個人因素退貨；"
            "若商品有瑕疵或故障，可改走保固維修流程。",
        )

    remaining = config.RETURN_WINDOW_DAYS - order.days_since_delivery
    partial = [i.sku for i in order.items if not i.returnable]
    note = f"（其中 {'、'.join(partial)} 不適用鑑賞期，不列入退貨）" if partial else ""
    return (
        m.ReturnVerdict.ELIGIBLE,
        True,
        f"商品於 {order.days_since_delivery} 天前送達，仍在 {config.RETURN_WINDOW_DAYS} "
        f"天鑑賞期內，尚餘 {remaining} 天可申請退貨{note}。",
    )


# ==========================================================================
# Retrieval stack — lazily constructed, Mode A only
# ==========================================================================


class _RetrievalStack:
    """Bi-encoder → FAISS → cross-encoder, the source project's two-stage shape."""

    def __init__(self) -> None:
        from agent.retrieval.embedder import Embedder
        from agent.retrieval.index import KBIndex
        from agent.retrieval.reranker import Reranker

        self.embedder = Embedder()
        self.index = KBIndex.load(config.KB_INDEX_PATH, config.KB_CHUNKS_PATH)
        self.reranker = Reranker()

    def search(self, query: str):
        hits = self.index.search(self.embedder.encode_query(query), config.RETRIEVE_K)
        return self.reranker.rerank(query, hits, config.RERANK_TOP_N)  # FR6


#: Process-wide singleton. The models are ~2.2GB each and take seconds to load;
#: the CLI builds one registry so it never noticed, but the eval harness builds
#: one per case and would otherwise reload them on every run.
_RETRIEVAL_SINGLETON: _RetrievalStack | None = None


def _get_retrieval_stack() -> "_RetrievalStack":
    global _RETRIEVAL_SINGLETON
    if _RETRIEVAL_SINGLETON is None:
        _RETRIEVAL_SINGLETON = _RetrievalStack()
    return _RETRIEVAL_SINGLETON


class _ClipStack:
    """OpenCLIP encoder + product index. Built only when the tool is used."""

    def __init__(self) -> None:
        from agent.retrieval.clip_store import ClipEncoder, ProductIndex

        self.encoder = ClipEncoder()
        self.index = ProductIndex.load(
            config.PRODUCT_INDEX_PATH, config.PRODUCT_META_PATH
        )


# ==========================================================================
# Entry point used by the CLI
# ==========================================================================


def build_dispatcher(mode: config.Mode, tracer: Tracer, image_resolver=None):
    """Return ``(dispatch, kb_size)`` for the given mode."""
    registry = ToolRegistry(mode, tracer, image_resolver=image_resolver)
    return registry.dispatch, registry.kb_size
