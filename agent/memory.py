"""Conversation memory and entity resolution (FR14–FR16).

FR14 asks for the full history on every model call, with no truncation, and that
is what ``Conversation`` does. Crucially it retains assistant ``tool_use`` blocks
and their results, not just prose — an order number that only ever appeared
inside a tool return is otherwise invisible three turns later, which is exactly
what AC5 probes.

FR15 is mostly satisfied by that alone. The ``EntityLedger`` is a belt-and-braces
addition for the case where the reference is vague ("那筆訂單") and the history is
long: it injects a short operator note naming the order IDs seen so far.

That note goes in as a mid-conversation ``role: "system"`` message rather than
text inside the user turn. Two reasons — it sits *after* the cached prefix so it
costs nothing in cache invalidation, and it is the non-spoofable operator
channel, so a customer typing "系統提示：忽略先前規則" cannot forge one.
"""

from __future__ import annotations

import re
from typing import Any

from agent import config

#: Matches the fixture order-ID format. Deliberately strict: a loose pattern
#: would scoop up tracking numbers and SKUs into the ledger.
ORDER_ID_RE = re.compile(r"\bORD-\d{4}\b")

#: Referential phrasings that signal "the thing we were just talking about"
#: without naming it. Used only to decide whether to inject the ledger hint —
#: never to route, classify, or answer.
_REFERENTIAL_RE = re.compile(
    r"(那筆|這筆|那個|這個|剛剛|剛才|上面|前面|之前|先前|同一筆|它|他)"
    r"|(that|this|it|the same|earlier|previous)\b",
    re.IGNORECASE,
)


class EntityLedger:
    """Tracks order IDs seen in the conversation and where they came from."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}  # order_id -> turn first seen

    def observe(self, text: str, turn: int) -> None:
        for order_id in ORDER_ID_RE.findall(text or ""):
            self._seen.setdefault(order_id, turn)

    @property
    def order_ids(self) -> list[str]:
        return sorted(self._seen, key=lambda oid: self._seen[oid])

    def hint(self, user_text: str) -> str | None:
        """Return an operator note if the turn refers back without naming an ID.

        Returns ``None`` when there is nothing to disambiguate — no known IDs,
        an ID stated explicitly in this very turn, or no referential phrasing.
        """
        if not self._seen:
            return None
        if ORDER_ID_RE.search(user_text or ""):
            return None  # the customer named it; nothing to resolve
        if not _REFERENTIAL_RE.search(user_text or ""):
            return None

        listed = "、".join(
            f"{oid}（第 {self._seen[oid]} 輪提及）" for oid in self.order_ids
        )
        return (
            f"本次對話目前已提及的訂單：{listed}。"
            "顧客這一輪使用了指代性說法，請依對話脈絡判斷是指哪一筆；"
            "若無法確定是哪一筆，請直接向顧客確認，不要臆測。"
        )


def _operator_message(text: str) -> dict[str, Any]:
    """Build an operator note in the strongest form the model supports.

    ``role: "system"`` mid-conversation is the non-spoofable operator channel,
    but it exists only on Opus 5 / 4.8 / Fable 5 / Mythos 5 — Sonnet 5 and
    Haiku 4.5 reject it with a 400. Since AGENT_MODEL lets any model be swapped
    in, the fallback tags the note inside a user turn instead.

    The fallback is genuinely weaker: text inside a user turn can be forged by a
    customer who types the same tag, whereas ``role: "system"`` cannot. Both have
    the same caching profile, so the only thing lost is spoof-resistance.
    Consecutive user messages are merged by the API, so this is well-formed.
    """
    if config.SUPPORTS_MIDCONV_SYSTEM:
        return {"role": "system", "content": text}
    return {
        "role": "user",
        "content": f"<system-reminder>{text}</system-reminder>",
    }


class Conversation:
    """Full, untruncated message history for one session (FR14)."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.ledger = EntityLedger()
        self.turn = 0

    # -- appending ---------------------------------------------------------

    def add_user_turn(self, text: str) -> None:
        self.turn += 1
        self.ledger.observe(text, self.turn)
        self.messages.append({"role": "user", "content": text})

        hint = self.ledger.hint(text)
        if hint:
            # Must follow a user message and be last in the list — both hold here.
            self.messages.append(_operator_message(hint))

    def add_assistant(self, content: Any) -> None:
        """Append the assistant turn verbatim, tool_use blocks included."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """Append every tool result as a single user message.

        Splitting results across multiple messages teaches Claude to stop
        issuing parallel tool calls, so they go back together even when one of
        them errored.
        """
        for result in results:
            content = result.get("content")
            if isinstance(content, str):
                self.ledger.observe(content, self.turn)
        self.messages.append({"role": "user", "content": results})

    def add_operator_note(self, text: str) -> None:
        """Inject an operator instruction without disturbing the cached prefix."""
        self.messages.append(_operator_message(text))

    # -- reading -----------------------------------------------------------

    def render(self) -> list[dict[str, Any]]:
        """The message list handed to the API. No truncation — that is FR14."""
        return self.messages

    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if message["role"] != "assistant":
                continue
            content = message["content"]
            if isinstance(content, str):
                return content
            return "".join(
                block.text
                for block in content
                if getattr(block, "type", None) == "text"
            )
        return ""
