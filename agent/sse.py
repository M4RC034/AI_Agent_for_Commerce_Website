"""Server-sent-event renderer — the web counterpart to ``agent/render.py``.

``AgentLoop`` talks to its renderer through four methods (``consume``,
``tool_running``, ``tool_receipt``, ``notice``). That is the whole contract, so
a second implementation can push the same moments to a browser instead of a
terminal without the loop knowing which one it has. The orchestration stays in
one place; only the presentation differs.

Events emitted, each a JSON object on an SSE ``data:`` line:

    {"type": "thinking",   "text": "..."}   summarised reasoning delta
    {"type": "token",      "text": "..."}   answer token
    {"type": "tool_start", "tool": "...", "label": "正在查詢您的訂單…"}
    {"type": "tool_end",   "tool": "...", "summary": "...", "ok": true}
    {"type": "notice",     "text": "..."}   step-budget close-out
    {"type": "done",       "steps": 3}
    {"type": "error",      "text": "..."}
"""

from __future__ import annotations

import queue
from contextlib import contextmanager
from typing import Any

from agent.tools.schemas import TOOL_LABELS


class SSERenderer:
    """Renderer that pushes loop events onto a queue for an SSE response.

    The agent loop is synchronous and the SSE response is a generator, so the
    loop runs on a worker thread and communicates through this queue. That is
    also why nothing here blocks: a slow browser must never stall the loop.
    """

    def __init__(self, sink: "queue.Queue[dict[str, Any]]", show_thinking: bool = False) -> None:
        self.sink = sink
        self.show_thinking = show_thinking
        self._pending_tool: str | None = None

    def _emit(self, **event: Any) -> None:
        self.sink.put(event)

    # -- the AgentLoop renderer contract -----------------------------------

    def consume(self, stream) -> None:
        for event in stream:
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    # Announced here rather than in tool_running so the browser
                    # shows the decision the moment the model commits to it,
                    # before the executor has done any work.
                    self._pending_tool = block.name
                    self._emit(
                        type="tool_start",
                        tool=block.name,
                        label=TOOL_LABELS.get(block.name, "處理中…"),
                    )

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    self._emit(type="token", text=delta.text)
                elif delta.type == "thinking_delta" and self.show_thinking:
                    self._emit(type="thinking", text=delta.thinking)

    @contextmanager
    def tool_running(self, name: str):
        # The spinner is the browser's job; the start event already fired.
        yield

    def tool_receipt(self, summary: str, ok: bool = True) -> None:
        self._emit(type="tool_end", tool=self._pending_tool, summary=summary, ok=ok)
        self._pending_tool = None

    def notice(self, text: str) -> None:
        self._emit(type="notice", text=text)

    # -- CLI-only methods, unused here but kept for interface parity --------

    def error(self, text: str) -> None:
        self._emit(type="error", text=text)
