"""Structured session tracing (NFR3).

Every tool call, retrieval query, and model judgment lands in a JSONL file under
``logs/``. Two things read it: the eval harness, which scores *trajectories*
rather than only final answers, and you, when writing the README's AI-tool-usage
disclosure and the RAG-vs-CAG trade-off — the token and latency numbers there
should come from real traces, not estimates.

One record per event, one file per session, append-only. Nothing here raises:
a broken log must never take down a customer conversation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    """Append-only JSONL event log for one session."""

    def __init__(self, session_id: str, log_dir: Path, enabled: bool = True) -> None:
        self.session_id = session_id
        self.enabled = enabled
        self.path = log_dir / f"{session_id}.jsonl"
        self.turn = 0
        self._t0 = time.monotonic()

        if self.enabled:
            log_dir.mkdir(parents=True, exist_ok=True)

    # -- core --------------------------------------------------------------

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            "session_id": self.session_id,
            "turn": self.turn,
            "event": event,
            **fields,
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # Logging is diagnostic, never load-bearing.
            pass

    # -- typed helpers -----------------------------------------------------

    def start_turn(self, user_input: str) -> None:
        self.turn += 1
        self.emit("user_turn", text=user_input)

    def model_response(self, step: int, message: Any, latency_s: float) -> None:
        """Record what the model decided, and what it cost.

        ``cache_read_input_tokens`` is the load-bearing field for the CAG mode
        write-up: zero across repeated turns means the cached prefix is being
        invalidated by something volatile.
        """
        usage = getattr(message, "usage", None)
        self.emit(
            "model_response",
            step=step,
            stop_reason=message.stop_reason,
            latency_s=round(latency_s, 3),
            tool_uses=[b.name for b in message.content if b.type == "tool_use"],
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
            },
        )

    def tool_call(self, step: int, name: str, tool_input: dict) -> None:
        self.emit("tool_call", step=step, tool=name, input=tool_input)

    def tool_result(
        self, step: int, name: str, ok: bool, payload: str, latency_s: float
    ) -> None:
        self.emit(
            "tool_result",
            step=step,
            tool=name,
            ok=ok,
            latency_s=round(latency_s, 3),
            payload=payload[:2000],
        )

    def retrieval(
        self,
        query: str,
        reformulation_of: str | None,
        attempt: int,
        verdict: str,
        top_score: float | None,
        chunk_ids: list[str],
    ) -> None:
        """FR2/FR3/FR4 evidence in one record.

        ``reformulation_of`` being non-null is what proves the self-correcting
        retrieval loop actually fired, rather than being merely available.
        """
        self.emit(
            "retrieval",
            query=query,
            reformulation_of=reformulation_of,
            attempt=attempt,
            sufficiency=verdict,
            top_score=top_score,
            chunk_ids=chunk_ids,
        )

    def guard_tripped(self, reason: str, **fields: Any) -> None:
        """A loop or retrieval budget was exhausted (FR13)."""
        self.emit("guard_tripped", reason=reason, **fields)

    def final(self, text: str, steps_used: int) -> None:
        self.emit("final", text=text, steps_used=steps_used)
