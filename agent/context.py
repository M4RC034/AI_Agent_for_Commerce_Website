"""Retrieval mode as a provider, not a branch (FR5, NFR4).

NFR4 forbids duplicating code paths for the two retrieval modes. The naive
reading of FR5 produces ``if mode == "cag"`` scattered through retrieval, prompt
assembly and the loop; instead the entire difference collapses into two methods.

    Mode A (RAG)  system = [base + rag addendum]
                  tools  = [retrieve_kb, *business]

    Mode B (CAG)  system = [base + cag addendum, <whole KB, cache_control>]
                  tools  = [*business]              # no retrieve_kb at all

``AgentLoop`` calls ``system_blocks()`` and ``tools()`` and never learns which
mode it is in. That is what makes AC7's cross-mode consistency check meaningful
rather than coincidental — there is only one loop to be consistent about.

Cache placement
    Render order is ``tools`` → ``system`` → ``messages``, so a breakpoint on the
    *last* system block caches the tool schemas and the system prompt together.
    Both modes get one. Nothing volatile appears in either prefix — today's date
    arrives via tool results, and the entity-ledger hint is a mid-conversation
    ``role: "system"`` message that sits after the cached history.
"""

from __future__ import annotations

from typing import Any, Protocol

from agent import config, prompts
from agent.tools import schemas


class ContextProvider(Protocol):
    """What the agent loop needs in order to make a model call."""

    mode: config.Mode

    def system_blocks(self) -> list[dict[str, Any]]: ...

    def tools(self) -> list[dict[str, Any]]: ...


class RagProvider:
    """Mode A — the knowledge base is reachable only through a tool.

    Retrieval becomes a decision rather than a preprocessing step (FR2): the
    model calls ``retrieve_kb`` when it judges the turn needs policy content,
    may call it again with a reformulated query when the sufficiency verdict
    comes back short (FR3), and skips it entirely for a pure order lookup.
    """

    mode = config.Mode.RAG

    def system_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": prompts.BASE_PROMPT + prompts.RAG_ADDENDUM,
                # Caches the tool schemas too — they render before system.
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def tools(self) -> list[dict[str, Any]]:
        return schemas.RETRIEVAL_TOOLS + schemas.BUSINESS_TOOLS + _multimodal_tools()


class CagProvider:
    """Mode B — the knowledge base rides along in the cached system prefix.

    No vector search happens at all: no embedding, no FAISS, no reranking. The
    trade-off is that every turn carries the whole corpus, paid for once per
    cache write and at roughly a tenth of input price on every read after that.

    The business tools stay available. CAG replaces *retrieval*, not the tool
    layer — an order lookup still has to hit the fixture store either way.
    """

    mode = config.Mode.CAG

    def __init__(self, kb_dir=None) -> None:
        # Read once at construction. Re-reading per turn would be both wasteful
        # and a caching hazard if a file changed mid-session.
        self._kb_text = prompts.load_kb_text(kb_dir or config.KB_DIR)

    @property
    def kb_text(self) -> str:
        return self._kb_text

    def system_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": prompts.BASE_PROMPT + prompts.CAG_ADDENDUM,
            },
            {
                "type": "text",
                "text": self._kb_text,
                # 1h rather than the 5-minute default: the KB write is the
                # expensive part of this mode, and a longer TTL keeps it alive
                # across eval runs and across gaps in a conversation. Break-even
                # is three reads at 2x write cost.
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]

    def tools(self) -> list[dict[str, Any]]:
        return schemas.BUSINESS_TOOLS + _multimodal_tools()


def _multimodal_tools() -> list[dict[str, Any]]:
    """Offer image search only when its index exists.

    An advertised tool that always errors is worse than no tool: the model will
    keep reaching for it and burn steps discovering it is broken.
    """
    if config.PRODUCT_INDEX_PATH.exists() and config.PRODUCT_META_PATH.exists():
        return schemas.MULTIMODAL_TOOLS
    return []


def build_provider(mode: config.Mode) -> ContextProvider:
    """Single point where the runtime ``--mode`` parameter is interpreted."""
    return CagProvider() if mode is config.Mode.CAG else RagProvider()
