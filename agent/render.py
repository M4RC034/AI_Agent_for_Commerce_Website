"""Terminal rendering for the streaming CLI (FR17, FR18).

Two requirements pull in opposite directions here. FR17 wants the final answer
token-by-token, which means consuming the stream as it arrives. FR18 wants the
user never to face a silent pause, and tool execution — a FAISS search, a
reranker forward pass — is exactly where the silence would fall.

The renderer therefore tracks four visual states:

    thinking      dim, under 思考中…   (only with --show-thinking)
    text          printed as it streams
    tool starting the zh-TW label from schemas.TOOL_LABELS
    tool running  a spinner, held until the executor returns

Note the model's *thinking* is what happens before the first token, and the
model's *tools* are what happens between tokens. Both are covered.
"""

from __future__ import annotations

from contextlib import contextmanager

from rich.console import Console

from agent.tools.schemas import TOOL_LABELS


class Renderer:
    """Renders one agent turn to the terminal."""

    def __init__(self, console: Console | None = None, show_thinking: bool = False) -> None:
        self.console = console or Console()
        self.show_thinking = show_thinking
        self._in_thinking = False
        self._wrote_text = False

    # -- streaming ---------------------------------------------------------

    def consume(self, stream) -> None:
        """Drain one model stream, rendering as it goes.

        Tool *inputs* arrive as ``input_json_delta`` fragments and are
        deliberately not parsed here — partial JSON is not valid JSON. The
        assembled input comes from ``stream.get_final_message()`` in the loop.
        """
        for event in stream:
            if event.type == "content_block_start":
                self._on_block_start(event)
            elif event.type == "content_block_delta":
                self._on_delta(event)
            elif event.type == "content_block_stop":
                self._close_thinking()

        self._close_thinking()
        if self._wrote_text:
            self.console.print()
            self._wrote_text = False

    def _on_block_start(self, event) -> None:
        block_type = event.content_block.type

        if block_type == "thinking" and self.show_thinking:
            self.console.print("\n[dim]思考中…[/dim]")
            self._in_thinking = True

        elif block_type == "tool_use":
            # FR18: announce the tool before it runs, not after.
            self._close_thinking()
            label = TOOL_LABELS.get(event.content_block.name, "處理中…")
            self.console.print(f"[cyan]⟳[/cyan] [dim]{label}[/dim]")

    def _on_delta(self, event) -> None:
        delta = event.delta

        if delta.type == "thinking_delta" and self.show_thinking:
            self.console.print(f"[dim]{delta.thinking}[/dim]", end="")

        elif delta.type == "text_delta":
            self._close_thinking()
            self.console.print(delta.text, end="", highlight=False, markup=False)
            self._wrote_text = True

    def _close_thinking(self) -> None:
        if self._in_thinking:
            self.console.print()
            self._in_thinking = False

    # -- tool execution ----------------------------------------------------

    @contextmanager
    def tool_running(self, name: str):
        """Hold a spinner for the duration of a tool call.

        This is the gap FR18 is really about: the model has already said what it
        wants to do, and the executor may take a second or more to answer.
        """
        label = TOOL_LABELS.get(name, "處理中…")
        with self.console.status(f"[dim]{label}[/dim]", spinner="dots"):
            yield

    def tool_receipt(self, summary: str, ok: bool = True) -> None:
        """One-line confirmation of what a tool actually returned."""
        mark = "[green]✓[/green]" if ok else "[yellow]![/yellow]"
        self.console.print(f"{mark} [dim]{summary}[/dim]")

    # -- chrome ------------------------------------------------------------

    def banner(self, mode: str, session_id: str, kb_chunks: int | None) -> None:
        kb = f"，知識庫 {kb_chunks} 段" if kb_chunks is not None else ""
        self.console.print()
        self.console.print("[bold]小電[/bold] [dim]· 電子商店客服助理[/dim]")
        self.console.print(f"[dim]模式 {mode.upper()}{kb}　工作階段 {session_id}[/dim]")
        self.console.print("[dim]輸入問題開始對話，輸入 /exit 結束。[/dim]")
        self.console.print()

    def prompt(self) -> str:
        return self.console.input("[bold cyan]您[/bold cyan] › ")

    def assistant_prefix(self) -> None:
        self.console.print("[bold]小電[/bold] › ", end="")

    def notice(self, text: str) -> None:
        self.console.print(f"[yellow]{text}[/yellow]")

    def error(self, text: str) -> None:
        self.console.print(f"[red]{text}[/red]")
