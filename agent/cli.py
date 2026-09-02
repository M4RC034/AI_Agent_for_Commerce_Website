"""Streaming CLI entrypoint.

    python -m agent.cli                      # Mode A (vector retrieval)
    python -m agent.cli --mode cag           # Mode B (KB in the cached prefix)
    python -m agent.cli --show-thinking      # render summarised reasoning
    python -m agent.cli --once "運費多少？"    # one turn, for scripted use

``--mode`` is the only thing that differs between the two RAG strategies (FR5,
NFR4); it selects a context provider and nothing downstream branches on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import anthropic
from rich.console import Console

from agent import config
from agent.context import build_provider
from agent.loop import AgentLoop
from agent.memory import Conversation
from agent.render import Renderer
from agent.trace import Tracer
from agent.tools.registry import build_dispatcher

EXIT_COMMANDS = {"/exit", "/quit", "/q", "exit", "quit"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="電子商店客服 AI 助理（agentic loop, Claude tool_use）",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in config.Mode],
        default=config.Mode.RAG.value,
        help="rag：向量檢索作為工具；cag：知識庫直接放進被快取的系統提示詞",
    )
    parser.add_argument("--session", default=None, help="工作階段 ID（預設隨機產生）")
    parser.add_argument(
        "--show-thinking", action="store_true", help="顯示模型的摘要式思考過程"
    )
    parser.add_argument("--no-log", action="store_true", help="停用 JSONL 追蹤記錄")
    parser.add_argument("--once", default=None, help="只處理單一輪對話後結束")
    return parser.parse_args(argv)


def build_loop(args: argparse.Namespace, console: Console) -> AgentLoop:
    mode = config.Mode(args.mode)
    session_id = args.session or f"{mode.value}-{uuid.uuid4().hex[:8]}"

    provider = build_provider(mode)
    renderer = Renderer(console=console, show_thinking=args.show_thinking)
    tracer = Tracer(session_id, config.LOG_DIR, enabled=not args.no_log)

    # The dispatcher owns the retrieval stack, so it is what decides whether the
    # embedding model gets loaded at all — Mode B never touches it.
    dispatch, kb_size = build_dispatcher(mode, tracer)

    renderer.banner(mode.value, session_id, kb_size)

    return AgentLoop(
        client=anthropic.Anthropic(),
        provider=provider,
        dispatch=dispatch,
        renderer=renderer,
        tracer=tracer,
        conversation=Conversation(),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]找不到 ANTHROPIC_API_KEY。[/red]\n"
            "請複製 .env.example 為 .env 並填入金鑰，或設定環境變數。"
        )
        return 1

    try:
        loop = build_loop(args, console)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    renderer = loop.renderer

    if args.once:
        renderer.assistant_prefix()
        loop.run_turn(args.once)
        return 0

    while True:
        try:
            user_input = renderer.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再見。[/dim]")
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            console.print("[dim]再見。[/dim]")
            return 0

        renderer.assistant_prefix()
        try:
            loop.run_turn(user_input)
        except anthropic.RateLimitError:
            renderer.error("目前請求過於頻繁，請稍候再試。")
        except anthropic.APIStatusError as exc:
            renderer.error(f"API 錯誤（{exc.status_code}）：{exc.message}")
        except anthropic.APIConnectionError:
            renderer.error("連線失敗，請檢查網路後再試。")
        except KeyboardInterrupt:
            console.print("\n[dim]已中斷本輪。[/dim]")

        console.print()


if __name__ == "__main__":
    sys.exit(main())
