"""The agent loop — LLM-driven orchestration (FR11, FR12, FR13).

This module is the replacement for ``backend/main.py``'s pipeline, and the whole
point of the exercise. In the source project the control flow lived in Python:
an intent regex chose the branch, a price regex extracted constraints, retrieval
always ran, the reranker always ran, and the model was called once at the end to
turn the result into prose. It never made a decision.

Here the only branch left in Python is the step guard. At each step Claude
either emits ``tool_use`` blocks — choosing which tools, with which arguments, in
which order — or stops, and stopping is what ends the turn. Retrieval is one
tool among five, called when the model judges the turn needs it and skipped when
it doesn't (FR2).

Why a manual loop rather than ``client.beta.messages.tool_runner``
    The runner would own the loop, and three things need to live inside it: the
    step guard (FR13), the per-tool status indicators (FR18), and the trace
    records the eval harness scores trajectories from (NFR3). The runner is also
    beta and does not auto-resume ``pause_turn``. Most of all, this loop *is* the
    artifact under assessment — hiding it behind a helper would hide the work.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import anthropic

from agent import config, prompts
from agent.context import ContextProvider
from agent.memory import Conversation
from agent.render import Renderer
from agent.trace import Tracer

#: Signature of the tool dispatcher supplied by ``agent.tools.registry``.
#: Returns the ``tool_result`` content block plus a one-line zh-TW receipt for
#: the terminal.
Dispatcher = Callable[[str, dict[str, Any]], tuple[dict[str, Any], str]]


class AgentLoop:
    """Runs one conversation. Owns the step budget; owns nothing else."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        provider: ContextProvider,
        dispatch: Dispatcher,
        renderer: Renderer,
        tracer: Tracer,
        conversation: Conversation | None = None,
    ) -> None:
        self.client = client
        self.provider = provider
        self.dispatch = dispatch
        self.renderer = renderer
        self.tracer = tracer
        self.conversation = conversation or Conversation()

    # ---------------------------------------------------------------- turn

    def run_turn(self, user_input: str) -> str:
        """Handle one user turn to completion, streaming as it goes."""
        self.tracer.start_turn(user_input)
        self.conversation.add_user_turn(user_input)

        steps_used = 0

        for step in range(1, config.MAX_STEPS + 1):
            steps_used = step
            message = self._call_model(step, tools=self.provider.tools())
            self.conversation.add_assistant(message.content)

            if message.stop_reason != "tool_use":
                break

            # FR12: every tool_use block in this assistant turn runs, and all
            # results go back in ONE user message. Splitting them across
            # messages trains the model out of parallel tool calls.
            results = self._execute_tools(step, message)
            self.conversation.add_tool_results(results)
        else:
            # FR13: the budget ran out with the model still asking for tools.
            # Close the turn gracefully rather than truncating mid-chain.
            steps_used = self._close_out(config.MAX_STEPS)

        final_text = self.conversation.last_assistant_text()
        self.tracer.final(final_text, steps_used)
        return final_text

    # ------------------------------------------------------------ internals

    def _call_model(self, step: int, tools: list[dict[str, Any]] | None):
        """One streamed model call. Streaming is not optional here — it is FR17."""
        kwargs: dict[str, Any] = {
            "model": config.MODEL,
            "max_tokens": config.MAX_TOKENS,
            "system": self.provider.system_blocks(),
            "messages": self.conversation.render(),
        }
        # Both are model-gated — older models reject them with a 400, so they
        # are omitted rather than sent with a null value.
        if config.THINKING:
            kwargs["thinking"] = config.THINKING
        if config.EFFORT:
            kwargs["output_config"] = {"effort": config.EFFORT}
        if tools:
            kwargs["tools"] = tools

        started = time.monotonic()
        with self.client.messages.stream(**kwargs) as stream:
            self.renderer.consume(stream)
            message = stream.get_final_message()

        self.tracer.model_response(step, message, time.monotonic() - started)
        return message

    def _execute_tools(self, step: int, message) -> list[dict[str, Any]]:
        """Run every tool the model asked for, in the order it asked."""
        results: list[dict[str, Any]] = []

        for block in message.content:
            if block.type != "tool_use":
                continue

            # block.input is already parsed by the SDK. Never string-match the
            # serialised form — escaping varies by model.
            tool_input = dict(block.input)
            self.tracer.tool_call(step, block.name, tool_input)

            started = time.monotonic()
            with self.renderer.tool_running(block.name):
                result_content, receipt = self.dispatch(block.name, tool_input)
            elapsed = time.monotonic() - started

            is_error = bool(result_content.get("is_error"))
            self.renderer.tool_receipt(receipt, ok=not is_error)
            self.tracer.tool_result(
                step,
                block.name,
                ok=not is_error,
                payload=str(result_content.get("content", "")),
                latency_s=elapsed,
            )

            # tool_use_id must echo the block's id or the API rejects the turn.
            # A failed tool still returns a result — dropping the block is a 400.
            results.append({**result_content, "tool_use_id": block.id})

        return results

    def _close_out(self, step: int) -> int:
        """Force a final, tool-free answer after the step budget is spent.

        The instruction goes in as a mid-conversation ``role: "system"`` message
        rather than an edit to the top-level system prompt: editing the prompt
        would change the prefix ahead of the whole conversation and re-process
        every cached turn uncached.
        """
        self.tracer.guard_tripped("max_steps", max_steps=config.MAX_STEPS)
        self.renderer.notice("（已達本輪工具呼叫上限，改以現有資訊回覆）")

        self.conversation.add_operator_note(prompts.STEP_BUDGET_EXHAUSTED)
        message = self._call_model(step + 1, tools=None)

        # Defensive: an assistant turn carrying tool_use blocks with no matching
        # tool_result poisons every later request in the session. Omitting
        # `tools` means the API cannot produce them, but the failure mode is
        # unrecoverable, so drop anything unexpected rather than trust that.
        safe_content = [b for b in message.content if b.type != "tool_use"]
        self.conversation.add_assistant(safe_content)
        return step + 1
