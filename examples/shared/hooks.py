"""Reusable hook helpers for RoomKit examples."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.telemetry.redaction import set_content_logging

_MAGENTA = "\033[35m"
_RESULT_PREVIEW = 300
_RESET = "\033[0m"


def log_tool_call(
    event,
    *,
    tool_names: Sequence[str] | None = None,
    label: str = "tool",
    show_result: bool = False,
) -> HookResult:
    """Format and print a tool call event, then return ``HookResult.allow()``.

    Intended as a helper called from an ``ON_TOOL_CALL`` hook body.

    Args:
        event: The ``ToolCallEvent`` passed to the hook.
        tool_names: If provided, only log calls matching these names.
        label: Display label shown in brackets (default ``"tool"``).
        show_result: Also print the start of what the tool returned, so a
            reply can be checked against it.

    Usage::

        from shared.hooks import log_tool_call

        @kit.hook(HookTrigger.ON_TOOL_CALL)
        async def show_tool_call(event, _ctx):
            return log_tool_call(event, tool_names=["activate_skill"], label="skill")
    """
    if tool_names is not None and event.name not in set(tool_names):
        return HookResult.allow()
    args = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
    print(f"\n{_MAGENTA}  [{label}] {event.name}({args}){_RESET}")
    if show_result and event.result is not None:
        text = str(event.result).replace("\n", " ")
        cut = (
            f"{text[:_RESULT_PREVIEW]}… ({len(text)} chars)"
            if len(text) > _RESULT_PREVIEW
            else text
        )
        print(f"{_MAGENTA}  [{label}] → {cut}{_RESET}")
    print()
    return HookResult.allow()


def enable_voice_debug(kit: RoomKit) -> None:
    """Turn-taking diagnostics: when speech starts and ends, and what became of it.

    Sets RoomKit's voice and AI loggers to DEBUG (the interruption decisions,
    suppressed segments, STT streams, the TTS cache, the AI turns, tool
    arguments and results), turns content logging on for them, and logs each
    speech segment's edges, so a reply can be traced back to the words that
    caused it. Local runs only: what was heard and said reaches the logs.
    """
    for name in ("roomkit.voice", "roomkit.channels.ai"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    set_content_logging(True)
    # The per-second pipeline lines drown the decisions.
    logging.getLogger("roomkit.voice.pipeline").setLevel(logging.INFO)
    logger = logging.getLogger("examples.voice_debug")

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(event, ctx):
        logger.info("[debug] speech start")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(event, ctx):
        logger.info("[debug] speech end")
