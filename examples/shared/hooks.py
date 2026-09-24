"""Reusable hook helpers for RoomKit examples."""

from __future__ import annotations

from collections.abc import Sequence

from roomkit import HookResult

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
