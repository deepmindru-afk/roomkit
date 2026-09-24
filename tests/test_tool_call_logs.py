"""What a tool call leaves in the logs (RMK-219).

INFO names the tool, its argument keys, the result size and the duration;
argument values and the result itself, which can carry personal data, appear
only at DEBUG, and only with content logging on (``ROOMKIT_LOG_CONTENT``).
"""

from __future__ import annotations

from contextlib import contextmanager

from roomkit.channels.ai import AIChannel, _current_loop_ctx, _ToolLoopContext
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry.redaction import set_content_logging

_LOGGER = "roomkit.channels.ai"


async def _handler(name: str, arguments: dict) -> str:
    return '{"cards": ["RMK-1", "RMK-2"]}'


def _channel() -> AIChannel:
    call = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            AIToolCall(id="t1", name="find_cards", arguments={"query": "alice@example.com"})
        ],
    )
    done = AIResponse(content="Two cards.", finish_reason="stop")
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=[call, done], streaming=True),
        tool_handler=_handler,
    )


@contextmanager
def _in_a_turn():
    token = _current_loop_ctx.set(_ToolLoopContext())
    try:
        yield
    finally:
        _current_loop_ctx.reset(token)


async def _run_turn(caplog, level: str) -> list[str]:
    caplog.set_level(level, logger=_LOGGER)
    ch = _channel()
    with _in_a_turn():
        ctx = AIContext(messages=[AIMessage(role="user", content="my cards?")])
        [d async for d in ch._run_streaming_tool_loop(ctx)]
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


async def test_info_names_the_call_but_not_its_values(caplog) -> None:
    lines = await _run_turn(caplog, "INFO")

    assert "Executing tool find_cards (call t1) with query" in lines
    assert any(
        line.startswith("Tool find_cards returned 29 chars in ") and line.endswith(" ms")
        for line in lines
    )
    assert not any("alice@example.com" in line or "RMK-1" in line for line in lines)


async def test_debug_redacts_the_values_by_default(caplog) -> None:
    lines = await _run_turn(caplog, "DEBUG")

    assert "Tool find_cards arguments: <redacted:30 chars>" in lines
    assert not any("alice@example.com" in line for line in lines)


async def test_debug_with_content_logging_shows_the_arguments_and_the_result(caplog) -> None:
    set_content_logging(True)
    try:
        lines = await _run_turn(caplog, "DEBUG")
    finally:
        set_content_logging(False)

    assert 'Tool find_cards arguments: {"query": "alice@example.com"}' in lines
    assert 'Tool find_cards result: {"cards": ["RMK-1", "RMK-2"]}' in lines
