"""RoomKit's decision on an ACP call's permission stands, however the agent
ends it (RMK-480, RFC §9.3).

A call whose permission the external tool handler refused is reported
refused, with the handler's reason, as on the AI door; one whose handler
raised is reported failed, with what failed. Whether the agent closes the
call failed, leaves it open as its turn ends, or never announced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import acp
import pytest
from acp import PromptResponse
from acp.schema import PermissionOption

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel


class _Denying(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=False, reason="not allowed here")


class _Raising(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        raise RuntimeError("approval service down")


_OPTIONS = [
    PermissionOption(option_id="a", name="Allow", kind="allow_once"),
    PermissionOption(option_id="r", name="Reject", kind="reject_once"),
]


async def _run(tmp_path: Path, handler: Any, *, starts: bool, closes: bool) -> tuple[Any, Any]:
    """One ACP turn whose agent asks permission for ``Write``; the call's
    reports and its stored END rows."""

    async def prompt(conn: Any, session_id: str, *args: Any, **kwargs: Any) -> PromptResponse:
        if starts:
            update = acp.start_tool_call(
                "tool-1", "Write", kind="edit", status="in_progress", raw_input={"path": "x"}
            )
            await conn.client.session_update(session_id, update)
        request = acp.update_tool_call("tool-1", title="Write", raw_input={"path": "x"})
        await conn.client.request_permission(session_id, request, _OPTIONS)
        if closes:
            closed = acp.update_tool_call(
                "tool-1", status="failed", raw_output={"error": "permission rejected"}
            )
            await conn.client.session_update(session_id, closed)
        return PromptResponse(stop_reason="end_turn")

    kit = RoomKit()
    channel, conn, _ = _channel(tmp_path, handler=handler, emit_updates=False)
    conn.prompt = lambda *a, **k: prompt(conn, *a, **k)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "sms")
    await kit.attach_channel("room-1", "acp-agent", category=ChannelCategory.INTELLIGENCE)
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def _audit(event: Any, ctx: Any) -> None:
        reports.append(event)

    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    await kit.process_inbound(message)
    rows = [
        e.content.outcome
        for e in await kit.store.list_events("room-1")
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    ]
    await kit.close()
    return reports, rows


ENDINGS = pytest.mark.parametrize(
    ("starts", "closes"),
    [(True, False), (False, False), (True, True)],
    ids=["left-open", "never-announced", "closed-failed"],
)


@ENDINGS
async def test_a_refused_permission_is_reported_refused(
    tmp_path: Path, starts: bool, closes: bool
) -> None:
    reports, rows = await _run(tmp_path, _Denying(), starts=starts, closes=closes)

    assert [(e.refused, e.cancelled) for e in reports] == [(True, False)]
    assert rows == ["refused"]
    if not closes:
        assert reports[0].result == '{"error": "not allowed here"}'


@ENDINGS
async def test_a_permission_whose_handler_raised_is_reported_failed(
    tmp_path: Path, starts: bool, closes: bool
) -> None:
    reports, rows = await _run(tmp_path, _Raising(), starts=starts, closes=closes)

    assert [(e.is_error, e.refused, e.cancelled) for e in reports] == [(True, False, False)]
    assert reports[0].error_detail == "RuntimeError: approval service down"
    assert rows == ["failed"]


async def test_a_raising_handler_s_call_left_open_reads_as_the_ai_door_reads_it(
    tmp_path: Path,
) -> None:
    reports, _ = await _run(tmp_path, _Raising(), starts=True, closes=False)

    assert reports[0].result == '{"error": "Tool \'Write\' failed (RuntimeError)"}'


class _ApprovesWithAnOverride(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=True, reason="fine by me", modified_input={"path": "y"})


async def test_an_approval_acp_cannot_apply_is_refused_with_the_channel_s_reason(
    tmp_path: Path,
) -> None:
    reports, rows = await _run(tmp_path, _ApprovesWithAnOverride(), starts=True, closes=False)

    assert [(e.refused, e.cancelled) for e in reports] == [(True, False)]
    assert "fine by me" not in str(reports[0].result)
    assert "ACP cannot apply" in str(reports[0].result)
    assert rows == ["refused"]
