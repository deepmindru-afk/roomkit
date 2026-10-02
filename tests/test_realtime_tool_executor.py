"""Every door of a speech-to-speech channel serves a call the same way (RFC §12.4, RMK-306).

The provider's function call, a call recovered from speech and a reasoning
backend's call run inside the same tool call context, a backend reads a
call's outcome with whether it failed, and Tool Search's results are bounded
as any tool result is, a tool's complete schema aside.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit, ToolCallResult
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.tools import current_tool_actor_id, current_tool_call
from roomkit.tools.result import bounded_result
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import until
from tests.test_realtime_fixed_tools import call, channel_context, tool
from tests.test_realtime_skills import _registry_with_skill

LOOKUP = {"name": "lookup", "description": "Look up", "parameters": {"type": "object"}}
REFUSED = {"name": "refused", "description": "Always refused", "parameters": {"type": "object"}}


class _Handler:
    def __init__(self) -> None:
        self.contexts: list[tuple[str, str, str]] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "refused":
            raise ToolRefusedError("not for you")
        ctx = current_tool_call()
        assert ctx is not None
        assert current_tool_actor_id() == "u1"
        self.contexts.append((ctx.room_id, ctx.tool_call_id, ctx.channel_id))
        return '{"found": true}'


class _Backend(ReasoningBackend):
    """Calls each tool of *names* once through ``execute_tool_call``."""

    def __init__(self, *names: str, arguments: dict[str, Any] | None = None) -> None:
        self.names = names
        self.arguments = arguments or {}
        self.results: list[ToolCallResult] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        assert request.execute_tool_call is not None
        for name in self.names:
            self.results.append(await request.execute_tool_call(name, dict(self.arguments)))
        yield ReasoningOutput("done", is_final=True)


async def _channel(handler: Any, **kwargs: Any) -> tuple[RoomKit, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider(full_duplex="reasoning_backend" in kwargs)
    kwargs.setdefault("tools", [LOOKUP, REFUSED])
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), tool_handler=handler, **kwargs
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, provider, await channel.start_session("r1", "u1", "ws")


async def test_every_door_runs_its_call_in_the_tool_call_context() -> None:
    handler, backend = _Handler(), _Backend("lookup")
    kit, provider, session = await _channel(handler, reasoning_backend=backend)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await provider.simulate_transcription(session, "call:lookup{}", "assistant")
    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: len(handler.contexts) == 3)

    assert {(room, channel) for room, _, channel in handler.contexts} == {("r1", "rt")}
    ids = [call_id for _, call_id, _ in handler.contexts]
    assert "c1" in ids
    assert any(i.startswith("recovered-") for i in ids)
    assert any(i.startswith("d1:") for i in ids)
    await kit.close()


async def test_a_backend_reads_whether_a_call_failed() -> None:
    backend = _Backend("lookup", "refused", "missing")
    kit, provider, session = await _channel(_Handler(), reasoning_backend=backend)

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: len(backend.results) == 3)

    assert [r.is_error for r in backend.results] == [False, True, True]
    assert json.loads(backend.results[0].text) == {"found": True}
    assert backend.results[1].text == "not for you"
    await kit.close()


MANY = [
    {
        "name": f"tool_{i}",
        "description": f"weather forecast number {i} " + "detail " * 40,
        "parameters": {"type": "object", "properties": {}},
    }
    for i in range(30)
]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("find_tools", {"query": "weather forecast"}), ("list_tools", {})],
    ids=["find", "inventory"],
)
async def test_tool_search_results_are_bounded(name: str, arguments: dict[str, Any]) -> None:
    kit, provider, session = await _channel(
        _Handler(), tools=MANY, tool_search=True, tool_result_max_length=300
    )

    await provider.simulate_tool_call(session, "c1", name, arguments)
    await until(lambda: bool(provider.tool_results))

    sent = provider.tool_results[0][2]
    assert len(sent) <= 300
    assert sent.endswith("characters]")
    await kit.close()


async def test_a_backend_call_named_like_a_skill_tool_reaches_the_host() -> None:
    """A recovered or a backend call reaches the host's tools only (RFC §21.1):
    a channel with skills and no declared catalogue serves a backend's
    ``read_skill_reference`` through its handler, never the skill's reference."""
    import tempfile
    from pathlib import Path

    host: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host.append(name)
        return "host answer"

    with tempfile.TemporaryDirectory() as root:
        registry = _registry_with_skill(Path(root), references=[("guide.md", "SECRET REFERENCE")])
        backend = _Backend(
            "read_skill_reference", arguments={"skill_name": "test-skill", "filename": "guide.md"}
        )
        kit, provider, session = await _channel(
            handler, tools=None, skills=registry, reasoning_backend=backend
        )
        await provider.simulate_delegation(session, "d1", "integrator")
        await until(lambda: bool(backend.results))

    assert host == ["read_skill_reference"]
    assert "SECRET" not in backend.results[0].text
    await kit.close()


async def test_a_call_tool_span_names_the_tool_it_carries() -> None:
    telemetry = MockTelemetryProvider()
    tools = [tool("calendar"), tool("projects")] + [tool(f"filler_{i}") for i in range(110)]
    async with channel_context(tools=tools) as ctx:
        _, channel, provider, session, _ = ctx
        channel._telemetry = telemetry
        await call(
            channel,
            provider,
            session,
            "call_tool",
            {"name": "calendar", "arguments_json": '{"action":"list"}'},
        )

    names = [s.name for s in telemetry.get_spans(SpanKind.REALTIME_TOOL_CALL)]
    assert "realtime_tool:calendar" in names


def test_a_bound_shorter_than_its_note_still_holds() -> None:
    assert bounded_result("x" * 100, 10, "t") == "x" * 10
    cut = bounded_result("x" * 1000, 100, "t")
    assert len(cut) == 100 and cut.endswith("[truncated: the result was 1000 characters]")
