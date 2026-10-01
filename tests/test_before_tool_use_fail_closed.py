"""BEFORE_TOOL_USE fails closed on every path (RFC §9.3, RMK-313, RMK-306).

It is the gate of a tool call, where an approval hook sits: a hook that
raises or times out refuses the call before it runs. The model reads the
same plain refusal on every channel, never the hook's error; the hook's name
and error reach ON_TOOL_CALL's observers on ``error_detail``. A deliberate
BLOCK's reason is the hook's words for the model, read on every door alike,
and every door emits the ``before_tool_use`` event, hooks or none.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.tools.external import PolicyExternalToolHandler
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_realtime_tool_policy import TOOLS, _Backend, _Calls, _channel
from tests.test_unified_tool_call import _ai_room, _call_one_tool

SECRET = "postgres://admin:hunter2@approvals/internal"
DENIED = "Tool 'delete_account' denied by pre-execution hook."


def _gate(kit: RoomKit, failure: str) -> None:
    """An approval hook that cannot answer: it raises, outlives its timeout, or
    returns something that is not a decision."""

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="approval", timeout=0.05)
    async def approval(event: ToolCallEvent, ctx: RoomContext) -> Any:
        if failure == "raises":
            raise ConnectionError(f"cannot reach {SECRET}")
        if failure == "unusable":
            return None
        await asyncio.sleep(1)
        return HookResult.allow()


_DETAILS = {
    "raises": f"approval: cannot reach {SECRET}",
    "times_out": "approval: timeout (0.05s)",
    "unusable": "approval: expected HookResult, got NoneType",
}


def _observed_detail(failure: str) -> str:
    return _DETAILS[failure]


FAILURES = pytest.mark.parametrize("failure", list(_DETAILS))


@FAILURES
async def test_the_ai_loop_refuses_the_call(streaming: bool, failure: str) -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "{}"

    kit, ch, room_id, observed, _ = await _ai_room(streaming=streaming, tool_handler=handler)
    _gate(kit, failure)

    run = await _call_one_tool(kit, ch, room_id, "get_weather")

    assert ran == []
    assert run.calls[0].result == json.dumps(
        {"error": "Tool 'get_weather' denied by pre-execution hook."}
    )
    assert [(e.is_error, e.error_detail) for e in observed] == [(True, _observed_detail(failure))]
    await kit.close()


@FAILURES
async def test_a_realtime_call_is_refused(failure: str) -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())
    _gate(kit, failure)

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results) and bool(calls.observed))

    assert calls.ran == []
    assert json.loads(provider.tool_results[0][2]) == {"error": DENIED}
    assert [e.error_detail for e in calls.observed] == [_observed_detail(failure)]
    await kit.close()


@FAILURES
async def test_a_recovered_spoken_call_is_refused(failure: str) -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())
    _gate(kit, failure)

    await provider.simulate_transcription(session, "call:delete_account{id:42}", "assistant")
    await until(lambda: bool(provider.injected_texts) and bool(calls.observed))

    assert calls.ran == []
    told = " ".join(text for _sid, text, _role in provider.injected_texts)
    assert "denied" in told and "hunter2" not in told
    assert [e.error_detail for e in calls.observed] == [_observed_detail(failure)]
    await kit.close()


@FAILURES
async def test_a_reasoning_backend_call_is_refused(failure: str) -> None:
    calls, backend = _Calls(), _Backend()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy(), backend=backend)
    _gate(kit, failure)

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(backend.results) and bool(calls.observed))

    assert calls.ran == []
    assert json.loads(backend.results[0]) == {"error": DENIED}
    assert [e.error_detail for e in calls.observed] == [_observed_detail(failure)]
    await kit.close()


@FAILURES
async def test_a_conference_call_is_refused(failure: str) -> None:
    calls = _Calls()
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=calls.conference
        ),
    )
    _gate(kit, failure)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        calls.observed.append(event)

    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None
    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results) and bool(calls.observed))

    assert calls.ran == []
    assert json.loads(provider.tool_results[0][2]) == {"error": DENIED}
    assert [e.error_detail for e in calls.observed] == [_observed_detail(failure)]
    await kit.close()


@FAILURES
async def test_an_external_handler_denies_the_call(failure: str) -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    handler = PolicyExternalToolHandler()
    kit._wire_external_tool_handler("agent", handler)
    _gate(kit, failure)

    decision = await handler.process_tool_call("delete_account", {"id": "42"}, room_id="r1")

    # The handler decides and reports the call itself (RFC §9.3): the same
    # refusal for the agent, the hook's error for the log.
    assert not decision.approved
    assert decision.reason == DENIED
    await kit.close()


async def test_a_hook_that_answers_still_lets_the_call_run() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="approval")
    async def approval(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert calls.ran == ["delete_account"]
    await kit.close()


async def test_a_deliberate_block_keeps_its_reason_beside_a_failed_hook() -> None:
    """Where a host made the trigger fail open again, a hook that raised did not
    refuse the call: the one that blocked did, in its own words, with no detail."""
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())
    kit.hook_engine.FAIL_CLOSED_TRIGGERS = frozenset()

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="logger", priority=0)
    async def logger(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        raise RuntimeError("log sink down")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="approval", priority=1)
    async def approval(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("needs a supervisor's approval")

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results) and bool(calls.observed))

    assert json.loads(provider.tool_results[0][2]) == {"error": "needs a supervisor's approval"}
    assert [e.error_detail for e in calls.observed] == [None]
    await kit.close()


# -- A deliberate BLOCK's reason reaches the model on every door (RMK-306) ----

REASON = "needs a supervisor's approval"


def _block(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="approval")
    async def approval(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block(REASON)


async def test_the_ai_loop_reads_a_blocks_reason(streaming: bool) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "{}"

    kit, ch, room_id, _, _ = await _ai_room(streaming=streaming, tool_handler=handler)
    _block(kit)

    run = await _call_one_tool(kit, ch, room_id, "get_weather")

    assert json.loads(run.calls[0].result) == {"error": REASON}
    await kit.close()


async def test_a_recovered_spoken_call_reads_a_blocks_reason() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())
    _block(kit)

    await provider.simulate_transcription(session, "call:delete_account{id:42}", "assistant")
    await until(lambda: bool(provider.injected_texts))

    assert REASON in provider.injected_texts[0][1]
    await kit.close()


async def test_a_reasoning_backend_reads_a_blocks_reason() -> None:
    calls, backend = _Calls(), _Backend()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy(), backend=backend)
    _block(kit)

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(backend.results))

    assert json.loads(backend.results[0]) == {"error": REASON}
    await kit.close()


async def test_a_conference_reads_a_blocks_reason() -> None:
    calls = _Calls()
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=calls.conference
        ),
    )
    _block(kit)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert json.loads(provider.tool_results[0][2]) == {"error": REASON}
    await kit.close()


async def test_an_external_handler_reads_a_blocks_reason() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    handler = PolicyExternalToolHandler()
    kit._wire_external_tool_handler("agent", handler)
    _block(kit)

    decision = await handler.process_tool_call("delete_account", {"id": "42"}, room_id="r1")

    assert decision.reason == REASON
    await kit.close()


async def _before_events(kit: RoomKit) -> list[Any]:
    seen: list[Any] = []

    @kit.on("before_tool_use")
    async def on_event(event: Any) -> None:
        seen.append(event.data)

    return seen


async def test_a_realtime_call_emits_before_tool_use_with_no_hook() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=ToolPolicy())
    seen = await _before_events(kit)

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert [(e["tool_name"], e["allowed"]) for e in seen] == [("delete_account", True)]
    await kit.close()


async def test_a_conference_call_emits_before_tool_use_with_no_hook() -> None:
    calls = _Calls()
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=calls.conference
        ),
    )
    seen = await _before_events(kit)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert [(e["tool_name"], e["allowed"]) for e in seen] == [("delete_account", True)]
    await kit.close()
