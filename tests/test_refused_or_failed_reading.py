"""A reasoning backend and the human-input tool read a refused call apart
from a failed one (RMK-465, RFC §9.3, §12.4.1).

``ToolCallResult.refused`` tells a backend, among the errors its gate gave a
call, a refusal; the backend's loop reads it refused and any other error
failed. The human-input tool reads a rejection (a human, an
``ON_USER_INPUT_REQUIRED`` hook) as a refusal, a request nobody answered in
time as a failure, and any other error, the handler giving the request up
included, as the generic failure, its message withheld from the model.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    HumanInputRejectedError,
    ToolCallResult,
)
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError, UnservedToolCallError
from roomkit.models.pending_input import PendingInputEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import until
from tests.test_realtime_reasoning import TestAgentReasoningBackend as _AgentBackendCase
from tests.test_realtime_tool_executor import LOOKUP, REFUSED, _Backend, _channel

FAILING = {"name": "failing", "description": "Always fails", "parameters": {"type": "object"}}


async def _handler(name: str, arguments: dict[str, Any]) -> str:
    if name == "refused":
        raise ToolRefusedError("not for you")
    if name == "failing":
        raise ToolFailedError("the database is down")
    return '{"found": true}'


async def test_the_channel_tells_a_backend_a_refusal_from_a_failure() -> None:
    backend = _Backend("lookup", "refused", "failing", "missing")
    kit, provider, session = await _channel(
        _handler, reasoning_backend=backend, tools=[LOOKUP, REFUSED, FAILING]
    )

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: len(backend.results) == 4)
    await kit.close()

    assert [(r.is_error, r.refused) for r in backend.results] == [
        (False, False),
        (True, True),  # its handler refused it
        (True, False),  # it ran and failed
        (True, True),  # the gate refused it: never declared
    ]


def _tool(name: str) -> dict[str, Any]:
    return {"name": name, "description": name, "parameters": {"type": "object"}}


async def test_the_channel_tells_a_backend_how_the_gate_and_the_hooks_ended_a_call() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "nobody":
            raise UnservedToolCallError(name)
        return '{"found": true}'

    backend = _Backend("denied", "withheld", "nobody")
    tools = [_tool("denied"), _tool("withheld"), _tool("nobody")]
    kit, provider, session = await _channel(handler, reasoning_backend=backend, tools=tools)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
    async def deny(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("no") if event.name == "denied" else HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
    async def withhold(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("held") if event.name == "withheld" else HookResult.allow()

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: len(backend.results) == 3)
    await kit.close()

    assert [(r.is_error, r.refused) for r in backend.results] == [
        (True, True),  # BEFORE_TOOL_USE refused it before it ran
        (True, False),  # it ran, and ON_TOOL_CALL withheld its result
        (True, False),  # nothing served it
    ]


@pytest.mark.parametrize("fails", ["raises", "no-executor"])
async def test_a_backend_reads_its_own_executor_s_failure_as_failed(fails: str) -> None:
    async def raising(name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raise RuntimeError("the executor broke")

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
            ),
            AIResponse(content="Done."),
        ]
    )
    request = _AgentBackendCase()._request(raising)
    if fails == "no-executor":
        request = replace(request, execute_tool_call=None)
    backend = AgentReasoningBackend(Agent("reasoner", provider=provider))

    _ = [o async for o in backend.run(request)]

    [part] = provider.calls[1].messages[-1].content
    assert (part.outcome, part.is_error) == ("failed", True)


@pytest.mark.parametrize(
    ("done", "outcome"),
    [
        (
            ToolCallResult('{"error": "outside business hours"}', is_error=True, refused=True),
            "refused",
        ),
        (ToolCallResult('{"error": "the database is down"}', is_error=True), "failed"),
        (ToolCallResult('{"status": "on time"}'), "served"),
    ],
    ids=["refused", "failed", "served"],
)
async def test_a_backend_s_loop_reads_the_gate_s_outcome(
    done: ToolCallResult, outcome: str
) -> None:
    async def execute(name: str, arguments: dict[str, Any]) -> ToolCallResult:
        return done

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
            ),
            AIResponse(content="Done."),
        ]
    )
    backend = AgentReasoningBackend(Agent("reasoner", provider=provider))

    _ = [o async for o in backend.run(_AgentBackendCase()._request(execute))]

    [part] = provider.calls[1].messages[-1].content
    assert part.outcome == outcome
    assert part.is_error is (outcome != "served")
    assert part.result == done.text


def _asking(reply: Any) -> HumanInputToolHandler:
    """The human-input tool, its request answered by *reply* when it arrives."""
    human = HumanInputToolHandler(tool_names={"ask"}, timeout=0.05)

    async def arrived(event: PendingInputEvent) -> bool:
        reply(human.handler, event.pending_id)
        return True

    human.handler._on_input_required = arrived
    return human


async def test_a_human_s_rejection_is_a_refusal_with_their_reason() -> None:
    human = _asking(lambda handler, pending_id: handler.reject(pending_id, "not today"))

    with pytest.raises(ToolRefusedError) as refused:
        await human("ask", {})

    assert "not today" in refused.value.message
    assert isinstance(refused.value.__cause__, HumanInputRejectedError)


async def test_a_human_who_never_answers_is_a_failure() -> None:
    human = _asking(lambda handler, pending_id: None)

    with pytest.raises(ToolFailedError) as failed:
        await human("ask", {})

    assert "timed out" in failed.value.message


async def test_any_other_error_takes_the_generic_failure_path() -> None:
    """A closed handler is no human's answer: neither a refusal nor a failure
    in the tool's words, so the model reads only that the call failed."""
    handler = HumanInputHandler()
    await handler.close()
    human = HumanInputToolHandler(tool_names={"ask"}, handler=handler)

    with pytest.raises(RuntimeError) as error:
        await human("ask", {})

    assert not isinstance(error.value, (ToolRefusedError, ToolFailedError))


async def test_a_caller_catching_runtime_error_still_catches_a_rejection() -> None:
    handler = HumanInputHandler()
    pending = await handler.create("confirm", {}, channel_id="ch1")
    handler.reject(pending.pending_id, "denied by admin")

    with pytest.raises(RuntimeError, match="denied by admin"):
        await handler.wait(pending.pending_id, timeout=1)


async def test_a_hook_s_block_is_a_refusal() -> None:
    async def blocked(event: PendingInputEvent) -> bool:
        return False

    human = HumanInputToolHandler(tool_names={"ask"}, timeout=1)
    human.handler._on_input_required = blocked

    with pytest.raises(ToolRefusedError) as refused:
        await human("ask", {})

    assert "Denied by ON_USER_INPUT_REQUIRED hook" in refused.value.message


@pytest.mark.parametrize("give_up", ["close", "release"])
async def test_a_request_the_handler_gives_up_takes_the_generic_failure_path(
    give_up: str,
) -> None:
    """Closed, or released, while the call waits: nobody's answer, so neither a
    refusal nor a failure in the tool's words, as when it was closed before."""
    human = _asking(lambda handler, pending_id: None)
    human.timeout = 5

    async def give_up_later() -> None:
        await asyncio.sleep(0.01)
        if give_up == "close":
            await human.handler.close()
        else:
            human.handler.release(next(iter(human.handler.pending)))

    task = asyncio.create_task(give_up_later())
    with pytest.raises(RuntimeError) as error:
        await human("ask", {})
    await task

    assert not isinstance(error.value, (ToolRefusedError, ToolFailedError))
    assert not isinstance(error.value, HumanInputRejectedError)
