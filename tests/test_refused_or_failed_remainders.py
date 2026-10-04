"""The line between a refused and a failed call, on the two doors RMK-459 left
(RMK-465, RFC §9.3, §12.4.1).

A reasoning backend's loop read every error the voice channel's gate gave a
call as a refusal: ``ToolCallResult`` said only ``is_error``. It now carries
``refused``, which the channel fills from the call's outcome, and the loop
reads a refusal as refused and any other error as failed. The human-input
tool read every ``RuntimeError`` as a refusal and a timeout as one too: only
a rejection (``HumanInputRejectedError``) is a refusal now, a timeout is a
failure, and any other error takes the generic failure path, its message
withheld from the model.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HumanInputRejectedError, ToolCallResult
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError
from roomkit.models.pending_input import PendingInputEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import until
from tests.test_realtime_reasoning import TestAgentReasoningBackend
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

    _ = [o async for o in backend.run(TestAgentReasoningBackend()._request(execute))]

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
    await asyncio.sleep(0)
