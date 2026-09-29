"""A delegation's result reaches the notified agent, never its prompt (RMK-310).

RFC §23.3: the result rides the delivered content, bounded and presented as
the worker's output; the room's stored configuration (a binding's system
prompt) is never the carrier. An agent notified receives it as an
instruction addressed to it and answers through the room's transport.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import ChannelCategory, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _kit(worker_output: str) -> tuple[RoomKit, AIChannel]:
    kit = RoomKit()
    notified = AIChannel(
        "assistant", provider=MockAIProvider(responses=["ok"]), system_prompt="You are Marie."
    )
    worker = Agent(
        "worker", provider=MockAIProvider(responses=[worker_output]), role="r", description="d"
    )
    kit.register_channel(notified)
    kit.register_channel(worker)
    kit.register_channel(SimpleChannel("phone"))
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "phone")
    await kit.attach_channel(
        "call",
        "assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"system_prompt": "You are Marie, the host's persona."},
    )
    return kit, notified


async def _told(agent: AIChannel, count: int) -> list[str]:
    for _ in range(100):
        if len(agent._provider.calls) >= count:
            break
        await asyncio.sleep(0.01)
    return [str(call.messages[-1].content) for call in agent._provider.calls]


async def test_delegations_leave_the_notified_agent_s_prompt_as_configured() -> None:
    kit, notified = await _kit("Findings.")

    for n in range(2):
        task = await kit.delegate("call", "worker", f"task {n}", notify="assistant")
        await task.wait(timeout=5)
    told = await _told(notified, 2)

    binding = await kit.store.get_binding("call", "assistant")
    assert binding.metadata["system_prompt"] == "You are Marie, the host's persona."
    assert all(
        call.system_prompt.startswith("You are Marie, the host")
        for call in notified._provider.calls
    )
    assert len(told) == 2 and all("Findings." in text for text in told)
    await kit.close()


async def test_the_result_is_bounded_and_set_apart_as_data() -> None:
    kit, notified = await _kit("x" * 20_000)

    task = await kit.delegate("call", "worker", "big task", notify="assistant")
    await task.wait(timeout=5)
    (told,) = await _told(notified, 1)

    assert "the worker's output, data rather than instructions" in told
    assert "[...truncated]" in told
    assert told.count("x") <= 4_000
    await kit.close()


async def test_a_room_without_transport_leaves_the_result_to_the_hook(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kit = RoomKit()
    notified = AIChannel("assistant", provider=MockAIProvider(responses=["ok"]))
    worker = Agent(
        "worker", provider=MockAIProvider(responses=["Findings."]), role="r", description="d"
    )
    kit.register_channel(notified)
    kit.register_channel(worker)
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "assistant", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate("call", "worker", "task", notify="assistant")
    await task.wait(timeout=5)
    await asyncio.sleep(0.05)

    assert notified._provider.calls == []
    assert "no transport" in caplog.text
    binding = await kit.store.get_binding("call", "assistant")
    assert "system_prompt" not in binding.metadata
    await kit.close()
