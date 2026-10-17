"""A strategy that takes an agent's turns runs on the room's owner, and each
generation it asks of the agent is a strategy turn through the turn backend
(RFC §10.6.1, §19.7.3)."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

PASS1 = "Formulate the task for the workers."


def _agent(name: str, *answers: str, **kwargs: Any) -> Agent:
    responses = [AIResponse(content=a) for a in answers] or [AIResponse(content=f"{name} here")]
    return Agent(name, provider=MockAIProvider(ai_responses=responses), role=name, **kwargs)


async def _room(kit: RoomKit, *agents: Agent) -> None:
    kit.register_channel(SimpleChannel("ops"))
    for agent in agents:
        kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")


async def test_a_supervisors_passes_run_on_the_owner_as_strategy_turns() -> None:
    boss = _agent("boss", "Look into it.", "Here is what we found.", system_prompt="You are Boss.")
    worker = _agent("worker", "worker result")
    kit = RoomKit()
    await _room(kit, boss, worker)
    await kit.install_strategy(
        "r1",
        Supervisor(
            supervisor=boss,
            workers=[worker],
            strategy="sequential",
            auto_delegate=True,
            refine_task=True,
            refine_instruction=PASS1,
        ),
    )
    answers: list[str] = []
    answer = boss.answer

    async def spy_answer(*args: Any) -> Any:
        task = asyncio.current_task()
        answers.append(task.get_name() if task is not None else "")
        return await answer(*args)

    boss.answer = spy_answer  # type: ignore[method-assign]

    await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="Why?")),
        room_id="r1",
    )

    # Two generations in the room (the task-formulation pass, the
    # presentation), each a strategy turn on an executor.
    assert len(answers) == 2
    assert all(name.startswith("roomkit-turn-") for name in answers)
    # The pass-1 instruction follows the prompt the turn resolved.
    prompts = [call.system_prompt or "" for call in boss.provider.calls]
    assert prompts[0].startswith("You are Boss.")
    assert PASS1 in prompts[0]
    assert PASS1 not in prompts[1]
    await kit.close()


async def test_an_agent_with_no_strategy_answers_as_itself() -> None:
    agent = _agent("solo", "hello")
    kit = RoomKit()
    await _room(kit, agent)
    await kit.attach_channel("r1", "solo", category=ChannelCategory.INTELLIGENCE)
    assert kit._turn_runners.get("r1", "solo") is None

    result = await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="hi")),
        room_id="r1",
    )

    assert result.response_events or result.event is not None
    await kit.close()
