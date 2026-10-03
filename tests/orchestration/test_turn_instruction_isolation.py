"""A strategy's per-turn instruction stays in its turn (RMK-310, RFC §19.7.3).

The supervisor's task-formulation pass rides a copy of the binding for that
call; the supervisor, shared by every room it serves, is never the carrier.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.agent import Agent
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.orchestration.strategies.supervisor.delegate import _formulate_task
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

PASS1 = "Formulate the task for the workers."


def _binding(room_id: str, **metadata: Any) -> ChannelBinding:
    return ChannelBinding(
        channel_id="sup",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata,
    )


async def test_two_rooms_in_parallel_never_share_the_instruction() -> None:
    supervisor = Agent("sup", provider=MockAIProvider(), system_prompt="You are Sup.")
    seen: dict[str, str] = {}
    both_started = asyncio.Event()
    started: list[str] = []

    async def on_event(event: Any, binding: ChannelBinding, context: RoomContext) -> ChannelOutput:
        seen[binding.room_id] = binding.metadata["system_prompt"]
        started.append(binding.room_id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return ChannelOutput.empty()

    await asyncio.gather(
        *(
            _formulate_task(
                RoomKit(),
                room,
                supervisor,
                on_event,
                make_event(room_id=room, body="go", channel_id="sms1"),
                _binding(room),
                RoomContext(room=Room(id=room)),
                PASS1,
            )
            for room in ("room-a", "room-b")
        )
    )

    assert supervisor.system_prompt == "You are Sup."
    assert seen == {room: f"You are Sup.\n\n{PASS1}" for room in ("room-a", "room-b")}


async def test_the_instruction_follows_the_prompt_the_turn_would_have_had() -> None:
    async def config(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(system_prompt="Per-room persona.")

    supervisor = Agent(
        "sup", provider=MockAIProvider(), system_prompt="Default.", config_provider=config
    )
    seen: list[str] = []

    async def on_event(event: Any, binding: ChannelBinding, context: RoomContext) -> ChannelOutput:
        seen.append(binding.metadata["system_prompt"])
        return ChannelOutput.empty()

    await _formulate_task(
        RoomKit(),
        "r1",
        supervisor,
        on_event,
        make_event(room_id="r1", body="go", channel_id="sms1"),
        _binding("r1"),
        RoomContext(room=Room(id="r1")),
        PASS1,
    )

    assert seen == [f"Per-room persona.\n\n{PASS1}"]
