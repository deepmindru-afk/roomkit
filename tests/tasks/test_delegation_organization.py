"""A delegation's child room belongs to its parent room's organization (RFC
§17.2, §23.3 step 1, RMK-686): what acts in it reads its tenant from the room,
as in the parent, and a read scoped to that organization finds it."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import RoomKit, RoomNotFoundError
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTask
from roomkit.tools.context import current_tool_room

LOOKUP = AITool(
    name="lookup", description="Look a record up", parameters={"type": "object", "properties": {}}
)


def _tool_round() -> list[AIResponse]:
    return [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
        ),
        AIResponse(content="Found it.", finish_reason="stop"),
    ]


async def _delegate(kit: RoomKit, room_id: str, agent_id: str, *, wait: bool) -> DelegatedTask:
    task = await kit.delegate(room_id, agent_id, "look the record up", wait=wait)
    if not wait:
        await task.wait(timeout=5.0)
    return task


@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_tool_in_the_child_room_reads_the_parents_organization(wait: bool) -> None:
    tenants: list[str | None] = []

    async def lookup(_name: str, _arguments: dict[str, Any]) -> str:
        room = current_tool_room()
        tenants.append(room.organization_id if room is not None else "no room")
        return "record"

    kit = RoomKit()
    kit.register_channel(
        AIChannel(
            "worker",
            provider=MockAIProvider(ai_responses=_tool_round()),
            tool_handler=lookup,
            tools=[LOOKUP],
        )
    )
    await kit.create_room(room_id="parent", organization_id="acme")

    task = await _delegate(kit, "parent", "worker", wait=wait)

    assert tenants == ["acme"]
    assert (await kit.get_room(task.child_room_id)).organization_id == "acme"
    await kit.close()


async def test_a_read_scoped_to_the_parents_organization_finds_the_child_room() -> None:
    kit = RoomKit()
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["done"])))
    await kit.create_room(room_id="parent", organization_id="acme")
    await kit.create_room(room_id="elsewhere", organization_id="globex")

    task = await _delegate(kit, "parent", "worker", wait=True)

    child = await kit.get_room(task.child_room_id, organization_id="acme")
    assert child.metadata["parent_room_id"] == "parent"
    with pytest.raises(RoomNotFoundError):
        await kit.get_room(task.child_room_id, organization_id="globex")
    assert {r.id for r in await kit.store.find_rooms(organization_id="acme")} == {
        "parent",
        task.child_room_id,
    }
    assert {r.id for r in await kit.store.find_rooms(organization_id="globex")} == {"elsewhere"}
    await kit.close()


async def test_a_delegation_from_a_child_room_stays_in_the_organization() -> None:
    kit = RoomKit()
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["a", "b"])))
    await kit.create_room(room_id="parent", organization_id="acme")
    task = await _delegate(kit, "parent", "worker", wait=True)

    nested = await _delegate(kit, task.child_room_id, "worker", wait=True)

    grandchild = await kit.get_room(nested.child_room_id)
    assert grandchild.organization_id == "acme"
    assert grandchild.metadata["parent_room_id"] == task.child_room_id
    await kit.close()


async def test_an_unscoped_parent_has_an_unscoped_child_room() -> None:
    kit = RoomKit()
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["done"])))
    await kit.create_room(room_id="parent")

    task = await _delegate(kit, "parent", "worker", wait=True)

    assert (await kit.get_room(task.child_room_id)).organization_id is None
    await kit.close()
