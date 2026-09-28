"""RoomKit — a tool policy on a realtime voice agent, per participant role.

A ``RealtimeVoiceChannel`` takes ``tool_policy=`` like an ``AIChannel``: a tool
the policy denies is never declared to the session, and a call that names it
anyway is refused before the handler and audited. Role overrides apply to the
session's participant.

Two callers reach the same voice agent here: a member, and an observer whose
role may read accounts but not close them. Each session prints the tools it was
declared, then the observer's model tries the denied tool anyway. It runs
against a mock provider: no API key, no audio device.

Run with:
    uv run python examples/realtime_tool_policy.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import (
    HookExecution,
    HookTrigger,
    RealtimeVoiceChannel,
    RoleOverride,
    RoomKit,
    ToolCallEvent,
    ToolPolicy,
)
from roomkit.models.participant import Participant
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("realtime_tool_policy")


def _tool(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string"}},
            "required": ["account"],
        },
    }


TOOLS = [
    _tool("lookup_account", "Read an account's balance and status"),
    _tool("close_account", "Close an account for good"),
]

# Everyone may use every tool, but an observer may not close anything.
POLICY = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["close_*"])})

closed: list[str] = []


async def accounts(name: str, arguments: dict[str, Any]) -> str:
    if name == "close_account":
        closed.append(arguments["account"])
        return json.dumps({"status": "closed", **arguments})
    return json.dumps({"account": arguments["account"], "balance": 120})


async def main() -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=accounts,
        tool_policy=POLICY,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    for participant_id, role in (("alice", "member"), ("bob", "observer")):
        await kit.store.add_participant(
            Participant(id=participant_id, room_id=room.id, channel_id="voice", role=role)
        )

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        outcome = "refused" if event.is_error else "served"
        print(f"  audit         : {event.name} {outcome}")

    for participant_id in ("alice", "bob"):
        session = await channel.start_session(room.id, participant_id, connection=None)
        connect = [c.args for c in provider.calls if c.method == "connect"][-1]
        print(f"\n{participant_id}'s session declares: {[t['name'] for t in connect['tools']]}")

    # The observer's model names the denied tool anyway.
    print("\nbob's model calls close_account")
    await provider.simulate_tool_call(session, "call-1", "close_account", {"account": "A-1"})
    await asyncio.sleep(0.1)
    print(f"  tool result   : {provider.tool_results[-1][2]}")
    print(f"  accounts closed: {closed}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
