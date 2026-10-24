"""Claude Code with an MCP server per room, every session opened in one mode, over ACP.

One ``ACPChannel`` serves two rooms, ``support`` and ``billing``. What each
session is opened with is set up once, on the channel:

- ``mcp_servers`` is a function of the room id: each room's session gets its own
  notes server (``examples/mcp_servers/notes_server.py``), told its room
  through ``NOTES_ROOM`` in its environment, so a note saved from ``support``
  is filed under ``support``.
- ``session_config={"mode": "default"}`` is set on every session the channel
  opens, before it is prompted, a session opened again after a recovery or an
  agent restart included. In ``default`` Claude Code asks its client before
  it calls an MCP tool, so each call reaches RoomKit's gate (the policy, then
  BEFORE_TOOL_USE). Under the ``auto`` mode a user may have set in their own
  Claude Code settings, it would call the tool without asking anyone.

The run asks the agent, in each room, to save a note, and prints what the gate
saw, what the agent answered and the session's mode.

Requires:
    pip install "roomkit[acp,mcp]"
    Node.js 22+ with ``npx``, and Claude Code signed in once:
    npx -y @agentclientprotocol/claude-agent-acp@0.61.0 --cli auth login --claudeai

Run with:
    uv run python examples/acp_room_mcp.py
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from acp.schema import EnvVariable, McpServerStdio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ACPChannel,
    AIResponseEvent,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomKit,
    TextContent,
    ToolCallEvent,
    WebSocketChannel,
)
from roomkit.tools import PolicyExternalToolHandler, ToolPolicy

setup_logging("acp_room_mcp")
logging.getLogger("roomkit").setLevel(logging.WARNING)

CLAUDE_AGENT_ACP = "@agentclientprotocol/claude-agent-acp@0.61.0"
NOTES_SERVER = Path(__file__).resolve().parent / "mcp_servers" / "notes_server.py"
ROOMS = ("support", "billing")

NOTES_ONLY = ToolPolicy(allow=["mcp__notes__*", "add_note", "list_notes"])
"""The notes tools, under Claude Code's alias and under their own name (RFC §21.1)."""


def notes_server(room_id: str) -> list[McpServerStdio]:
    """The MCP servers a session of *room_id* gets: a notes server filed under it."""
    return [
        McpServerStdio(
            name="notes",
            command=sys.executable,
            args=[str(NOTES_SERVER)],
            env=[EnvVariable(name="NOTES_ROOM", value=room_id)],
        )
    ]


async def build(
    answers: asyncio.Queue[AIResponseEvent], workspace: str
) -> tuple[RoomKit, ACPChannel]:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("desk"))
    agent = ACPChannel(
        "claude-code",
        command=["npx", "--no-install", CLAUDE_AGENT_ACP],
        cwd=workspace,
        inherit_env=["HOME", "PATH", "USER", "XDG_CONFIG_HOME"],
        mcp_servers=notes_server,
        session_config={"mode": "default"},
        external_tool_handler=PolicyExternalToolHandler(policy=NOTES_ONLY),
    )
    kit.register_channel(agent)
    for room_id in ROOMS:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "desk")
        await kit.attach_channel(room_id, "claude-code", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="show-gate")
    async def show_gate(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        print(f"   [gate] room {event.room_id}: {event.name} {event.arguments}")
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def answered(event: AIResponseEvent, _ctx: RoomContext) -> None:
        await answers.put(event)

    return kit, agent


async def main() -> None:
    if shutil.which("npx") is None:
        raise SystemExit("npx is needed to run Claude Code over ACP (Node.js 22+)")
    answers: asyncio.Queue[AIResponseEvent] = asyncio.Queue()
    with tempfile.TemporaryDirectory(prefix="roomkit-acp-room-mcp-") as workspace:
        kit, agent = await build(answers, workspace)
        try:
            await ask_each_room(kit, agent, answers)
        finally:
            await kit.close()


async def ask_each_room(
    kit: RoomKit, agent: ACPChannel, answers: asyncio.Queue[AIResponseEvent]
) -> None:
    """Ask the agent to save a note in each room, and print what came back."""
    for room_id in ROOMS:
        request = f"Save a note saying 'call the customer back' for the {room_id} team."
        print(f"\n[{room_id}] you: {request}")
        await kit.process_inbound(
            InboundMessage(channel_id="desk", sender_id="you", content=TextContent(body=request)),
            room_id=room_id,
        )
        answer = await asyncio.wait_for(answers.get(), timeout=180)
        print(f"[{room_id}] Claude Code: {answer.response_content.strip()}")
        config = await kit.channel_call(agent.channel_id, room_id, "session_config")
        print(f"[{room_id}] session mode: {config.get('mode')}")


if __name__ == "__main__":
    asyncio.run(main())
