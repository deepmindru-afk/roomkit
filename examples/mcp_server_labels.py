"""RoomKit -- a tool policy that names one MCP server, with two servers composed.

Two MCP servers both serve ``delete_records``. A model that names MCP tools the
way prompts written for Claude Code do calls ``mcp__<server>__<tool>``. Each
provider is given ``label=``, the host's name for its server: its tool handler
serves only the aliases naming that server, so the alias reaches the server it
names whatever the order of the chain, and a policy that names a server holds.

A policy judges an alias under both its names (RFC §21.1). To keep a server
out, deny its aliases (``deny=["mcp__archive__*"]``); to admit one server only,
allow its aliases and its tools' names (``allow=["mcp__crm__*",
"delete_records"]``). A provider without ``label=`` serves an alias whatever
server it names: such a policy cannot hold a call to it.

    RealtimeVoiceChannel (tool_policy=deny mcp__archive__*)
        └── compose_tool_handlers
              ├── MCPToolProvider(label="archive") ── stdio ── records_server.py archive
              └── MCPToolProvider(label="crm")     ── stdio ── records_server.py crm

Runs on the mock realtime provider, so it needs no API key: the model's tool
calls are simulated.

Requirements:
    pip install roomkit[mcp]

Run:
    uv run python examples/mcp_server_labels.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit, ToolPolicy
from roomkit.tools import MCPToolProvider, compose_tool_handlers
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("mcp_server_labels")

RECORDS_SERVER = str(Path(__file__).resolve().parent / "mcp_servers" / "records_server.py")

CALLS = ["mcp__crm__delete_records", "mcp__archive__delete_records", "mcp__crm__search_records"]


def records(label: str) -> MCPToolProvider:
    """The records server for *label*, started as a subprocess and named by it."""
    return MCPToolProvider.from_command(sys.executable, [RECORDS_SERVER, label], label=label)


async def main() -> None:
    async with records("crm") as crm, records("archive") as archive:
        provider = MockRealtimeProvider()
        voice = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            # The archive first: without labels, every alias would reach it.
            tool_handler=compose_tool_handlers(archive.as_tool_handler(), crm.as_tool_handler()),
            tool_policy=ToolPolicy(deny=["mcp__archive__*"]),  # keep the archive out
        )
        kit = RoomKit()
        kit.register_channel(voice)
        await kit.create_room(room_id="records")
        await kit.attach_channel("records", "voice")
        session = await voice.start_session("records", "operator", "ws")

        for index, name in enumerate(CALLS, start=1):
            await provider.simulate_tool_call(session, f"call-{index}", name, {"query": "2019"})
            for _ in range(100):
                if len(provider.tool_results) == index:
                    break
                await asyncio.sleep(0.05)
            logger.info("%s -> %s", name, provider.tool_results[-1][2])
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
