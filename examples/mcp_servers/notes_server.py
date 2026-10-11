"""A tiny MCP server over stdio, for examples/mcp_stdio_tools.py, mcp_read_resource.py
and acp_room_mcp.py.

Keeps notes in memory for as long as it runs. Any MCP server started as a
command works the same way; this one only needs the ``mcp`` package. Started
with ``NOTES_ROOM`` in its environment, it files its notes under that room.

    python examples/mcp_servers/notes_server.py   # speaks MCP on stdin/stdout
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("notes", log_level="WARNING")  # its logs go to our stderr
_notes: list[str] = []
_room = os.environ.get("NOTES_ROOM", "")


@server.tool()
def add_note(text: str) -> str:
    """Save a note."""
    _notes.append(text)
    where = f" in room {_room}" if _room else ""
    return f"Saved note #{len(_notes)}{where}."


@server.tool()
def list_notes() -> list[str]:
    """Every note saved so far, oldest first."""
    return list(_notes)


@server.resource("notes://about", mime_type="text/plain")
def about() -> str:
    """What this server keeps, as a resource a host reads without calling a tool."""
    return "In-memory notes: add_note saves one, list_notes reads them back, oldest first."


if __name__ == "__main__":
    server.run()
