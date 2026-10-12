"""What a registry needs of an MCP server to serve its skills (RFC §21.2, §24.7)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class MCPSkillServer(Protocol):
    """The three things a registry asks of a connected MCP server.

    :class:`~roomkit.tools.mcp.MCPToolProvider` is one. Nothing here relies on
    a session: a skill's identity, the entry it was loaded from and the content
    verified are the registry's, so a transport without a session serves them
    alike.
    """

    @property
    def server_extensions(self) -> Mapping[str, Mapping[str, Any]]:
        """The extensions the server declared when it connected, by identifier."""
        ...

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request the provider does not model; its result as the server sent it."""
        ...

    async def read_resource(self, uri: str) -> Any:
        """The server's ``resources/read`` result for *uri*."""
        ...
