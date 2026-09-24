"""MCPToolProvider — bridge MCP servers into RoomKit's AITool/ToolHandler system."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from roomkit.core.exceptions import ToolRefusedError
from roomkit.providers.ai.base import AITool

logger = logging.getLogger("roomkit.tools.mcp")

_DEFAULT_CALL_TIMEOUT = 30.0
"""Seconds a tool call waits: one default shared by :meth:`MCPToolProvider.call_tool`
and the tool handler, so the two cannot drift apart."""

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

# Upper bound for publishing a structured result on the tool-call context
# (serialized size). Tool-call events ride the room event pipeline — DB rows,
# WebSocket broadcasts, audit — so a pathological multi-megabyte payload must
# not tag along; every realistic widget payload is far below this.
_STRUCTURED_CONTENT_MAX_BYTES = 512 * 1024


def _publish_structured_content(result: Any) -> None:
    """Expose ``CallToolResult.structuredContent`` to the tool-call context.

    The ToolHandler contract flattens results to the LLM-facing string, which
    large-result eviction may later replace with a placeholder. UI surfaces
    (MCP Apps widgets) need the structured payload verbatim, so it travels
    out-of-band on the ToolCallContext when one is active.
    """
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        return
    from roomkit.tools.context import _current_tool_call

    ctx = _current_tool_call.get()
    if ctx is None:
        return
    try:
        if len(json.dumps(structured)) > _STRUCTURED_CONTENT_MAX_BYTES:
            logger.warning(
                "structuredContent dropped: exceeds %d bytes", _STRUCTURED_CONTENT_MAX_BYTES
            )
            return
    except (TypeError, ValueError):
        return
    ctx.structured_content = structured


class MCPToolProvider:
    """Discover and invoke tools from an MCP server.

    Three transports: ``streamable_http`` (default) and ``sse`` for a server
    reached by URL, and ``stdio`` for a server RoomKit starts as a subprocess
    and talks to over its stdin/stdout — the way most MCP servers are run.

    Usage::

        async with MCPToolProvider.from_url("http://localhost:8000/mcp") as mcp:
            tools = mcp.get_tools()          # list[AITool]
            handler = mcp.as_tool_handler()   # ToolHandler for AIChannel

        async with MCPToolProvider.from_command("uvx", ["mcp-server-time"]) as mcp:
            ...

    Enter and exit it in the same task (``async with``): the MCP SDK's
    transports hold anyio cancel scopes that must close where they opened.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if transport not in ("streamable_http", "sse", "stdio"):
            raise ValueError(f"Unsupported transport: {transport!r}")
        if transport == "stdio":
            if not command:
                raise ValueError("the stdio transport needs a command to start the server")
            if url or headers:
                raise ValueError("url and headers are for the HTTP transports, not stdio")
        else:
            if not url:
                raise ValueError(f"the {transport} transport needs a url")
            if command or args or env is not None or cwd is not None:
                raise ValueError(f"command, args, env and cwd are for stdio, not {transport}")
        self._url = url
        self._transport = transport
        self._tool_filter = tool_filter
        self._headers = headers or {}
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._tools: list[AITool] = []
        self._tool_set: set[str] = set()
        self._connected = False

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for the given URL.

        The provider is not connected until used as an async context manager.

        Args:
            url: MCP server URL.
            transport: ``"streamable_http"`` (default) or ``"sse"``.
            tool_filter: Optional predicate to include only matching tool names.
            headers: Optional HTTP headers sent with every request.

        Returns:
            An MCPToolProvider instance (not yet connected).
        """
        return cls(url, transport=transport, tool_filter=tool_filter, headers=headers)

    @classmethod
    def from_command(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        tool_filter: Callable[[str], bool] | None = None,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for a server started as a subprocess (stdio).

        Entering the provider starts ``command`` with ``args``; exiting stops it.

        Args:
            command: Executable of the MCP server (``"uvx"``, ``"npx"``, a path).
            args: Its arguments, passed as a list: no shell is involved.
            env: Extra environment variables for the server. The MCP SDK starts
                it with a minimal environment (``HOME``, ``PATH``, ``USER``…),
                not the whole of this process's: pass what the server needs,
                an API key say, here.
            cwd: Working directory of the server.
            tool_filter: Optional predicate to include only matching tool names.

        Returns:
            An MCPToolProvider instance (not yet connected).
        """
        return cls(
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            tool_filter=tool_filter,
        )

    @property
    def _target(self) -> str:
        """The server, as logs name it: the command alone, as its args may hold secrets."""
        if self._transport == "stdio":
            return self._command or ""
        return self._url or ""

    async def __aenter__(self) -> MCPToolProvider:
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError(
                "MCPToolProvider requires the 'mcp' package. "
                "Install it with: pip install roomkit[mcp]"
            ) from None
        if self._stack is not None:
            raise RuntimeError("MCPToolProvider is already connected; exit it before re-entering")

        # Everything entered goes on one stack, so a failure half-way (a server
        # that exits, an initialize that errors) still closes what was opened:
        # for stdio, that is the server process.
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            listed = await session.list_tools()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session
        self._tools = []
        self._tool_set = set()
        for tool in listed.tools:
            if self._tool_filter and not self._tool_filter(tool.name):
                continue
            # FastMCP serializes a tool's tags into `_meta["fastmcp"]["tags"]`;
            # surface them so Tool Search can match this tool cross-lingually.
            meta = getattr(tool, "meta", None)
            tags = meta.get("fastmcp", {}).get("tags", []) if isinstance(meta, dict) else []
            ai_tool = AITool(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if tool.inputSchema else {},
                tags=tags or [],
            )
            self._tools.append(ai_tool)
            self._tool_set.add(tool.name)

        self._connected = True
        logger.info(
            "Connected to MCP server %s (%s) — discovered %d tools",
            self._target,
            self._transport,
            len(self._tools),
        )
        return self

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        """Open this provider's MCP transport on *stack*; return its two streams."""
        if self._transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._command or "",
                args=self._args,
                env=self._env,
                cwd=self._cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if self._transport == "sse":
            from mcp.client.sse import sse_client

            client = sse_client(self._url or "", headers=self._headers)
            read, write = await stack.enter_async_context(client)
            return read, write
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        # The SDK leaves a client it is handed open: the stack closes it.
        http = await stack.enter_async_context(create_mcp_http_client(headers=self._headers))
        client = streamable_http_client(self._url or "", http_client=http)
        read, write, _session_id = await stack.enter_async_context(client)
        return read, write

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._connected = False
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.__aexit__(exc_type, exc_val, exc_tb)

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MCPToolProvider is not connected. Use 'async with' to connect first."
            )

    def get_tools(self) -> list[AITool]:
        """Return discovered tools as RoomKit AITool instances."""
        self._ensure_connected()
        return list(self._tools)

    def get_tools_as_dicts(self) -> list[dict[str, Any]]:
        """Return discovered tools as plain dicts (for binding metadata)."""
        self._ensure_connected()
        return [t.model_dump() for t in self._tools]

    @property
    def tool_names(self) -> list[str]:
        """Return the names of all discovered tools."""
        self._ensure_connected()
        return [t.name for t in self._tools]

    async def _invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
    ) -> tuple[str, bool]:
        """Call the tool and return ``(body, refused)``.

        The server's ``isError`` is the one place this outcome exists, and both
        entry points below need it: :meth:`call_tool` renders it into the error
        envelope its callers have always received, while the tool handler
        raises, because a tool loop cannot recognise a refusal in a body.
        """
        self._ensure_connected()
        result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout=timeout)

        if result.isError:
            parts = [getattr(c, "text", str(c)) for c in result.content]
            return " ".join(parts), True

        _publish_structured_content(result)

        # Extract text from content parts
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
            else:
                texts.append(str(content))

        if len(texts) == 1:
            return str(texts[0]), False
        return json.dumps(texts), False

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> str:
        """Call a tool on the MCP server and return the result as a string.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.
            timeout: Maximum seconds to wait for a response.

        Returns:
            Result string. Single TextContent → plain text; multi-part → JSON array;
            error results → ``{"error": "..."}``.

        The error envelope is this method's contract and does not change. A tool
        loop reads :meth:`as_tool_handler` instead, which raises
        :class:`~roomkit.core.exceptions.ToolRefusedError` so the outcome does
        not have to be recognised in the body.
        """
        body, refused = await self._invoke(name, arguments, timeout=timeout)
        return json.dumps({"error": body}) if refused else body

    def as_tool_handler(self, *, gate_discovery: bool = True) -> ToolHandler:
        """Return a ToolHandler suitable for ``AIChannel(tool_handler=...)``.

        Unknown tools (not from this MCP server) return
        ``{"error": "Unknown tool: <name>"}``, which allows composition
        via ``compose_tool_handlers``.

        ``gate_discovery=False`` forwards every name to the server instead. A
        gateway that routes by name prefix and authenticates the caller per
        call serves tools this connection never listed — a server whose
        ``tools/list`` answers only behind the caller's own credential, say —
        and a host with its own allow-list in front has already decided what
        the model may call. Such a handler produces no ``Unknown tool``
        envelope, so it sits last in a ``compose_tool_handlers`` chain:
        nothing after it would be reached.

        A tool the server *refused* raises
        :class:`~roomkit.core.exceptions.ToolRefusedError` either way: the tool
        loop marks the call failed and hands the server's message to the model
        unchanged.
        """
        self._ensure_connected()

        async def _handler(name: str, arguments: dict[str, Any]) -> str:
            lookup = name
            # Strip mcp__<server>__ prefix if present (e.g. from system prompt naming)
            if lookup.startswith("mcp__") and "__" in lookup[5:]:
                lookup = lookup.split("__", 2)[-1]
            if gate_discovery and lookup not in self._tool_set:
                return json.dumps({"error": f"Unknown tool: {name}"})
            body, refused = await self._invoke(lookup, arguments, timeout=_DEFAULT_CALL_TIMEOUT)
            if refused:
                # The server declined; say so instead of returning a body the
                # loop would have to recognise, and keep the server's words —
                # they are what the model is meant to read.
                raise ToolRefusedError(body)
            return body

        return _handler
