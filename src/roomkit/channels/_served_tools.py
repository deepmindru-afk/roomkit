"""What a host tool becomes under a name the channel serves itself (RFC §21.1).

A channel serves some tools itself (skill activation, Tool Search, the eviction
re-read, sandbox commands). A tool of the host under one of those names would
be declared with the host's schema and served by the channel, so it is refused
when given at construction and not declared when it arrives later. And no name
is declared twice: a provider rejects a duplicate name.

Shared by every channel kind whatever the shape of its tool definitions (an
``AITool``, a realtime tool dict).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Container, Iterable

from roomkit.channels._tool_search_constants import TOOL_SEARCH_INFRA_TOOL_NAMES

logger = logging.getLogger("roomkit.channels.tools")


def refuse_served_names(
    names: Iterable[str | None], served: Container[str], channel_id: str
) -> None:
    """Refuse a host tool given at construction under a name the channel serves."""
    for name in names:
        if name is None or name not in served:
            continue
        hint = " or pass tool_search=False" if name in TOOL_SEARCH_INFRA_TOOL_NAMES else ""
        raise ValueError(
            f"Tool {name!r} is a tool channel {channel_id!r} serves itself: "
            f"rename it{hint} (RFC §21.1)"
        )


class CollisionLog:
    """The collisions a channel reported, each once: a wiring diagnostic, not a turn event."""

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._reported: set[tuple[str, str]] = set()

    def served(self, name: str) -> None:
        self._once(
            name, "served", "Channel %s does not declare the host's %r: it serves it itself"
        )

    def duplicate(self, name: str) -> None:
        self._once(
            name, "duplicate", "Channel %s declares %r once: it was given twice, the later is kept"
        )

    def _once(self, name: str, kind: str, message: str) -> None:
        if (name, kind) in self._reported:
            return
        self._reported.add((name, kind))
        logger.warning(message, self._channel_id, name)


def declared_once[T](
    tools: Iterable[T],
    name_of: Callable[[T], str | None],
    served: Container[str],
    log: CollisionLog,
) -> list[T]:
    """*tools* with none under a served name, and each name once, the later kept.

    The later definition is the one of whoever serves the call (a channel adds
    orchestration's tools after the host's). A tool without a name is kept as
    it is: nothing can collide with it.
    """
    kept: dict[object, T] = {}
    for index, tool in enumerate(tools):
        name = name_of(tool)
        if name is None:
            kept[("unnamed", index)] = tool
            continue
        if name in served:
            log.served(name)
            continue
        if name in kept:
            log.duplicate(name)
            del kept[name]
        kept[name] = tool
    return list(kept.values())


def dict_tool_name(tool: object) -> str | None:
    """The name of a tool given as a dict (a realtime declaration), if it has one."""
    name = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) else None
