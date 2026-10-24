"""Event-driven message sources for RoomKit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.sources.base import (
    BaseSourceProvider,
    EmitCallback,
    SourceHealth,
    SourceProvider,
    SourceStatus,
)

if TYPE_CHECKING:
    from roomkit.sources.buzz import (
        KIND_HUDDLE_STARTED,
        BuzzRelaySource,
        huddle_announcement_parser,
        parse_buzz_event,
    )
    from roomkit.sources.discord import DiscordGatewaySource, parse_discord_message
    from roomkit.sources.neonize import WhatsAppPersonalSourceProvider
    from roomkit.sources.sse import SSESource
    from roomkit.sources.websocket import WebSocketSource

__all__ = [
    "BaseSourceProvider",
    "EmitCallback",
    "SourceHealth",
    "SourceProvider",
    "SourceStatus",
    # Lazy imports for optional sources
    "WebSocketSource",
    "SSESource",
    "WhatsAppPersonalSourceProvider",
    "DiscordGatewaySource",
    "BuzzRelaySource",
    # The parsers a source takes as ``parser=``
    "KIND_HUDDLE_STARTED",
    "huddle_announcement_parser",
    "parse_buzz_event",
    "parse_discord_message",
]

__getattr__ = lazy_exports(
    __name__,
    {
        "BuzzRelaySource": "roomkit.sources.buzz",
        "DiscordGatewaySource": "roomkit.sources.discord",
        "KIND_HUDDLE_STARTED": "roomkit.sources.buzz",
        "SSESource": "roomkit.sources.sse",
        "WebSocketSource": "roomkit.sources.websocket",
        "WhatsAppPersonalSourceProvider": "roomkit.sources.neonize",
        "huddle_announcement_parser": "roomkit.sources.buzz",
        "parse_buzz_event": "roomkit.sources.buzz",
        "parse_discord_message": "roomkit.sources.discord",
    },
)
