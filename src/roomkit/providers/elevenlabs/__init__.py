"""ElevenLabs Conversational AI provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig

if TYPE_CHECKING:
    from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"ElevenLabsRealtimeProvider": "roomkit.providers.elevenlabs.realtime"}
    )

__all__ = ["ElevenLabsRealtimeConfig", "ElevenLabsRealtimeProvider"]
