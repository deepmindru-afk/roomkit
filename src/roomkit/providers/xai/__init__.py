"""xAI (Grok) provider — OpenAI-compatible chat, plus realtime speech-to-speech.

``XAIRealtimeProvider`` needs the ``websockets`` package, so its module loads
on first use of the name, not with this package. The chat provider only needs
the lazily-imported ``openai`` SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig, XAIImageConfig, XAIRealtimeConfig
from roomkit.providers.xai.image import XAIImageProvider

if TYPE_CHECKING:
    from roomkit.providers.xai.realtime import XAIRealtimeProvider
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"XAIRealtimeProvider": "roomkit.providers.xai.realtime"}
    )

__all__ = [
    "XAIAIProvider",
    "XAIConfig",
    "XAIImageConfig",
    "XAIImageProvider",
    "XAIRealtimeConfig",
    "XAIRealtimeProvider",
]
