"""Google Gemini provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig, GeminiImageConfig
from roomkit.providers.gemini.image import GeminiImageProvider
from roomkit.providers.gemini.vertex import GeminiVertexConfig, GeminiVertexProvider

if TYPE_CHECKING:
    from roomkit.providers.gemini.realtime import GeminiLiveProvider
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"GeminiLiveProvider": "roomkit.providers.gemini.realtime"}
    )

__all__ = [
    "GeminiAIProvider",
    "GeminiConfig",
    "GeminiImageConfig",
    "GeminiImageProvider",
    "GeminiLiveProvider",
    "GeminiVertexConfig",
    "GeminiVertexProvider",
]
