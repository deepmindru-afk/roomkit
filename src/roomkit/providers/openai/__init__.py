"""OpenAI provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider

if TYPE_CHECKING:
    from roomkit.providers.openai.live import OpenAILiveProvider
    from roomkit.providers.openai.live_config import HostedReasoning, IntegratorReasoning
    from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

__all__ = [
    "HostedReasoning",
    "IntegratorReasoning",
    "OpenAIAIProvider",
    "OpenAIConfig",
    "OpenAIImageConfig",
    "OpenAIImageProvider",
    "OpenAILiveProvider",
    "OpenAIRealtimeProvider",
]

__getattr__ = lazy_exports(
    __name__,
    {
        "HostedReasoning": "roomkit.providers.openai.live_config",
        "IntegratorReasoning": "roomkit.providers.openai.live_config",
        "OpenAILiveProvider": "roomkit.providers.openai.live",
        "OpenAIRealtimeProvider": "roomkit.providers.openai.realtime",
    },
)
