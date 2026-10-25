"""WhatsApp provider abstractions and mock implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.providers.whatsapp.base import WhatsAppProvider
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider

if TYPE_CHECKING:
    from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"WhatsAppPersonalProvider": "roomkit.providers.whatsapp.personal"}
    )

__all__ = [
    "MockWhatsAppProvider",
    "WhatsAppPersonalProvider",
    "WhatsAppProvider",
]
