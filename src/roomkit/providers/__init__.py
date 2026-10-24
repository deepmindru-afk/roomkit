"""Messaging and AI providers, one package per vendor or channel type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports

if TYPE_CHECKING:
    from roomkit.providers.url_safety import validate_public_url

__all__ = ["validate_public_url"]

__getattr__ = lazy_exports(__name__, {"validate_public_url": "roomkit.providers.url_safety"})
