"""Offline metadata for Meta's image-generation model (Muse Image).

Hand-maintained list returned by ``MetaImageProvider.available_models``,
verified against the live Meta Model API on 2026-09-27: generation and JSON
edits with up to six references, ``n`` up to 2 tried (Meta documents 10), the
three output formats (WebP by default), ``size`` as an aspect ratio including
3:1, and ``"auto"``.

No ``pricing``: Meta bills a flat $0.01 per generated image, whatever the
tokens the response reports. :class:`~roomkit.providers.ai.base.ModelPricing`
states per-token rates, and a per-image charge restated per token would be a
wrong number rather than a missing one — as for the xAI and DALL·E entries.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY
from roomkit.providers.image.options import ImageCapabilities, ImageModelInfo

_MUSE_IMAGE = ImageCapabilities(
    options=["output_format", "moderation", "thinking_level", "search_types"],
    formats=["png", "jpeg", "webp"],
    thinking_levels=["minimal", "high"],
    search_types=["web_search", "image_search"],
    # Any "WIDTHxHEIGHT" is taken as an aspect ratio; the provider checks the
    # shape itself (see MetaImageProvider._size).
    sizes=["auto"],
    flexible_size=True,
    max_images=10,
    # Meta states no maximum for multi-image composition; six references were
    # verified. This bound only stops an obviously oversized request early.
    max_references=16,
    verified=date(2026, 9, 27),
)

MODELS: list[ModelInfo] = [
    ImageModelInfo(
        id="muse-image-1.0",
        display_name="Muse Image 1.0",
        supports_vision=True,
        capabilities=[IMAGE_GEN_CAPABILITY, "edit"],
        image=_MUSE_IMAGE,
    ),
]
