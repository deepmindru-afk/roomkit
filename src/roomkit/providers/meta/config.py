"""Meta Model API provider configuration — images (Muse Image)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr

MetaImageTool = Literal["web_search", "image_search", "shell"]


class MetaImageConfig(BaseModel):
    """Meta Muse Image provider configuration (RFC §25).

    Attributes:
        api_key: Meta Model API key.
        base_url: Meta Model API endpoint. Override only to point at a proxy.
        model: Image model id — see :mod:`roomkit.providers.meta.image_models`.
        tools: What the image generator may do on its own while it draws:
            ``"web_search"`` (look up facts), ``"image_search"`` (fetch visual
            references), ``"shell"`` (run code for charts and layouts). Meta
            enables all three when a request says nothing, so RoomKit always
            says: none unless listed here. A prompt sent with a search tool
            leaves Meta for the web.
        reasoning_strength: ``"high"`` (Meta's default: several refinement
            passes) or ``"low"`` (one pass), billed the same per image.
            ``None`` leaves the service default.
        moderation: ``"auto"`` or ``"low"``, or ``None`` for the default.
        output_format: ``"png"``, ``"jpeg"`` or ``"webp"``; ``None`` leaves
            Meta's default, WebP.
        timeout: HTTP request timeout in seconds; a drawing takes ~10 s at
            ``reasoning_strength="low"`` (measured 2026-09-27).
        connect_timeout: TCP connect timeout in seconds.
        max_retries: SDK-level retry count. 0 because RoomKit's RetryPolicy
            handles retries at the right layer.
    """

    api_key: SecretStr
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-image-1.0"
    tools: list[MetaImageTool] = []
    reasoning_strength: Literal["low", "high"] | None = None
    moderation: Literal["auto", "low"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0
    max_retries: int = 0
