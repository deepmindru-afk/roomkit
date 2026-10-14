"""How a media reads, for a model and for a text-only reader (RFC §6.4, §6.6).

The URL predicates live in :mod:`roomkit._media_url` (the content model's
validator reads them); this module adds what needs the content models: whether
a media is an image every vision provider takes, and the labels a text-only
reader (a channel without media, a model that does not see, a memory's
summary, an ACP agent's catch-up) reads in a media's place. A label never
shows a ``data:`` URI or a reference, which mean nothing to a reader and may be
long.
"""

from __future__ import annotations

from roomkit._media_url import (
    VISION_IMAGE_TYPES,
    data_uri_media_type,
    is_data_uri,
    is_link,
    is_reference,
    link_or_type,
    media_type,
)
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    EventContent,
    MediaContent,
    TextContent,
    VideoContent,
)

__all__ = [
    "VISION_IMAGE_TYPES",
    "audio_label",
    "content_text",
    "data_uri_media_type",
    "is_data_uri",
    "is_link",
    "is_reference",
    "is_vision_image",
    "media_label",
    "video_label",
]


def is_vision_image(content: MediaContent) -> bool:
    """Whether *content* is an image of a type every vision provider takes."""
    declared = data_uri_media_type(content.url) or media_type(content.mime_type)
    return declared in VISION_IMAGE_TYPES


def media_label(content: MediaContent) -> str:
    """How *content* reads as text (RFC §6.6): its caption, else its filename,
    else its URL when a link, else its media type."""
    name = content.caption or content.filename or link_or_type(content.url, content.mime_type)
    return f"[Media: {name}]"


def audio_label(content: AudioContent) -> str:
    """How *content* reads as text (RFC §6.6): its transcript, else its URL when a link."""
    if content.transcript:
        return content.transcript
    return f"[Voice message: {link_or_type(content.url, content.mime_type)}]"


def video_label(content: VideoContent) -> str:
    """How *content* reads as text (RFC §6.6): its URL when a link, else its type."""
    return f"[Video: {link_or_type(content.url, content.mime_type)}]"


def content_text(content: EventContent) -> str | None:
    """What a text-only reader reads of *content*: its text, a media as its
    label, a composite's parts joined; None for content of another kind."""
    if isinstance(content, TextContent):
        return content.body
    if isinstance(content, MediaContent):
        return media_label(content)
    if isinstance(content, CompositeContent):
        parts = (content_text(part) for part in content.parts)
        return "\n".join(part for part in parts if part)
    return None
