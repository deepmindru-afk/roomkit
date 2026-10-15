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

from collections.abc import Callable

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
    RichContent,
    TextContent,
    VideoContent,
)

__all__ = [
    "VISION_IMAGE_TYPES",
    "audio_label",
    "content_text",
    "content_words",
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
    """What a text-only reader reads of *content* (RFC §6.6): its text (rich
    content's plain text), each media as its label, a composite's parts
    joined; None for content of another kind."""
    return _read(content, _label)


def content_words(content: EventContent) -> str | None:
    """What the author wrote in *content* (RFC §6.6): its text, a media's
    caption, a voice message's transcript, never a file name or a type (a name
    read in ``nova.png`` would address an agent); None for content of another
    kind."""
    return _read(content, _words)


type _Media = MediaContent | AudioContent | VideoContent


def _read(content: EventContent, media: Callable[[_Media], str]) -> str | None:
    """*content* as text, each media read by *media*."""
    if isinstance(content, TextContent):
        return content.body
    if isinstance(content, RichContent):
        return content.plain_text or content.body
    if isinstance(content, MediaContent | AudioContent | VideoContent):
        return media(content)
    if isinstance(content, CompositeContent):
        parts = (_read(part, media) for part in content.parts)
        return "\n".join(part for part in parts if part)
    return None


def _label(content: _Media) -> str:
    if isinstance(content, AudioContent):
        return audio_label(content)
    if isinstance(content, VideoContent):
        return video_label(content)
    return media_label(content)


def _words(content: _Media) -> str:
    if isinstance(content, AudioContent):
        return content.transcript or ""
    if isinstance(content, MediaContent):
        return content.caption or ""
    return ""
