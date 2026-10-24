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
from decimal import Decimal

from roomkit._media_url import (
    VISION_IMAGE_TYPES,
    canonical_data_uri,
    data_uri_media_type,
    is_data_uri,
    is_link,
    is_reference,
    link_or_type,
    lowercase_scheme,
    media_type,
)
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    LocationContent,
    MediaContent,
    RichContent,
    SystemContent,
    TemplateContent,
    TextContent,
    ToolCallContent,
    VideoContent,
)

__all__ = [
    "VISION_IMAGE_TYPES",
    "audio_label",
    "canonical_data_uri",
    "content_text",
    "content_words",
    "coordinate_text",
    "data_uri_media_type",
    "is_data_uri",
    "is_link",
    "is_reference",
    "is_vision_image",
    "location_label",
    "lowercase_scheme",
    "media_label",
    "text_form",
    "video_label",
    "vision_image_type",
]


def vision_image_type(content: MediaContent) -> str:
    """The type *content* is read and sent as: its ``data:`` URI's, else its
    own, folded (``image/jpg`` and ``IMAGE/JPEG`` are ``image/jpeg``)."""
    return data_uri_media_type(content.url) or media_type(content.mime_type)


def is_vision_image(content: MediaContent) -> bool:
    """Whether *content* is an image of a type every vision provider takes."""
    return vision_image_type(content) in VISION_IMAGE_TYPES


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


def location_label(content: LocationContent) -> str:
    """How *content* reads as text (RFC §6.6): its label or address with its
    coordinates, its coordinates alone with neither."""
    name = content.label or content.address
    point = f"({coordinate_text(content.latitude)}, {coordinate_text(content.longitude)})"
    return f"[Location: {name} {point}]" if name else f"[Location: {point}]"


def coordinate_text(value: float) -> str:
    """How a coordinate reads in a text form (RFC §6.6): the shortest decimal
    that reads back as *value*, with one decimal at least and never an
    exponent, so that every implementation writes it alike (``45.0``,
    ``0.00005``, never ``45`` nor ``5e-05``)."""
    text = format(Decimal(repr(value)), "f")
    return text if "." in text else f"{text}.0"


def text_form(content: EventContent) -> str:
    """The text *content* reads as (RFC §6.6), whatever its kind: what
    :func:`content_text` reads of a text and a media, a location, a template,
    an edit and a delete by their fallback, a system notice by its body, a
    tool call as ``[Tool call: <name>]``, and a composite as its parts' text
    forms in order, one per line, an empty one skipped. What an
    instruction's fingerprint is of (§10.1.1 step 6), so every
    implementation of the RFC reads it alike."""
    if isinstance(content, CompositeContent):
        return "\n".join(text for part in content.parts if (text := text_form(part)))
    if isinstance(content, LocationContent):
        return location_label(content)
    if isinstance(content, TemplateContent):
        return content.body or f"[Template: {content.template_id}]"
    if isinstance(content, SystemContent):
        return content.body
    if isinstance(content, EditContent):
        return f"Correction: {text_form(content.new_content)}"
    if isinstance(content, DeleteContent):
        return "[Message deleted]"
    if isinstance(content, ToolCallContent):
        return f"[Tool call: {content.tool_name}]"
    return content_text(content) or ""


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
