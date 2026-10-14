"""How a media URL is read: a data URI, a link, or a reference (RFC §5, §6.4).

A ``MediaContent`` URL is a ``data:`` URI, which carries the media; a link
(``http`` or ``https``), which a reader fetches; or a reference: any other
scheme, which the application chooses and RoomKit never interprets, naming a
media the application holds (its ``media_resolver`` returns the bytes). A leaf
module: the router's transcoder and the AI channel both read it.
"""

from __future__ import annotations

from roomkit.models.event import MediaContent

_LINK_SCHEMES = ("http://", "https://")

VISION_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""The image types every vision provider takes (RFC §6.4): what RoomKit sends
as an image, a tool's replayed or an author's."""


def is_data_uri(url: str) -> bool:
    """Whether *url* carries its media inline."""
    return url[: len("data:")].lower() == "data:"


def is_link(url: str) -> bool:
    """Whether *url* is a link a reader fetches (``http`` or ``https``)."""
    return url[: len("https://")].lower().startswith(_LINK_SCHEMES)


def data_uri_media_type(url: str) -> str | None:
    """The media type a ``data:`` URI declares, lowercased; None for any other URL."""
    if not is_data_uri(url):
        return None
    return url[len("data:") :].split(",", 1)[0].split(";", 1)[0].lower()


def is_vision_image(content: MediaContent) -> bool:
    """Whether *content* is an image of a type every vision provider takes."""
    media_type = data_uri_media_type(content.url) or content.mime_type.lower()
    return media_type in VISION_IMAGE_TYPES


def is_reference(url: str) -> bool:
    """Whether *url* names a media the application holds and resolves."""
    return not is_data_uri(url) and not is_link(url)


def media_label(content: MediaContent) -> str:
    """How *content* reads as text (RFC §6.6): its caption, else its filename,
    else its URL when a link, else its media type; never a data URI or a
    reference, which mean nothing to a reader and may be long."""
    url = content.url if is_link(content.url) else None
    return f"[Media: {content.caption or content.filename or url or content.mime_type}]"
