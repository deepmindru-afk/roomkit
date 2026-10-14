"""How a media URL is read, before any model exists: a data URI, a link, or a
reference (RFC §5, §6.4).

A ``data:`` URI carries the media; a link (``http`` or ``https``) is fetched by
its reader; a reference, ``<scheme>://...`` in a scheme the application
chooses, names a media the application holds. No import from RoomKit: the
content model's own validator reads URLs with these.
"""

from __future__ import annotations

import re

_LINK_SCHEMES = ("http://", "https://")
_REFERENCE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]+)://([^\\]*)$")
# A scheme a reader could fetch or open itself, or run.
_REFUSED_SCHEMES = frozenset({"file", "ftp", "javascript", "vbscript"})
_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}

VISION_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""The image types every vision provider takes (RFC §6.4): what RoomKit sends
as an image, a tool's replayed or an author's."""


def is_data_uri(url: str) -> bool:
    """Whether *url* carries its media inline."""
    return url[: len("data:")].lower() == "data:"


def is_link(url: str) -> bool:
    """Whether *url* is a link a reader fetches (``http`` or ``https``)."""
    return url[: len("https://")].lower().startswith(_LINK_SCHEMES)


def is_reference(url: str) -> bool:
    """Whether *url* names a media the application holds and resolves."""
    return not is_data_uri(url) and not is_link(url)


def is_safe_reference(url: str) -> bool:
    """Whether *url* is a reference no reader could take for something else:
    two letters of scheme at least (one reads as a Windows drive), not a scheme
    a reader fetches, opens or runs, no ``..`` segment and no backslash a path
    reader could follow."""
    match = _REFERENCE.match(url)
    return (
        match is not None
        and match.group(1).lower() not in _REFUSED_SCHEMES
        and ".." not in match.group(2).split("/")
    )


def media_type(mime_type: str) -> str:
    """*mime_type* as RoomKit compares it: lowercased, parameters aside, aliases folded."""
    bare = mime_type.split(";", 1)[0].strip().lower()
    return _TYPE_ALIASES.get(bare, bare)


def data_uri_media_type(url: str) -> str | None:
    """The media type a ``data:`` URI declares, as :func:`media_type` reads it;
    None for any other URL."""
    if not is_data_uri(url):
        return None
    return media_type(url[len("data:") :].split(",", 1)[0])


def link_or_type(url: str, mime_type: str | None) -> str:
    """What a label shows of a media: its URL when a link, else its media type."""
    return url if is_link(url) else (mime_type or "media")
