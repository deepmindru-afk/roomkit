"""The visibility grammar (RFC §7.3): its keywords, the normal form of a
value, and the channel ids a value can name."""

from __future__ import annotations

from roomkit.models.enums import Visibility

KEYWORDS = frozenset(
    {
        Visibility.ALL,
        Visibility.NONE,
        Visibility.INTERNAL,
        Visibility.TRANSPORT,
        Visibility.INTELLIGENCE,
    }
)
"""The keywords of the visibility grammar (RFC §7.3), lowercase as written."""


def parse_visibility(value: str) -> str:
    """*value* in its normal form (RFC §7.3): a keyword as written, or the
    listed channel ids, each once, sorted.

    Whitespace around the value and around each id is ignored. Raises
    :class:`ValueError` for an empty value, an empty item, or a keyword
    inside a list.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("a visibility may not be empty")
    if stripped in KEYWORDS:
        return stripped
    items = [item.strip() for item in stripped.split(",")]
    if not all(items):
        raise ValueError(f"visibility {value!r} has an empty item")
    keywords = [item for item in items if item in KEYWORDS]
    if keywords:
        raise ValueError(f"visibility {value!r} lists the keyword {keywords[0]!r}")
    return ",".join(sorted(set(items)))


def check_channel_id(channel_id: str) -> None:
    """Refuse a channel id a visibility could not name (RFC §7.3): a
    keyword, an id holding a comma, or one starting or ending with
    whitespace. Raises :class:`ValueError`."""
    if not channel_id or channel_id != channel_id.strip():
        raise ValueError(f"channel id {channel_id!r} is empty or has surrounding whitespace")
    if channel_id in KEYWORDS:
        raise ValueError(f"channel id {channel_id!r} is a visibility keyword")
    if "," in channel_id:
        raise ValueError(f"channel id {channel_id!r} holds a comma")
