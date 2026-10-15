"""What an image an author sent becomes in a model's input (RFC §6.4).

A ``MediaContent`` an event brings (alone, or a part of a ``CompositeContent``)
reaches a model that sees as an image after its caption, with the words of the
event. Its URL is a data URI, a link handed to the provider as it is, or a
reference the application's :data:`MediaResolver` turns into bytes as the turn
is built: the store and every broadcast carry the reference alone, and nothing
is kept from one turn to the next.

Per turn the newest ``max_author_images`` images go, history and input counted
together, of a type and a size every vision provider takes. Anything else
takes the image's place as a text mention, never a failed turn: a media that
is no such image reads as its label (RFC §6.6), an image past the bound as an
earlier image not shown, a reference that does not resolve as an image
unavailable, an image a provider refused as refused. A request that carried
author images and failed on a provider error not retried marks them refused in
the room (:class:`RefusedImages`): one the provider refused would fail every
later turn while it stays in the window.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from roomkit._media import (
    canonical_data_uri,
    is_data_uri,
    is_reference,
    is_vision_image,
    media_label,
    vision_image_type,
)
from roomkit._text import quoted
from roomkit.models.event import CompositeContent, EventContent, MediaContent, RoomEvent
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.providers.utils import to_data_uri

logger = logging.getLogger("roomkit.channels.ai")

MediaResolver = Callable[[MediaContent, str], Awaitable[bytes | None]]
"""An application's resolver (RFC §6.4): ``(media, room_id)`` → the bytes of the
media a reference names, or None when it has none to give. Handed to the
framework (``RoomKit(media_resolver=...)``), it serves every AI channel
registered on it. Any participant can write a reference: answer only for a
media the room *room_id* may show, None for any other."""

DEFAULT_MAX_AUTHOR_IMAGES = 4
"""The newest author images a turn sends, history and input together."""

MAX_AUTHOR_IMAGE_BYTES = 5 * 1024 * 1024
"""The size of an image every vision provider takes: Anthropic's documented
per-image limit, read on the image's bytes (Claude Haiku 5.5 took 5.4 MiB, its
base64 7.2 MiB, on 2026-10-14)."""

ImageKey = tuple[str, int]
"""An author image: the id of the event that brought it, and its part's index."""

_EARLIER = "Earlier image not shown"
_UNAVAILABLE = "Image unavailable"
_REFUSED = "Image refused by the model's provider"
_NAME_CHARS = 120
_MAX_ROOMS = 100  # FIFO cap across rooms a shared channel serves


def media_of(content: EventContent) -> list[tuple[int, MediaContent]]:
    """The media *content* carries, each with its part's index (0 for a lone media)."""
    if isinstance(content, MediaContent):
        return [(0, content)]
    if isinstance(content, CompositeContent):
        return [
            (i, part) for i, part in enumerate(content.parts) if isinstance(part, MediaContent)
        ]
    return []


@dataclass(frozen=True)
class AuthorImages:
    """What each author image of one turn becomes: the image part it is sent as,
    or the mention that takes its place. A media it does not name reads as its
    label (RFC §6.6)."""

    decided: Mapping[ImageKey, AIImagePart | str] = field(default_factory=dict)

    @property
    def sent(self) -> frozenset[ImageKey]:
        """The images the turn sends as images."""
        return frozenset(
            key for key, part in self.decided.items() if isinstance(part, AIImagePart)
        )

    def parts(
        self, event_id: str, index: int, media: MediaContent
    ) -> list[AITextPart | AIImagePart]:
        """The parts *media* reads as: its caption then its image, else its mention."""
        decided = self.decided.get((event_id, index))
        if isinstance(decided, AIImagePart):
            caption = [AITextPart(text=media.caption)] if media.caption else []
            return [*caption, decided]
        return [AITextPart(text=decided if decided is not None else media_label(media))]


async def author_images(
    events: Sequence[RoomEvent],
    *,
    room_id: str,
    resolver: MediaResolver | None,
    limit: int | None,
    refused: frozenset[ImageKey],
) -> AuthorImages:
    """Decide what each image *events* carry becomes, the newest first, and
    resolve the references among those sent."""
    decided, references = _decide(events, limit=limit, refused=refused)
    resolved = await asyncio.gather(
        *(_resolved(media, room_id, resolver) for media in references.values())
    )
    decided.update(zip(references, resolved, strict=True))
    return AuthorImages(decided)


def _decide(
    events: Sequence[RoomEvent], *, limit: int | None, refused: frozenset[ImageKey]
) -> tuple[dict[ImageKey, AIImagePart | str], dict[ImageKey, MediaContent]]:
    """The image or mention each image of *events* gets, and the references
    still to resolve among the images kept: the newest *limit* of them."""
    decided: dict[ImageKey, AIImagePart | str] = {}
    references: dict[ImageKey, MediaContent] = {}
    kept = 0
    for event in reversed(events):
        for index, media in reversed(media_of(event.content)):
            key = (event.id, index)
            if key in decided or key in references or not is_vision_image(media):
                continue
            if key in refused:
                decided[key] = _mention(_REFUSED, media)
            elif limit is not None and kept >= limit:
                decided[key] = _mention(_EARLIER, media)
            elif is_reference(media.url):
                kept += 1
                references[key] = media
            else:
                kept += 1
                decided[key] = _inline(media)
    return decided, references


def _inline(media: MediaContent) -> AIImagePart | str:
    """A data URI or a link as the image part sent, its type folded as it was
    read; a data URI with no data or past the size, a mention."""
    kind = vision_image_type(media)
    if not is_data_uri(media.url):
        return AIImagePart(url=media.url, mime_type=kind)
    if not 0 < _data_uri_bytes(media.url) <= MAX_AUTHOR_IMAGE_BYTES:
        return _mention(_UNAVAILABLE, media)
    return AIImagePart(url=canonical_data_uri(media.url), mime_type=kind)


def _data_uri_bytes(url: str) -> int:
    """The decoded size of a base64 ``data:`` URI, read off its length."""
    return len(url.split(",", 1)[-1]) * 3 // 4


async def _resolved(
    media: MediaContent, room_id: str, resolver: MediaResolver | None
) -> AIImagePart | str:
    """The image a reference names, as the application's resolver gives it,
    or the mention that takes its place when it gives none to send."""
    if resolver is None:
        return _mention(_UNAVAILABLE, media)
    try:
        data = await resolver(media, room_id)
    except Exception:
        logger.warning("The media resolver failed on an image in room %s", room_id, exc_info=True)
        return _mention(_UNAVAILABLE, media)
    if not data or len(data) > MAX_AUTHOR_IMAGE_BYTES:
        return _mention(_UNAVAILABLE, media)
    kind = vision_image_type(media)
    return AIImagePart(url=to_data_uri(data, kind), mime_type=kind)


def _mention(kind: str, media: MediaContent) -> str:
    """The text that takes an image's place, naming it as its author did."""
    name = media.caption or media.filename or media.mime_type
    return f"[{kind}: {quoted(name, _NAME_CHARS)}]"


class RefusedImages:
    """The author images a provider refused, per room: later turns send a
    mention in their place."""

    def __init__(self) -> None:
        self._by_room: OrderedDict[str, frozenset[ImageKey]] = OrderedDict()

    def of(self, room_id: str) -> frozenset[ImageKey]:
        """The images refused in *room_id*."""
        return self._by_room.get(room_id, frozenset())

    def add(self, room_id: str, keys: Iterable[ImageKey]) -> None:
        """Record *keys* as refused in *room_id*."""
        self._by_room[room_id] = self.of(room_id) | frozenset(keys)
        self._by_room.move_to_end(room_id)
        while len(self._by_room) > _MAX_ROOMS:
            self._by_room.popitem(last=False)
