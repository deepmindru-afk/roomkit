"""An answer of the agent cut off by a barge-in, as the AI channel uses it (RFC §6.4).

A voice channel records the cut (RFC §12.3.13 step 2): an internal event holding
the text handed to speech and how long it played, naming the answer it cut by
its channel and the event that answer responds to. The AI channel marks that
answer as interrupted in the context its next turn reads, quoting the words
the person heard of it, and hands a speak policy the cut when the agent has not
answered since.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from roomkit._text import named_blocks, quoted
from roomkit.core._changes import is_deleted
from roomkit.models.enums import ChannelType, Visibility
from roomkit.models.event import TextContent
from roomkit.speaking.base import CutReply

if TYPE_CHECKING:
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

CUT_MARK_OPENING = "[The person cut you off during this answer"
"""Opens the mark the context adds to an answer that was cut off; the rest of
the mark says when, and how far they heard it."""

CUT_NOTE = (
    "[Cut answers, from the runtime: when the person cuts you off, your answer "
    "stays whole in the conversation, followed by a mark that says how far they "
    "heard it. Only an answer followed by such a mark was cut; every answer of "
    "yours without one was heard to its end.]"
)
"""The turn's note while the history holds a cut answer: without it, a model
asked whether it was cut off answers for the last cut it reads, however many
whole answers followed it."""

_SPOKEN_CHARS_PER_S = 11
"""Letters a voice says per second, at its slowest, when the length of the
answer's audio is unknown. Voxtral Speech 3 said 12 to 15: slow on purpose, a
quote that ends early costs a resumed answer a few words said again, one that
ends late costs the words the person never heard."""

_MARGIN_WORDS = 2
"""Words taken off the estimate made from the audio's share: on Voxtral it was
within 3 words of what an STT heard, one way or the other."""

_QUOTE_CHARS = 400
"""The longest quote of what was heard: its end is kept, where the person
stopped hearing."""

_LETTERS = re.compile(r"\w")


def cut_records(context: RoomContext, channel_id: str) -> dict[str | None, RoomEvent]:
    """The latest cut record of each of *channel_id*'s answers in *context*, by
    the event the answer responds to.

    Only a record a voice channel of the room wrote counts: internal, with no
    participant, from a channel bound to the room as voice. Metadata is anyone's
    to write, and a record taken at its word would put a sender's text in the
    agent's mouth; a malformed one is skipped rather than fail every turn of the
    room it stays in.
    """
    voices = {b.channel_id for b in context.bindings if b.channel_type == ChannelType.VOICE}
    records: dict[str | None, RoomEvent] = {}
    for event in context.recent_events:
        # A deleted record quotes what was withheld (RFC §10.3).
        if _is_cut_record(event, channel_id, voices) and not is_deleted(event):
            records[event.metadata.get("answer_responds_to")] = event
    return records


def _is_cut_record(event: RoomEvent, channel_id: str, voices: set[str]) -> bool:
    metadata = event.metadata
    return (
        metadata.get("interrupted") is True
        and metadata.get("answer_channel_id") == channel_id
        and isinstance(metadata.get("answer_responds_to"), str | None)
        and event.visibility == Visibility.INTERNAL
        and event.source.participant_id is None
        and event.source.channel_type == ChannelType.VOICE
        and event.source.channel_id in voices
    )


def cut_answers(
    events: Iterable[RoomEvent], records: dict[str | None, RoomEvent], channel_id: str
) -> dict[str, RoomEvent]:
    """The cut record of each cut answer among *events*, by the id of the event
    that ends the answer: the last event of *channel_id*'s answer to each event
    *records* names."""
    last: dict[str | None, str] = {}
    for event in events:
        if event.source.channel_id == channel_id and event.responds_to in records:
            last[event.responds_to] = event.id
    # A deleted answer's record would quote the words its delete withheld.
    deleted = {event.id for event in events if is_deleted(event)}
    last = {
        responds_to: event_id for responds_to, event_id in last.items() if event_id not in deleted
    }
    return {event_id: records[responds_to] for responds_to, event_id in last.items()}


def cut_mark(record: RoomEvent, clean: Callable[[str], str]) -> str:
    """The mark that follows the answer *record* cut: when it was cut, and the
    words the person heard of it, *clean* having replaced any copy of a runtime
    mark in them.

    The words are the agent's, which a participant may have led it to say:
    they are quoted on one line between quotes they cannot close (RFC §6.4),
    the start of a long quote left out.
    """
    played_ms = _played_ms(record)
    when = f"{CUT_MARK_OPENING}, {played_ms / 1000:.1f} s in"
    heard = heard_part(_record_text(record), played_ms, _played_share(record))
    if not heard:
        return f"{when}: they heard little or none of it.]"
    quote = quoted(_ending(named_blocks(clean(heard)), _QUOTE_CHARS), _QUOTE_CHARS)
    return f"{when}: they heard it up to {quote} and not the rest.]"


def _ending(text: str, limit: int) -> str:
    """*text*'s end within *limit* characters, cut at a word, opening on "…"."""
    if len(text) <= limit:
        return text
    tail = text[len(text) - limit + 1 :]
    space = tail.find(" ")
    if 0 <= space < limit // 2:
        tail = tail[space + 1 :]
    return "…" + tail.lstrip()


def heard_part(text: str, played_ms: int, share: float | None) -> str:
    """The words of *text* the person surely heard when it played *played_ms*,
    *share* of its audio when that is known.

    No word is timed: the words are taken by their letters, at the audio's
    share less :data:`_MARGIN_WORDS`, or at a slow speaking rate when the
    audio's length is unknown, so the quote ends early rather than late.
    """
    tokens = text.split()
    sizes = [len(_LETTERS.findall(token)) for token in tokens]
    if share is None:
        budget = _SPOKEN_CHARS_PER_S * played_ms / 1000
        margin = 0
    else:
        budget = share * sum(sizes)
        margin = _MARGIN_WORDS
    said, taken = 0, 0
    for size in sizes:
        if said + size > budget + 1:
            break
        said += size
        taken += 1
    words = [i for i in range(taken) if sizes[i]]
    if len(words) <= margin:
        return ""
    end = words[len(words) - 1 - margin] if margin else taken - 1
    return " ".join(tokens[: end + 1])


def _record_text(record: RoomEvent) -> str:
    return record.content.body if isinstance(record.content, TextContent) else ""


def _played_share(record: RoomEvent) -> float | None:
    """The share of its audio the record says played, ``None`` when it does not
    say, or not as a percentage."""
    value = record.metadata.get("played_percentage")
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 100:
        return None
    return value / 100


def cut_reply(
    recent: Iterable[RoomEvent], records: dict[str | None, RoomEvent], channel_id: str
) -> CutReply | None:
    """The agent's latest answer in *recent*, when it was cut off."""
    latest = None
    for event in recent:
        if event.source.channel_id == channel_id:
            latest = event
    if latest is None or is_deleted(latest):
        return None
    record = records.get(latest.responds_to)
    if record is None or record.created_at < latest.created_at:
        return None
    return CutReply(text=_record_text(record), played_ms=_played_ms(record), at=record.created_at)


def _played_ms(record: RoomEvent) -> int:
    """The record's ``played_ms``, 0 when it is not a non-negative number."""
    value = record.metadata.get("played_ms")
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return 0
    return int(value)
