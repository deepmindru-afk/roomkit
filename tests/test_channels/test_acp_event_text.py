"""``acp_event_text``: the text an ACP agent reads for an event (RFC A.9.1).

A host building an ACP prompt of its own reads an event as the channel does:
rich content as its plain-text rendering, where ``extract_event_text`` keeps
the markup.
"""

from __future__ import annotations

from roomkit import acp_event_text
from roomkit.channels.base import Channel
from roomkit.memory import extract_event_text
from roomkit.models.event import LocationContent, RichContent
from tests.conftest import make_event


def test_text_is_its_body() -> None:
    event = make_event(body="hello")

    assert acp_event_text(event) == "hello"


def test_rich_content_is_read_as_its_plain_text() -> None:
    rich = RichContent(body="**Invoice** ready", plain_text="Invoice ready")
    event = make_event().model_copy(update={"content": rich})

    assert acp_event_text(event) == "Invoice ready"
    assert extract_event_text(event) != "Invoice ready"


def test_rich_content_without_plain_text_falls_back_to_its_body() -> None:
    event = make_event().model_copy(update={"content": RichContent(body="**bold**")})

    assert acp_event_text(event) == "**bold**"


def test_other_content_carries_no_text_as_on_any_channel() -> None:
    location = LocationContent(latitude=45.5, longitude=-73.6, label="Montreal")
    event = make_event().model_copy(update={"content": location})

    assert acp_event_text(event) == Channel.extract_text(event) == ""
