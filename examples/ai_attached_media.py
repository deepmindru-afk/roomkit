"""Files an application keeps beside a message's words, seen by the model.

An application that stores a message as its words, with the files the person
attached in a record of its own (the event's metadata here), does not have to
reshape every message into a composite for a model to see its images. It hands
the framework an ``attached_media`` reader: ``(event) -> media``, the media its
record names, the same ones in the same order each time it is asked. Every AI
channel reads the events of a turn through it, the history and the event it
answers alike, with those media after the event's own content; a reference
among them is resolved by the ``media_resolver``, under the rules of any image
an author sent (the newest four, a mention for anything else). The store and
every broadcast keep the event as stored: its words, and its metadata.

This example sends a question with a colour swatch recorded in the message's
metadata, then a follow-up that needs the image again, and shows what the
model received and what the store kept.

Run with:
    uv run python examples/ai_attached_media.py              # offline, mock
    ANTHROPIC_API_KEY=... uv run python examples/ai_attached_media.py anthropic
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import struct
import zlib

from shared import require_env, setup_logging

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.event import MediaContent, RoomEvent
from roomkit.providers.ai import AIImagePart, AIProvider, MockAIProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

logger = setup_logging("ai_attached_media")


def red_square_png(size: int = 32) -> bytes:
    """A small solid red PNG, built without an imaging library."""
    row = b"\x00" + b"\xd0\x20\x20" * size  # filter byte, then RGB pixels
    pixels = zlib.compress(row * size)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


class FileStore:
    """The application's file store: each uploaded file belongs to one room."""

    def __init__(self) -> None:
        self._files: dict[str, tuple[str, bytes]] = {}

    def upload(self, room_id: str, file_id: str, data: bytes) -> str:
        self._files[file_id] = (room_id, data)
        return file_id

    async def resolve(self, media: MediaContent, room_id: str) -> bytes | None:
        """The resolver handed to the framework: bytes for this room's files only."""
        owner, data = self._files.get(media.url.removeprefix("files://"), ("", b""))
        return data if owner == room_id else None


def attached_files(event: RoomEvent) -> list[MediaContent]:
    """The reader handed to the framework: the files the application recorded."""
    return [
        MediaContent(url=f"files://{f['id']}", mime_type=f["mime_type"], filename=f["name"])
        for f in event.metadata.get("files", [])
    ]


def provider_for(name: str) -> AIProvider:
    if name == "anthropic":
        env = require_env("ANTHROPIC_API_KEY")
        return AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-haiku-5-5")
        )
    return MockAIProvider(responses=["It is red.", "Square."], vision=True)


async def build_kit(files: FileStore, provider: AIProvider) -> RoomKit:
    """A room where a person talks to an assistant that reads the files beside words."""
    kit = RoomKit(media_resolver=files.resolve, attached_media=attached_files)
    kit.register_channel(WebSocketChannel("user"))
    kit.register_channel(AIChannel("assistant", provider=provider))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "user")
    await kit.attach_channel("room", "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit


async def converse(kit: RoomKit, file_id: str) -> None:
    """A question with the swatch recorded beside it, then a follow-up that needs it again."""
    swatch = {"id": file_id, "name": "swatch.png", "mime_type": "image/png"}
    turns = [
        ("What colour is the swatch I attached?", {"files": [swatch]}),
        ("And what shape is it? One word.", {}),
    ]
    for asked, metadata in turns:
        message = InboundMessage(
            channel_id="user",
            sender_id="person",
            content=TextContent(body=asked),
            metadata=metadata,
        )
        result = await kit.process_inbound(message, room_id="room")
        answer = " ".join(
            e.content.body for e in result.response_events if isinstance(e.content, TextContent)
        )
        logger.info("Asked: %s", asked)
        logger.info("Answered: %s", answer)


def report_images_sent(provider: MockAIProvider) -> None:
    """What the mock model received on each turn."""
    for turn, context in enumerate(provider.calls, 1):
        images = [
            part.url[:30] + "..."
            for message in context.messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIImagePart)
        ]
        logger.info("Turn %d: the model received %d image(s): %s", turn, len(images), images)


async def report_store(kit: RoomKit) -> None:
    """What the room's store kept of the person's messages: words and metadata, unchanged."""
    for event in await kit.store.list_events("room"):
        if event.source.channel_id != "user":
            continue
        logger.info(
            "Stored: %s %r, files=%s",
            type(event.content).__name__,
            getattr(event.content, "body", ""),
            [f["name"] for f in event.metadata.get("files", [])],
        )


async def main(provider_name: str) -> None:
    files = FileStore()
    provider = provider_for(provider_name)
    kit = await build_kit(files, provider)
    try:
        file_id = files.upload("room", "swatch-42", red_square_png())
        await converse(kit, file_id)
        if isinstance(provider, MockAIProvider):
            report_images_sent(provider)
        await report_store(kit)
    finally:
        await kit.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", nargs="?", default="mock", choices=["mock", "anthropic"])
    asyncio.run(main(parser.parse_args().provider))
