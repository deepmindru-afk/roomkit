"""An image by reference: the application keeps its media, the model still sees it.

An application that keeps its users' media in private storage does not have
to publish them, nor put their bytes in the room's events. It stores a
reference instead: a ``MediaContent`` whose URL is in a scheme it chooses
(``media://`` here), and hands the framework a ``media_resolver``. As an AI
channel builds each turn, it asks the resolver for the bytes of the images
the turn shows; the store and every broadcast keep the reference alone.

The newest four images authors sent go to the model as images, with their
words, on later turns too; anything else (a document, an image past the
bound, a reference the resolver has no bytes for) reads as a short mention.
Any participant can write a reference, so the resolver answers only for the
media of the room it is asked for.

This example sends a colour swatch with a question, then asks a follow-up
that needs the image again, and shows what the model received.

Run with:
    uv run python examples/ai_image_reference.py              # offline, mock
    ANTHROPIC_API_KEY=... uv run python examples/ai_image_reference.py anthropic
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
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.models import CompositeContent, MediaContent
from roomkit.providers.ai import AIImagePart, AIProvider, MockAIProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

logger = setup_logging("ai_image_reference")


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


class PrivateStorage:
    """The application's media store: each reference belongs to one room."""

    def __init__(self) -> None:
        self._media: dict[str, tuple[str, bytes]] = {}

    def put(self, room_id: str, name: str, data: bytes) -> str:
        url = f"media://{name}"
        self._media[url] = (room_id, data)
        return url

    async def resolve(self, media: MediaContent, room_id: str) -> bytes | None:
        """The resolver handed to the framework: bytes for this room's media only."""
        owner, data = self._media.get(media.url, ("", b""))
        return data if owner == room_id else None


def provider_for(name: str) -> AIProvider:
    if name == "anthropic":
        env = require_env("ANTHROPIC_API_KEY")
        return AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-haiku-5-5")
        )
    return MockAIProvider(responses=["It is red.", "Square."], vision=True)


async def build_kit(storage: PrivateStorage, provider: AIProvider) -> RoomKit:
    """A room where a person talks to an assistant whose media stay private."""
    kit = RoomKit(media_resolver=storage.resolve)
    kit.register_channel(WebSocketChannel("user"))
    kit.register_channel(AIChannel("assistant", provider=provider))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "user")
    await kit.attach_channel("room", "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit


async def converse(kit: RoomKit, image: MediaContent) -> None:
    """A question about the image, then a follow-up that needs it again."""
    question = "What colour is this swatch?"
    follow_up = "And what shape is it? One word."
    turns = [
        (question, CompositeContent(parts=[TextContent(body=question), image])),
        (follow_up, TextContent(body=follow_up)),
    ]
    for asked, content in turns:
        message = InboundMessage(channel_id="user", sender_id="person", content=content)
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
    """What the room's store kept of the image: the reference alone."""
    urls = [
        part.url
        for event in await kit.store.list_events("room")
        if isinstance(event.content, CompositeContent)
        for part in event.content.parts
        if isinstance(part, MediaContent)
    ]
    logger.info("The store keeps the reference alone: %s", urls)


async def main(provider_name: str) -> None:
    storage = PrivateStorage()
    provider = provider_for(provider_name)
    kit = await build_kit(storage, provider)
    try:
        swatch = storage.put("room", "swatch-42", red_square_png())
        await converse(kit, MediaContent(url=swatch, mime_type="image/png", filename="swatch.png"))
        if isinstance(provider, MockAIProvider):
            report_images_sent(provider)
        await report_store(kit)
    finally:
        await kit.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", nargs="?", default="mock", choices=["mock", "anthropic"])
    asyncio.run(main(parser.parse_args().provider))
