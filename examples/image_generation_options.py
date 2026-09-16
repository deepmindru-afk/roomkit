"""Typed image options and observable outcomes; mock by default, no key required.

Run: uv run python examples/image_generation_options.py
Paid: uv run --extra openai python examples/image_generation_options.py --provider openai
Paid: uv run --extra gemini python examples/image_generation_options.py --provider gemini
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from roomkit import ImageAttempt, ImageGenerationError, ImageOptions, MockImageProvider
from roomkit.providers.gemini import GeminiImageConfig, GeminiImageProvider
from roomkit.providers.openai import OpenAIImageConfig, OpenAIImageProvider

logger = logging.getLogger("roomkit.examples.images")


async def report(attempt: ImageAttempt) -> None:
    logger.info("%s %s usage=%s", attempt.id, attempt.status, attempt.usage)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["mock", "openai", "gemini"], default="mock")
    args = parser.parse_args()
    factories = {
        "mock": MockImageProvider,
        "openai": lambda: OpenAIImageProvider(
            OpenAIImageConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-image-2.5-flare",
            )
        ),
        "gemini": lambda: GeminiImageProvider(
            GeminiImageConfig(
                api_key=os.environ["GEMINI_API_KEY"],
                model="gemini-3.1-flash-image",
            )
        ),
    }
    options = {
        "mock": ImageOptions(),
        "openai": ImageOptions(quality="high", output_format="webp", partial_images=1),
        "gemini": ImageOptions(aspect_ratio="16:9", image_size="1K"),
    }
    provider = factories[args.provider]()
    try:
        try:
            results = await provider.generate_with_options(
                "An origami fox on a white background",
                options=options[args.provider],
                on_progress=report if args.provider != "mock" else None,
            )
        except ImageGenerationError as error:
            logger.error("%s; available outcomes=%s", error, len(error.attempts))
            results = error.results
        for result in results:
            logger.info("Received %s (%s bytes)", result.mime_type, len(result.decoded()))
    finally:
        await provider.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
