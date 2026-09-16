"""Preview, completion and interruption have distinct observable outcomes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.image import ImageAttempt, ImageGenerationError, ImageOptions
from tests.test_providers.test_openai_image import PNG_B64, _provider, _usage


class Stream:
    def __init__(self, *, interrupted: bool = False) -> None:
        self.interrupted = interrupted
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(type="image_generation.partial_image", b64_json=PNG_B64)
        yield SimpleNamespace(
            type="image_generation.completed",
            b64_json=PNG_B64,
            usage=_usage(100, 0, 1100),
            output_format="png",
        )
        if self.interrupted:
            raise RuntimeError("stream interrupted")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("interrupted", [False, True])
async def test_stream_preserves_finals_and_usage_even_when_interrupted(interrupted: bool) -> None:
    provider = _provider()
    stream = Stream(interrupted=interrupted)
    provider._client.images.generate = AsyncMock(return_value=stream)
    events: list[ImageAttempt] = []

    async def progress(event: ImageAttempt) -> None:
        events.append(event)

    if interrupted:
        with pytest.raises(ImageGenerationError) as caught:
            await provider.generate_with_options(
                "a fox", options=ImageOptions(partial_images=1), on_progress=progress
            )
        results = caught.value.results
        assert caught.value.attempts[0].usage["output_image_tokens"] == 1100
    else:
        results = await provider.generate_with_options(
            "a fox", options=ImageOptions(partial_images=1), on_progress=progress
        )
    assert len(results) == 1
    assert results[0].usage["output_image_tokens"] == 1100
    assert stream.closed
    assert [event.status for event in events][:2] == ["started", "preview"]
    assert provider._client.images.generate.await_args.kwargs["stream"] is True
