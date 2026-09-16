"""Consume Images API previews without confusing them with final results."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from roomkit.providers.image.base import ImageAttempt, ImageProgressCallback, ImageResult
from roomkit.providers.image.options import plain_metadata
from roomkit.providers.image.usage import openai_image_usage


async def consume_image_stream(
    method: Any,
    kwargs: dict[str, Any],
    attempt: ImageAttempt,
    on_progress: ImageProgressCallback | None,
) -> Any:
    """Retain usage from completion events, including a truncated final stream."""
    stream = await method(**kwargs, stream=True)
    images: list[Any] = []
    metadata: dict[str, Any] = {}
    try:
        async for event in stream:
            kind = getattr(event, "type", "")
            if kind.endswith(".partial_image"):
                if on_progress and getattr(event, "b64_json", None):
                    mime = "image/" + (
                        getattr(event, "output_format", None) or kwargs.get("output_format", "png")
                    )
                    preview = ImageResult(
                        data=f"data:{mime};base64,{event.b64_json}",
                        mime_type=mime,
                        attempt_id=attempt.id,
                    )
                    await on_progress(
                        attempt.model_copy(update={"status": "preview", "results": [preview]})
                    )
            elif kind.endswith(".completed"):
                images.append(SimpleNamespace(b64_json=event.b64_json, revised_prompt=None))
                metadata = {
                    key: getattr(event, key, None)
                    for key in ("usage", "size", "quality", "background", "output_format")
                }
                attempt.raw_usage = plain_metadata(metadata.get("usage")) or {}
                attempt.usage = openai_image_usage(metadata.get("usage"))
                mime = "image/" + (
                    metadata.get("output_format") or kwargs.get("output_format", "png")
                )
                attempt.results.append(
                    ImageResult(
                        data=f"data:{mime};base64,{event.b64_json}",
                        mime_type=mime,
                        attempt_id=attempt.id,
                        usage=attempt.usage if len(images) == 1 else {},
                        raw_usage=attempt.raw_usage if len(images) == 1 else {},
                        effective_options=attempt.effective_options,
                    )
                )
            elif kind == "error":
                raise RuntimeError(getattr(event, "message", "Image stream failed"))
    finally:
        await stream.close()
    return SimpleNamespace(data=images, **metadata)
