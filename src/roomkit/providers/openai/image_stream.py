"""Consume Images API previews without confusing them with final results."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from roomkit.providers.image.base import ImageAttempt, ImageProgressCallback, ImageResult
from roomkit.providers.image.options import plain_metadata
from roomkit.providers.image.usage import openai_image_usage


class _ImageStream:
    def __init__(
        self,
        kwargs: dict[str, Any],
        attempt: ImageAttempt,
        on_progress: ImageProgressCallback | None,
    ) -> None:
        self.kwargs = kwargs
        self.attempt = attempt
        self.on_progress = on_progress
        self.images: list[Any] = []
        self.metadata: dict[str, Any] = {}

    def _mime(self, event: Any) -> str:
        return "image/" + (
            getattr(event, "output_format", None) or self.kwargs.get("output_format", "png")
        )

    async def preview(self, event: Any) -> None:
        if not self.on_progress or not getattr(event, "b64_json", None):
            return
        mime = self._mime(event)
        preview = ImageResult(
            data=f"data:{mime};base64,{event.b64_json}", mime_type=mime, attempt_id=self.attempt.id
        )
        await self.on_progress(
            self.attempt.model_copy(update={"status": "preview", "results": [preview]})
        )

    async def completed(self, event: Any) -> None:
        self.images.append(SimpleNamespace(b64_json=event.b64_json, revised_prompt=None))
        self.metadata = {
            key: getattr(event, key, None)
            for key in ("usage", "size", "quality", "background", "output_format")
        }
        self.attempt.raw_usage = plain_metadata(self.metadata.get("usage")) or {}
        self.attempt.usage = openai_image_usage(self.metadata.get("usage"))
        mime = self._mime(event)
        self.attempt.results.append(
            ImageResult(
                data=f"data:{mime};base64,{event.b64_json}",
                mime_type=mime,
                attempt_id=self.attempt.id,
                provider_request_id=self.attempt.provider_request_id,
                usage=self.attempt.usage if len(self.images) == 1 else {},
                raw_usage=self.attempt.raw_usage if len(self.images) == 1 else {},
                effective_options=self.attempt.effective_options,
            )
        )

    async def error(self, event: Any) -> None:
        raise RuntimeError(getattr(event, "message", "Image stream failed"))


async def consume_image_stream(
    method: Any,
    kwargs: dict[str, Any],
    attempt: ImageAttempt,
    on_progress: ImageProgressCallback | None,
) -> Any:
    """Retain usage from completion events, including a truncated final stream."""
    stream = await method(**kwargs, stream=True)
    response = getattr(stream, "response", None)
    if response is not None:
        attempt.provider_request_id = response.headers.get("x-request-id")
    state = _ImageStream(kwargs, attempt, on_progress)
    handlers = {"partial_image": state.preview, "completed": state.completed, "error": state.error}
    try:
        async for event in stream:
            handler = handlers.get(getattr(event, "type", "").rsplit(".", 1)[-1])
            if handler:
                await handler(event)
    finally:
        await stream.close()
    return SimpleNamespace(
        data=state.images, _request_id=attempt.provider_request_id, **state.metadata
    )
