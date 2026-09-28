"""Mock vision provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.response_schema import check_schema_answer, check_schema_request
from roomkit.video.vision.base import VisionProvider, VisionResult

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing.

    Returns pre-configured descriptions in round-robin order and
    records every frame submitted for analysis.

    Example::

        provider = MockVisionProvider(
            descriptions=["A person waving", "An empty room"],
        )
        result = await provider.analyze_frame(frame)
        assert result.description == "A person waving"
        assert len(provider.calls) == 1
    """

    def __init__(
        self,
        descriptions: list[str] | None = None,
        *,
        labels: list[list[str]] | None = None,
        response_schema: bool = False,
    ) -> None:
        self.descriptions = descriptions or [
            "A video frame",
            "A person in a room",
        ]
        self.labels = labels or [["person"], ["room"]]
        self.calls: list[VideoFrame] = []
        self._index = 0
        self._response_schema = response_schema

    @property
    def supports_response_schema(self) -> bool:
        """``response_schema=True`` makes the scripted descriptions answer a
        schema: each must then be a JSON document satisfying it."""
        return self._response_schema

    async def analyze_frame(
        self,
        frame: VideoFrame,
        *,
        prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> VisionResult:
        if response_schema is not None:
            check_schema_request(
                response_schema, supported=self._response_schema, provider="mock-vision"
            )
        self.calls.append(frame)
        desc = self.descriptions[self._index % len(self.descriptions)]
        frame_labels = self.labels[self._index % len(self.labels)]
        # A description that fails the check is still spent: the next call
        # plays the next one, as a real model would answer afresh.
        self._index += 1
        if response_schema is not None:
            check_schema_answer(desc, schema=response_schema, provider="mock-vision")
        return VisionResult(
            description=desc,
            labels=frame_labels,
        )
