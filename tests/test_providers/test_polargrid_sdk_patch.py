"""The polargrid-sdk patch: what it works around, and that the SDK still needs it.

Every test drives a real ``polargrid.PolarGrid`` client whose server lines are
faked below its ``_stream_post``, the one seam the SDK and the patch share.
"""

from __future__ import annotations

from typing import Any

import polargrid
import pytest

from roomkit.providers.polargrid import sdk_patch

_REQUEST: dict[str, Any] = {
    "model": "qwen-3.8-27b",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hi."},
    ],
    "max_tokens": 40,
    "temperature": 0.2,
    "stream": True,
}
_LINE = {
    "id": "chatcmpl-0",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "qwen-3.8-27b",
}
_TEXT = {**_LINE, "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]}
_USAGE = {
    **_LINE,
    "choices": [],
    "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
}


def _client(lines: list[dict[str, Any]], sent: list[dict[str, Any]]) -> polargrid.PolarGrid:
    # Nothing listens there: a request that misses the seam fails at once.
    client = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")

    async def stream_post(endpoint: str, body: dict[str, Any]) -> Any:
        sent.append(body)
        for line in lines:
            yield line

    client._stream_post = stream_post  # type: ignore[method-assign]
    return client


async def _read(stream: Any) -> list[Any]:
    return [chunk async for chunk in stream]


def test_the_sdk_still_has_no_streamed_usage() -> None:
    """The day this fails, polargrid-sdk can ask for a stream's usage and
    hands it over: remove ``providers/polargrid/sdk_patch.py`` and call
    ``client.chat_completion_stream`` again."""
    asks = "stream_options" in polargrid.ChatCompletionRequest.model_fields
    reads = "usage" in polargrid.ChatCompletionChunk.model_fields
    assert not (asks and reads), (
        "polargrid-sdk now streams the usage: remove providers/polargrid/sdk_patch.py "
        "and call client.chat_completion_stream again"
    )


async def test_the_usage_the_stream_asked_for_comes_last() -> None:
    sent: list[dict[str, Any]] = []

    chunks = await _read(
        sdk_patch.chat_completion_stream(polargrid, _client([_TEXT, _USAGE], sent), _REQUEST)
    )

    assert sent[0]["stream_options"] == {"include_usage": True}
    assert isinstance(chunks[0], polargrid.ChatCompletionChunk)
    assert chunks[0].choices[0].delta.content == "Hi"
    last = chunks[-1]
    assert isinstance(last, sdk_patch.UsageChunk) and last.choices == ()
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (8, 2)


async def test_the_body_is_the_sdks_own_and_asks_for_the_usage() -> None:
    by_sdk: list[dict[str, Any]] = []
    by_patch: list[dict[str, Any]] = []

    await _read(_client([_TEXT], by_sdk).chat_completion_stream(dict(_REQUEST)))
    await _read(
        sdk_patch.chat_completion_stream(polargrid, _client([_TEXT], by_patch), dict(_REQUEST))
    )

    assert by_patch[0] == {**by_sdk[0], "stream_options": {"include_usage": True}}


async def test_an_error_line_raises_as_the_sdk_raises_it() -> None:
    lines = [_TEXT, {"error": {"message": "edge overloaded"}}]

    with pytest.raises(polargrid.NetworkError, match="edge overloaded") as by_sdk:
        await _read(_client(lines, []).chat_completion_stream(dict(_REQUEST)))
    with pytest.raises(polargrid.NetworkError, match="edge overloaded") as by_patch:
        await _read(sdk_patch.chat_completion_stream(polargrid, _client(lines, []), _REQUEST))

    assert type(by_patch.value) is type(by_sdk.value)


@pytest.mark.parametrize(
    "fields",
    [{"messages": []}, {"max_tokens": "lots"}],
    ids=["checked by the client", "refused by its model"],
)
async def test_a_request_the_sdk_refuses_is_refused_alike(fields: dict[str, Any]) -> None:
    bad = {**_REQUEST, **fields}

    with pytest.raises(polargrid.ValidationError):
        await _read(_client([], []).chat_completion_stream(dict(bad)))
    with pytest.raises(polargrid.ValidationError):
        await _read(sdk_patch.chat_completion_stream(polargrid, _client([], []), bad))


async def test_a_line_the_sdk_cannot_read_is_skipped_alike() -> None:
    unreadable = {**_LINE, "choices": "not a list"}
    by_sdk = await _read(_client([unreadable, _TEXT], []).chat_completion_stream(dict(_REQUEST)))
    by_patch = await _read(
        sdk_patch.chat_completion_stream(polargrid, _client([unreadable, _TEXT], []), _REQUEST)
    )

    assert [c.choices[0].delta.content for c in by_patch] == [
        c.choices[0].delta.content for c in by_sdk
    ]
