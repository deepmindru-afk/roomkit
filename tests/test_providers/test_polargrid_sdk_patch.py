"""The polargrid-sdk patch: what it works around, and that the SDK still needs it.

Every test drives a real ``polargrid.PolarGrid`` client whose server lines are
faked below its ``_stream_post``, the one seam the SDK and the patch share.
"""

from __future__ import annotations

import logging
from typing import Any

import polargrid
import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    ProviderError,
    StreamDone,
    StreamTextDelta,
)
from roomkit.providers.polargrid import sdk_patch
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig

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


async def test_the_sdk_still_cannot_stream_its_usage() -> None:
    """The day this fails, polargrid-sdk asks for a stream's usage and hands
    it over itself: remove ``providers/polargrid/sdk_patch.py`` as its
    docstring says."""
    sent: list[dict[str, Any]] = []
    request = dict(_REQUEST)
    if "stream_options" in polargrid.ChatCompletionRequest.model_fields:
        request["stream_options"] = {"include_usage": True}

    chunks = await _read(_client([_TEXT, _USAGE], sent).chat_completion_stream(request))

    asks = bool((sent[0].get("stream_options") or {}).get("include_usage"))
    hands_over = any(getattr(chunk, "usage", None) for chunk in chunks)
    assert not (asks and hands_over), (
        "polargrid-sdk now streams the usage: remove providers/polargrid/sdk_patch.py "
        "as its docstring says"
    )


def test_the_sdk_still_replaces_a_zero_temperature() -> None:
    """The day this fails, polargrid-sdk sends the values it is given: drop
    ``_GIVEN_AS_IS`` from ``providers/polargrid/sdk_patch.py``."""
    client = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")
    request = polargrid.ChatCompletionRequest(**{**_REQUEST, "temperature": 0.0})

    body = client._build_chat_completion_body(request, stream_override=False)

    assert body["temperature"] != 0.0, (
        "polargrid-sdk now sends a zero temperature: drop _GIVEN_AS_IS from "
        "providers/polargrid/sdk_patch.py"
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
    [
        {"messages": [{"role": "developer", "content": "x"}]},
        {"messages": []},
        {"max_tokens": "lots"},
        {"model": ""},
    ],
    ids=["a message its model refuses", "no message", "a field its model refuses", "no model"],
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


async def test_a_usage_line_it_cannot_read_is_dropped_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unreadable = {**_LINE, "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": None}}

    with caplog.at_level(logging.WARNING, logger="roomkit.providers.polargrid"):
        chunks = await _read(
            sdk_patch.chat_completion_stream(polargrid, _client([_TEXT, unreadable], []), _REQUEST)
        )

    assert not any(isinstance(chunk, sdk_patch.UsageChunk) for chunk in chunks)
    assert chunks[0].choices[0].delta.content == "Hi"
    assert "usage dropped" in caplog.text


def _provider(lines: list[dict[str, Any]]) -> PolarGridAIProvider:
    provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model="qwen-3.8-27b"))
    provider._client = _client(lines, [])
    return provider


async def test_a_message_polargrid_refuses_is_not_retried() -> None:
    context = AIContext(messages=[AIMessage(role="developer", content="x")])

    with pytest.raises(ProviderError) as refused:
        await _read(_provider([_TEXT]).generate_structured_stream(context))

    assert refused.value.retryable is False


async def test_a_turn_whose_usage_cannot_be_read_still_ends() -> None:
    unreadable = {**_LINE, "choices": [], "usage": {"prompt_tokens": 8}}
    context = AIContext(messages=[AIMessage(role="user", content="Say hi.")])

    events = await _read(_provider([_TEXT, unreadable]).generate_structured_stream(context))

    assert "".join(e.text for e in events if isinstance(e, StreamTextDelta)) == "Hi"
    [done] = [e for e in events if isinstance(e, StreamDone)]
    assert done.usage == {}


async def test_an_error_line_ends_the_turn_as_a_transient_failure() -> None:
    lines = [_TEXT, {"error": {"message": "edge overloaded"}}]
    context = AIContext(messages=[AIMessage(role="user", content="Say hi.")])

    with pytest.raises(ProviderError, match="edge overloaded") as failed:
        await _read(_provider(lines).generate_structured_stream(context))

    assert failed.value.retryable is True
