"""The ollama SDK patch: what it works around, and that the SDK still needs it.

Every test drives the real ``ollama.AsyncClient`` on a mock transport, so the
bodies compared are the ones the SDK puts on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest

from roomkit.providers.ollama import sdk_patch

# Only keywords the Ollama server reads (its ToolProperty): the SDK drops
# every nested one, which is what the patch exists for.
NESTED = {
    "type": "object",
    "required": ["booking"],
    "properties": {
        "booking": {
            "type": "object",
            "description": "The booking.",
            "properties": {
                "zq_code": {"type": "string"},
                "guest": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "rooms": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"kind": {"type": "string"}}},
                },
                "tier": {"type": "string", "enum": ["basic", "suite"]},
            },
            "required": ["zq_code"],
        },
        "when": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "$defs": {"place": {"type": "string"}},
}
FLAT = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def _tool(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": "book_room", "description": "Book a room.", "parameters": parameters},
    }


_REQUEST: dict[str, Any] = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Book ABC123 for 2 nights.", "images": ["aGVsbG8="]},
        {
            "role": "assistant",
            "content": "",
            "thinking": "A booking.",
            "tool_calls": [{"function": {"name": "book_room", "arguments": {"q": 1}}}],
        },
        {"role": "tool", "content": "booked", "tool_name": "book_room"},
    ],
    "options": {"temperature": 0.2, "num_predict": 64},
    "think": True,
    "keep_alive": "5m",
}
_ANSWER = {
    "model": "qwen3:8b",
    "created_at": "2026-10-02T00:00:00Z",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
    "done_reason": "stop",
}


def _client(sent: list[dict[str, Any]]) -> ollama.AsyncClient:
    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if body.get("stream"):
            lines = [json.dumps({**_ANSWER, "done": False}), json.dumps(_ANSWER)]
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(200, json=_ANSWER)

    return ollama.AsyncClient(host="http://ollama.test", transport=httpx.MockTransport(answer))


async def test_the_sdk_still_drops_nested_tool_schemas() -> None:
    """The day this fails, ollama-python sends whole every keyword the server
    reads: remove ``providers/ollama/sdk_patch.py`` as its docstring says."""
    sent: list[dict[str, Any]] = []
    minimal = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Book."}]}

    await _client(sent).chat(**minimal, tools=[_tool(NESTED)])

    declared = sent[0]["tools"][0]["function"]["parameters"]
    assert declared != NESTED, (
        "ollama-python now sends nested tool schemas whole: "
        "remove providers/ollama/sdk_patch.py as its docstring says"
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_a_declaration_goes_out_as_given(stream: bool) -> None:
    sent: list[dict[str, Any]] = []
    client = _client(sent)

    answer = await sdk_patch.chat(ollama, client, **_REQUEST, tools=[_tool(NESTED)], stream=stream)
    # A stream's request leaves as it is read.
    responses = [part async for part in answer] if stream else [answer]

    assert sent[0]["tools"] == [_tool(NESTED)]
    assert all(isinstance(part, ollama.ChatResponse) for part in responses)
    assert responses[-1].message.content == "ok"


@pytest.mark.parametrize("stream", [False, True])
async def test_the_rest_of_the_request_is_what_the_sdk_sends(stream: bool) -> None:
    by_sdk: list[dict[str, Any]] = []
    by_patch: list[dict[str, Any]] = []
    request = {**_REQUEST, "tools": [_tool(FLAT)], "stream": stream}

    sdk_answer = await _client(by_sdk).chat(**request)
    patch_answer = await sdk_patch.chat(ollama, _client(by_patch), **request)
    if stream:
        _ = [part async for part in sdk_answer], [part async for part in patch_answer]

    assert {k: v for k, v in by_patch[0].items() if k != "tools"} == {
        k: v for k, v in by_sdk[0].items() if k != "tools"
    }


async def test_messages_go_out_as_the_sdk_sends_them(tmp_path: Path) -> None:
    """An image given by its path is read and encoded, an unknown key left
    out, and a request that says nothing of streaming does not stream."""
    image = tmp_path / "cat.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    messages = [
        {"role": "user", "content": "Look.", "images": [str(image), "aGVsbG8="], "mood": "x"}
    ]
    request = {"model": "qwen3:8b", "messages": messages, "tools": [_tool(FLAT)]}
    by_sdk: list[dict[str, Any]] = []
    by_patch: list[dict[str, Any]] = []

    await _client(by_sdk).chat(**request)
    await sdk_patch.chat(ollama, _client(by_patch), **request)

    assert by_patch[0]["stream"] is False
    assert {k: v for k, v in by_patch[0].items() if k != "tools"} == {
        k: v for k, v in by_sdk[0].items() if k != "tools"
    }


async def test_a_request_without_tools_is_the_sdks_own() -> None:
    by_sdk: list[dict[str, Any]] = []
    by_patch: list[dict[str, Any]] = []

    await _client(by_sdk).chat(**_REQUEST)
    await sdk_patch.chat(ollama, _client(by_patch), **_REQUEST)

    assert by_patch == by_sdk
