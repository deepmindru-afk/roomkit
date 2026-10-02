"""A patch of the ollama SDK, kept apart until the SDK keeps tool schemas whole.

ollama-python validates each tool declaration into its ``Tool`` model before
the request leaves, and that model's ``Property`` keeps only ``type``,
``items``, ``description`` and ``enum``: a nested object's ``properties`` and
``required``, an ``anyOf``, a ``$ref`` and every constraint are dropped
(measured on ollama-python 0.6.2 and 0.6.3). The Ollama server keeps them,
and the model needs them: asked to book with a code under a declared
``booking.zq_code``, qwen3:8b on Ollama 0.17.5 answered ``code`` through the
SDK and ``zq_code`` without it.

Upstream: reported as ollama/ollama-python#429 (January 2025, closed as fixed
while still present) and #724 (September 2026); the fix, PR #725, is unmerged
as of 2026-10-02.

:func:`chat` sends a request that declares tools as ``AsyncClient.chat``
would, with the declarations as given, through the client's own request
method, so its transport, streaming and response type stay the SDK's. Remove
this module and call ``client.chat`` again once an ollama-python release
keeps nested properties: ``test_the_sdk_still_drops_nested_tool_schemas`` in
``tests/test_providers/test_ollama_sdk_patch.py`` fails on that release.
"""

from __future__ import annotations

from typing import Any


async def chat(client: Any, response_type: type, **request: Any) -> Any:
    """``client.chat(**request)``, its tool declarations sent as given.

    *response_type* is the SDK's ``ChatResponse``, which the client parses
    each answer into. A request without tools goes through ``client.chat``
    unchanged: only a declaration has anything for the SDK to drop.
    """
    tools = request.pop("tools", None)
    if not tools:
        return await client.chat(**request)
    body = {key: value for key, value in request.items() if value is not None}
    # As the SDK's ``_copy_messages`` sends them: a message without its empty
    # fields.
    body["messages"] = [
        {key: value for key, value in message.items() if value}
        for message in request.get("messages") or []
    ]
    body["tools"] = list(tools)
    stream = bool(request.get("stream", False))
    return await client._request(response_type, "POST", "/api/chat", json=body, stream=stream)
