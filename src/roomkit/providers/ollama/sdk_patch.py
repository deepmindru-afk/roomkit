"""A patch of the ollama SDK, kept apart until the SDK sends tool schemas whole.

ollama-python validates each tool declaration into its ``Tool`` model before
the request leaves, and that model's ``Property`` keeps only ``type``,
``items``, ``description`` and ``enum``: a nested object's ``properties`` and
``required`` and an ``anyOf`` are dropped (measured on ollama-python 0.6.2
and 0.6.3), although the Ollama server reads them. The model needs them:
asked to book with a code under a declared ``booking.zq_code``, qwen3:8b on
Ollama 0.17.5 answered ``code`` through the SDK and ``zq_code`` without it.
(The server itself drops a ``$ref`` and the constraints, ``minLength`` and
the like: no client can send those through.)

Upstream: ollama/ollama-python#724, with a fix proposed in PR #725 (nested
``properties`` only); first reported as #429.

:func:`chat` sends a request that declares tools as ``AsyncClient.chat``
would, its messages built by the SDK's own ``Message`` and ``Image``, with
the declarations as given, through the client's own request method, so its
transport, streaming and response type stay the SDK's.

Remove it when ``test_the_sdk_still_drops_nested_tool_schemas`` in
``tests/test_providers/test_ollama_sdk_patch.py`` fails: the SDK then sends
whole every keyword the server reads. Undo with it: the two ``sdk_patch.chat``
calls and ``_chat_response`` in ``providers/ollama/ai.py`` (back to
``self._client.chat(**kwargs)``), and in ``tests/test_providers/test_ollama.py``
the ``_request`` mock and the three tests asserting on it.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _message(sdk: ModuleType, message: dict[str, Any]) -> dict[str, Any]:
    """*message* as the SDK's ``_copy_messages`` sends it: its empty fields
    dropped, an image read from its path and encoded, unknown keys left out."""
    fields = {
        key: [sdk.Image(value=image) for image in value] if key == "images" else value
        for key, value in message.items()
        if value
    }
    return sdk.Message.model_validate(fields).model_dump(exclude_none=True)


async def chat(sdk: ModuleType, client: Any, **request: Any) -> Any:
    """``client.chat(**request)``, its tool declarations sent as given.

    *sdk* is the ``ollama`` module the client comes from. A request without
    tools goes through ``client.chat`` unchanged: only a declaration has
    anything for the SDK to drop.
    """
    tools = request.pop("tools", None)
    if not tools:
        return await client.chat(**request)
    stream = bool(request.pop("stream", False))
    body = {key: value for key, value in request.items() if value is not None}
    body["messages"] = [_message(sdk, message) for message in request.get("messages") or []]
    body["tools"] = list(tools)
    body["stream"] = stream
    return await client._request(sdk.ChatResponse, "POST", "/api/chat", json=body, stream=stream)
