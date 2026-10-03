"""The ElevenLabs SDK patch and the canary that says when to drop it (RMK-440).

The SDK's ``ClientTools`` answers a call to an unregistered name itself;
``providers/elevenlabs/sdk_patch.py`` hands such a call to the channel. The
canary fails once the SDK stops answering it itself, or stops dispatching
through ``handle``: the patch is then dropped, or reworked.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.providers.elevenlabs import sdk_patch

conversation = pytest.importorskip("elevenlabs.conversational_ai.conversation")


async def _execute(tools: Any, name: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    tools.start()
    try:
        tools.execute_tool(name, {"tool_call_id": "t1"}, results.append)
        for _ in range(100):
            if results:
                break
            await asyncio.sleep(0.01)
    finally:
        tools.stop()
    [result] = results
    return result


async def test_the_sdk_still_answers_an_unregistered_tool_itself() -> None:
    tools = conversation.ClientTools(loop=asyncio.get_running_loop())

    result = await _execute(tools, "secret_op")

    assert result["is_error"] is True
    assert "not registered" in result["result"]


async def test_the_patch_routes_an_unregistered_tool_through_the_sdks_dispatch() -> None:
    routed: list[str] = []

    async def route(name: str, parameters: dict[str, Any]) -> str:
        routed.append(name)
        return "refused by the channel"

    tools = sdk_patch.client_tools(
        conversation.ClientTools, loop=asyncio.get_running_loop(), route=route
    )

    result = await _execute(tools, "secret_op")

    assert routed == ["secret_op"]
    assert (result["result"], result["is_error"]) == ("refused by the channel", False)


async def test_the_sdk_sends_nothing_for_a_cancelled_handler() -> None:
    """How the provider sends nothing for a call no result can name."""
    tools = conversation.ClientTools(loop=asyncio.get_running_loop())

    async def unanswerable(parameters: dict[str, Any]) -> str:
        raise asyncio.CancelledError

    tools.register("lookup", unanswerable, is_async=True)
    results: list[dict[str, Any]] = []
    tools.start()
    try:
        tools.execute_tool("lookup", {"tool_call_id": "t1"}, results.append)
        await asyncio.sleep(0.1)
    finally:
        tools.stop()

    assert results == []
