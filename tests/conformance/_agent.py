"""The scripted agent of a scenario: the Python core's own AI channel over a
scripted model and scripted tools.

roomkit-specs ``conformance/README.md`` ("Scripted agents") defines it: an
``ai`` channel with a ``model`` is the implementation's AI channel, whose
provider answers its n-th generation with the model's n-th round, and whose
tools each answer every call one way.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai import AIContext, AIProvider, AIResponse, AITool, AIToolCall
from roomkit.tools.policy import ToolPolicy


class ModelPastItsScriptError(Exception):
    """The agent asked the scripted model for a generation its script lacks."""


class ScriptedModel(AIProvider):
    """A provider answering each generation with the next round of a script,
    and keeping what each request carried."""

    def __init__(self, rounds: list[Mapping[str, Any]]) -> None:
        self._rounds = list(rounds)
        self.requests: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    async def generate(self, context: AIContext) -> AIResponse:
        self.requests.append({"tool_results": tool_results(context)})
        position = len(self.requests) - 1
        if position >= len(self._rounds):
            raise ModelPastItsScriptError(f"generation {position + 1} is past the script")
        return _response(self._rounds[position])


def _response(round_: Mapping[str, Any]) -> AIResponse:
    calls = [
        AIToolCall(
            id=call["id"],
            name=call["name"],
            arguments=dict(call.get("arguments", {})),
            partial=bool(call.get("cut", False)),
        )
        for call in round_.get("calls", [])
    ]
    return AIResponse(
        content=round_.get("text", ""),
        finish_reason=round_.get("finish_reason", "tool_calls" if calls else "stop"),
        usage=dict(round_.get("usage", {})),
        tool_calls=calls,
    )


def tool_results(context: AIContext) -> list[dict[str, Any]]:
    """The tool results a request carries, in order, as the model reads them."""
    found: list[dict[str, Any]] = []
    for message in context.messages:
        if isinstance(message.content, str):
            continue
        for part in message.content:
            if getattr(part, "type", None) == "tool_result":
                found.append({"tool_call_id": part.tool_call_id, "content": part.result})
    return found


_KINDS: dict[str, type[Exception]] = {}


def _error_of_kind(kind: str) -> type[Exception]:
    """An exception class named *kind*: what a tool that ``raises`` raises."""
    if kind not in _KINDS:
        _KINDS[kind] = type(kind, (Exception,), {})
    return _KINDS[kind]


def _tool_handler(tools: list[Mapping[str, Any]]) -> Any:
    by_name = {tool["name"]: tool for tool in tools}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        tool = by_name[name]
        if "returns" in tool:
            return str(tool["returns"])
        if "raises" in tool:
            raise _error_of_kind(tool["raises"]["kind"])(tool["raises"]["message"])
        if "fails" in tool:
            raise ToolFailedError(tool["fails"])
        if "refuses" in tool:
            raise ToolRefusedError(tool["refuses"])
        await asyncio.Event().wait()
        raise AssertionError("a tool that hangs never answers")

    return handler


def scripted_agent(spec: Mapping[str, Any]) -> tuple[AIChannel, ScriptedModel]:
    """The AI channel of a scripted agent, and its model."""
    model = ScriptedModel(spec["model"]["rounds"])
    tools = spec.get("tools", [])
    kwargs: dict[str, Any] = {}
    if "tool_policy" in spec:
        policy = spec["tool_policy"]
        kwargs["tool_policy"] = ToolPolicy(
            allow=list(policy.get("allow", [])), deny=list(policy.get("deny", []))
        )
    if "max_tool_rounds" in spec:
        kwargs["max_tool_rounds"] = spec["max_tool_rounds"]
    channel = AIChannel(
        spec["id"],
        provider=model,
        tools=[
            AITool(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=dict(tool.get("parameters", {})),
            )
            for tool in tools
        ],
        tool_handler=_tool_handler(tools),
        **kwargs,
    )
    return channel, model


def tool_call_record(payload: ToolCallEvent) -> dict[str, Any]:
    """A tool hook's payload as a scenario reads it (README, "Expectations")."""
    return {
        "name": payload.name,
        "tool_call_id": payload.tool_call_id,
        "arguments": payload.arguments,
        "result": payload.result,
        "is_error": payload.is_error,
        "refused": payload.refused,
        "cancelled": payload.cancelled,
        "error_detail": payload.error_detail,
    }


def patched_tool_call(payload: ToolCallEvent, patch: Mapping[str, Any]) -> ToolCallEvent:
    """A modify's ``event`` read as a patch on a tool hook's payload."""
    return dataclasses.replace(payload, **dict(patch))
