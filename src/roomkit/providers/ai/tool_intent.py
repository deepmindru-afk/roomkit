"""A tool call's intent: the short sentence a model writes with a call to tell
the person what the call does (RFC §6.4).

An agent with ``tool_call_intents`` on shows the model each of its tools'
schemas with one optional property more, :data:`INTENT_PROPERTY`. The tool
loop takes that property out of a call's arguments before anything reads
them, and carries its value, normalized, as the call's ``intent``: the tool
never receives it, and a closed schema never refuses it. A tool whose own
schema declares the property keeps it as its argument.

The intent is display text the model wrote: it grants nothing, and says
nothing of what the call did. The rules live here once, for the schemas the
model is shown, the calls it makes and the history it reads back.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Container, Iterable
from functools import cache
from typing import Any

from roomkit.providers.ai.base import AITool, StreamToolCall

logger = logging.getLogger("roomkit.providers.ai")

INTENT_PROPERTY = "_roomkit_intent"
"""The reserved property a call's intent rides in."""

INTENT_MAX_CHARS = 200
"""The longest intent kept, in characters (code points)."""

_INTENT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        "One short sentence, in the language of the conversation, telling the user "
        "what this call does. Shown to the user; never passed to the tool."
    ),
}

# The bidirectional formatting characters: they can make the text a person
# reads differ from the text that was written.
_BIDI_CONTROLS = frozenset(
    chr(code) for code in (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A))
)


def normalized_intent(value: Any) -> str | None:
    """*value* as a call's intent: a string with each control character a
    space, the bidirectional controls removed, its whitespace runs one space,
    trimmed and cut at :data:`INTENT_MAX_CHARS`; ``None`` when nothing usable
    is left, or *value* is not a string."""
    if not isinstance(value, str):
        return None
    kept = (
        " " if unicodedata.category(char) == "Cc" else char
        for char in value
        if char not in _BIDI_CONTROLS
    )
    text = " ".join("".join(kept).split())
    return text[:INTENT_MAX_CHARS].rstrip() or None


def presented_tools(tools: list[AITool]) -> list[AITool]:
    """*tools* as an agent with intents shows them to its model: each schema
    with the reserved property, one that cannot take it as it is."""
    return [_presented(tool) for tool in tools]


def _presented(tool: AITool) -> AITool:
    parameters = tool.parameters
    if not _takes_intent(parameters):
        _log_unpresented(tool.name)
        return tool
    properties = {**parameters.get("properties", {}), INTENT_PROPERTY: _INTENT_SCHEMA}
    shown = {**parameters, "type": "object", "properties": properties}
    return tool.model_copy(update={"parameters": shown})


def _takes_intent(parameters: dict[str, Any]) -> bool:
    """Whether a tool's schema can take the reserved property: an object
    schema that does not declare it already."""
    properties = parameters.get("properties", {})
    return (
        parameters.get("type", "object") == "object"
        and isinstance(properties, dict)
        and INTENT_PROPERTY not in properties
    )


@cache
def _log_unpresented(name: str) -> None:
    """Say once that the tool *name* is shown without the reserved property."""
    logger.warning(
        "Tool %s is shown without %s: its schema declares it, or is not an object; "
        "its calls carry no intent",
        name,
        INTENT_PROPERTY,
    )


def owning_tools(*toolsets: Iterable[AITool] | None) -> frozenset[str]:
    """The names, in *toolsets*, of the tools whose own schema declares the
    reserved property: their calls keep it as an argument."""
    return frozenset(
        tool.name for tools in toolsets for tool in tools or () if _declares_intent(tool)
    )


def _declares_intent(tool: AITool) -> bool:
    properties = tool.parameters.get("properties")
    return isinstance(properties, dict) and INTENT_PROPERTY in properties


def take_intent(call: StreamToolCall, owners: Container[str]) -> StreamToolCall:
    """*call* without the reserved property, its value normalized as the
    call's intent; *call* itself when it does not carry the property, when its
    arguments do not read, or when its tool is one of *owners*."""
    if call.partial or call.name in owners or INTENT_PROPERTY not in call.arguments:
        return call
    arguments = dict(call.arguments)
    intent = normalized_intent(arguments.pop(INTENT_PROPERTY))
    return call.model_copy(update={"arguments": arguments, "intent": intent})


def with_intent(arguments: dict[str, Any], intent: str | None) -> dict[str, Any]:
    """*arguments* as a replay shows the model its call: with its intent in
    the reserved property, when it has one."""
    if intent is None:
        return arguments
    return {**arguments, INTENT_PROPERTY: intent}
