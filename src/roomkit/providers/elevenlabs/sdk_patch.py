"""A patch of the ElevenLabs SDK, kept apart: unregistered client tools.

The SDK's ``ClientTools.handle`` answers a call to a name no handler was
registered for itself, ``Tool 'x' is not registered``, as an error result on
the wire (measured on elevenlabs 2.69.0). A tool the agent's dashboard
declares and RoomKit withholds (its tool policy, Tool Search, skill gating)
is then called without RoomKit ever seeing it, where every other realtime
provider hands such a call to the channel, whose gate refuses and reports it
(RFC §12.4).

:func:`client_tools` builds the SDK's ``ClientTools`` with that one method
overridden: a name with no handler goes to *route*, the bridge every declared
name has, and is served, refused and reported as any call. Everything else,
the registry, the dispatch and the result on the wire, stays the SDK's.

The provider also counts on one more behaviour of the SDK: its dispatch
answers every outcome of a handler on the wire but a cancellation, so a
handler that raises ``asyncio.CancelledError`` sends nothing. That is how a
call no result can name (no id, or an id still in flight) gets none
(``_hand_on_unanswerable``). The canary
``test_the_sdk_sends_nothing_for_a_cancelled_handler`` watches it.

Remove the patch when ``test_the_sdk_still_answers_an_unregistered_tool_itself``
in ``tests/test_providers/test_elevenlabs_sdk_patch.py`` fails: the SDK then
hands such calls on itself. Undo with it, in ``providers/elevenlabs/realtime.py``,
the ``sdk_patch.client_tools`` call (back to ``ClientTools(loop=…)``) and
``_route_unregistered`` with its ``functools.partial`` import.

A second patch keeps a call's id the service's (RMK-442). The SDK hands a
``client_tool_call`` to its tools as ``{"tool_call_id": <the service's>,
**parameters}``: a parameter the model wrote under ``tool_call_id`` replaces
the id, and the SDK answers on the wire under the model's text; a message with
no ``tool_call_id`` raises ``KeyError``, which ends the conversation.
:func:`conversation` builds the SDK's ``AsyncConversation`` with its message
handling rewriting such a call first: the model's parameters go under one
reserved key, which :func:`split_call` reads back, and a missing id becomes
``None``, a call the channel refuses as one without an id. Parameters that are
no object (a string, a list), which the SDK's ``**parameters`` raised on and
the conversation ended with, go on as the model's text, which the channel
refuses as unreadable (RFC §6.4). Remove it, with
:func:`conversation` and :func:`split_call` (back to ``AsyncConversation`` and
the ``tool_call_id`` pop in ``_make_tool_handler``), when
``test_the_sdk_still_lets_a_parameter_replace_the_call_id`` fails.

``pyproject.toml`` caps ``elevenlabs`` below the next minor, and the
``providers`` extra installs it so the canaries run in CI: move the cap once
they pass on the new minor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.providers.ai.tool_calls import readable_arguments

Route = Callable[[str, dict[str, Any]], Awaitable[Any]]


def client_tools(sdk_class: Any, *, loop: asyncio.AbstractEventLoop, route: Route) -> Any:
    """The SDK's *sdk_class* (``ClientTools``) on *loop*, a name with no
    registered handler routed to *route* rather than answered by the SDK."""

    class _RoutingClientTools(sdk_class):
        async def handle(self, tool_name: str, parameters: dict[str, Any]) -> Any:
            if tool_name not in self.tools:
                return await route(tool_name, parameters)
            return await super().handle(tool_name, parameters)

    return _RoutingClientTools(loop=loop)


CALL_ID = "tool_call_id"
"""Where the SDK puts the service's call id among a call's parameters."""

_ARGUMENTS = "roomkit.arguments"
"""Where the model's own parameters travel, out of the SDK's reach."""


def conversation(sdk_class: Any) -> Any:
    """The SDK's *sdk_class* (``AsyncConversation``) with a ``client_tool_call``
    rewritten before the SDK reads it: its id stays the service's."""

    class _ServiceIdConversation(sdk_class):
        async def _handle_message_core_async(self, message: Any, message_handler: Any) -> Any:
            return await super()._handle_message_core_async(
                _service_id_first(message), message_handler
            )

    return _ServiceIdConversation


def _service_id_first(message: Any) -> Any:
    """*message* with a client tool call's parameters under one reserved key,
    so none replaces the service's id, and a missing id ``None``."""
    if not isinstance(message, dict) or message.get("type") != "client_tool_call":
        return message
    call = dict(message.get("client_tool_call") or {})
    call["parameters"] = {_ARGUMENTS: _model_arguments(call.get("parameters"))}
    call.setdefault(CALL_ID, None)
    return {**message, "client_tool_call": call}


def _model_arguments(parameters: Any) -> dict[str, Any] | str:
    """The model's parameters as every realtime provider reads a call's
    (``readable_arguments``): a mapping, or the model's text, which the
    channel refuses as unreadable (RFC §6.4)."""
    return readable_arguments(parameters)


def split_call(parameters: dict[str, Any]) -> tuple[str, dict[str, Any] | str]:
    """A call's id, the service's, and the model's arguments, from the
    parameters the SDK hands a tool; a ``tool_call_id`` the model wrote is
    one of the arguments (RFC §12.4)."""
    call_id = str(parameters.get(CALL_ID) or "")
    if _ARGUMENTS in parameters:
        arguments = parameters[_ARGUMENTS]
        return call_id, dict(arguments) if isinstance(arguments, dict) else str(arguments)
    return call_id, {key: value for key, value in parameters.items() if key != CALL_ID}
