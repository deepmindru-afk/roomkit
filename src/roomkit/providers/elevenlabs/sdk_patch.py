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

``pyproject.toml`` caps ``elevenlabs`` below the next minor, and the
``providers`` extra installs it so the canaries run in CI: move the cap once
they pass on the new minor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

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
