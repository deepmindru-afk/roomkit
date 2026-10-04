"""The human-input tools one channel serves (RFC §9.3, §21.6).

A channel given a :class:`~roomkit.tools.human_input.HumanInputToolHandler`
(``human_input_handler=``) serves its tools itself, on every door it has: an
``AIChannel`` turn, a realtime voice session (the provider's call, a call
recovered from speech, a reasoning backend's), a conference. Each holds one
:class:`ChannelHumanInput`, which gives the tools the same rules wherever the
model calls them:

* declared beside the host's tools and served before the host's handler;
* bounded by the handler's own ``timeout``, never by the channel's default
  call bound: a person takes the time they take;
* each request announced through ``ON_USER_INPUT_REQUIRED`` once the channel
  is registered with a kit, a BLOCK rejecting it, the request naming the
  channel type of the door it was asked on;
* the requests still open settled when the channel closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.models.enums import ChannelType
    from roomkit.providers.ai.base import AITool
    from roomkit.tools.human_input import HumanInputToolHandler, OnInputRequiredCallback


class ChannelHumanInput:
    """The human-input tools of one channel object: what it declares, what it
    serves, and the scope of requests it owns."""

    def __init__(self, tools: HumanInputToolHandler, channel_type: ChannelType) -> None:
        self._tools = tools
        self._channel_type = channel_type
        # The token naming this channel object as the owner of its id's
        # requests, handed back on close: a channel displaced under the same
        # id and torn down later closes nothing its replacement holds.
        self._registration: int | None = None

    @property
    def names(self) -> frozenset[str]:
        """Every tool name it serves: a call to one asks a person."""
        return frozenset(self._tools.tool_names)

    @property
    def definitions(self) -> list[AITool]:
        """The tools it declares to the model; a name it serves without a
        definition is declared by the host's tools."""
        return self._tools.tools

    @property
    def declared_names(self) -> frozenset[str]:
        """The names its definitions carry: no other tool may take one."""
        return frozenset(tool.name for tool in self.definitions)

    def serves(self, name: str) -> bool:
        """Whether a call to *name* asks a person."""
        return name in self._tools.tool_names

    async def serve(self, name: str, arguments: dict[str, Any]) -> str:
        """The person's answer to the call, asked on this channel's door."""
        return await self._tools.ask(name, arguments, channel_type=self._channel_type)

    def register(self, channel_id: str, on_input_required: OnInputRequiredCallback) -> None:
        """Announce the requests of the channel *channel_id* through
        *on_input_required* (the kit's ``ON_USER_INPUT_REQUIRED`` hooks), this
        channel object owning them."""
        handler = self._tools.handler
        self._registration = handler._set_on_input_required(channel_id, on_input_required)

    async def close(self, channel_id: str) -> None:
        """Settle the requests the channel still has open, and take no more
        until it registers again."""
        await self._tools.handler.close(channel_id=channel_id, registration=self._registration)
