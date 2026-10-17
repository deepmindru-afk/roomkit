"""The messages of a turn's link (RFC §10.6.3).

Every message a turn backend carries is one of these records, serialized to
JSON whatever the transport, the in-memory one included: no object of the turn
crosses between the room owner and the executor. A link carries an
:class:`Envelope` per message, numbered per direction, so a gap or a
duplicate is seen on arrival.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AITool


class OwnerTool(BaseModel):
    """A tool the room owner serves a turn, as the request lists it (RFC §10.6.5)."""

    tool: AITool
    waits_by_design: bool = False
    """Bounded by its own bound, not the channel's default call bound (RFC §21.6)."""


class TurnRequest(BaseModel):
    """What the owner sends an executor to start one turn."""

    turn_id: str
    room_id: str
    channel_id: str
    event: RoomEvent
    binding: ChannelBinding
    context: RoomContext
    answers: bool = True
    """False for an injected event's delivery: its output is discarded."""
    strategy_turn: bool = False
    """A generation a strategy's runner asks: the channel answers, past its
    speak policy and its thinker (RFC §10.6.1)."""
    notes: list[str] = Field(default_factory=list)
    """The runtime's own blocks for the turn (a discussion's turn notes)."""
    in_place: bool = False
    """A discussion's turn: the event keeps its place in the history."""
    owner_tools: list[OwnerTool] = Field(default_factory=list)
    """The tools the owner serves for the channel in the room: the channel
    declares them to its model, and a call to one is a ``tool_call``."""
    trace: dict[str, str] | None = None


class ErrorRecord(BaseModel):
    """An exception as it crosses: its class's name, its message, what
    classifies it, and its causes in order (RFC §10.6.8)."""

    kind: str
    message: str
    args: list[Any] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    """The error's attributes whose values JSON carries; the others stay
    on the executor."""
    causes: list[ErrorRecord] = Field(default_factory=list)
    traceback: str | None = None
    """Where the error was raised on the executor, formatted."""


class WireOutput(BaseModel):
    """A ``ChannelOutput`` without its stream."""

    responded: bool = False
    response_events: list[RoomEvent] = Field(default_factory=list)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata_updates: dict[str, Any] = Field(default_factory=dict)
    error: ErrorRecord | None = None


# -- Frames: executor to owner ------------------------------------------------


class Accepted(BaseModel):
    type: Literal["accepted"] = "accepted"
    executor_id: str


class Heartbeat(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"


class Output(BaseModel):
    type: Literal["output"] = "output"
    output: WireOutput
    streams: bool = False


class Item(BaseModel):
    """One stream item; ``item["kind"]`` names it (see ``_codec``)."""

    type: Literal["item"] = "item"
    item: dict[str, Any]


class ToolArgs(BaseModel):
    """The arguments a call runs with once its gate fixed them."""

    type: Literal["tool_args"] = "tool_args"
    start: int
    arguments: dict[str, Any]


class ToolEnded(BaseModel):
    """A call's end, known before its end item was read."""

    type: Literal["tool_ended"] = "tool_ended"
    start: int
    end: dict[str, Any]


class Record(BaseModel):
    type: Literal["record"] = "record"
    record: dict[str, Any]


class Error(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorRecord


class End(BaseModel):
    type: Literal["end"] = "end"


# -- Calls: executor to owner, each answered once by its result --------------


class CallContext(BaseModel):
    """What a call's handler reads of the turn it serves (RFC §10.6.5, §21.4)."""

    tool_call_id: str
    room_id: str
    channel_id: str
    chain_depth: int = 0
    actor_id: str | None = None
    allowed_names: list[str] | None = None
    """The names the turn admits; ``None`` before its toolset was resolved."""
    requester: str | None = None


class ToolCall(BaseModel):
    """A call to a tool the owner serves: its handler runs there."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    """Unique on the link: two calls under one model id stay two calls."""
    name: str
    arguments: dict[str, Any]
    """The arguments the call's gate left."""
    context: CallContext


class CallCancel(BaseModel):
    """The executor withdraws a call: its turn was cut or its bound expired."""

    type: Literal["call_cancel"] = "call_cancel"
    call_id: str


# -- Control: owner to executor -----------------------------------------------


class Pull(BaseModel):
    type: Literal["pull"] = "pull"
    count: int = Field(default=1, ge=1)


class Close(BaseModel):
    type: Literal["close"] = "close"


class Cancel(BaseModel):
    type: Literal["cancel"] = "cancel"
    reason: str


class ToolResult(BaseModel):
    """The owner's answer to one ``tool_call``."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    outcome: Literal["served", "failed", "unserved"]
    """``unserved``: nothing on the owner serves the call (RFC §21.4)."""
    result: dict[str, Any] | None = None
    """The handler's answer as the model reads it, when served."""
    error: ErrorRecord | None = None
    structured_content: dict[str, Any] | None = None
    """The structured copy the handler wrote on its call (RFC §21.4)."""


Frame = Annotated[
    Accepted
    | Heartbeat
    | Output
    | Item
    | ToolArgs
    | ToolEnded
    | Record
    | Error
    | End
    | ToolCall
    | CallCancel,
    Field(discriminator="type"),
]
Control = Annotated[Pull | Close | Cancel | Heartbeat | ToolResult, Field(discriminator="type")]


class FrameEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Frame


class ControlEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Control
