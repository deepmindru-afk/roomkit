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
    notes: list[str] = Field(default_factory=list)
    """The runtime's own blocks for the turn (a discussion's turn notes)."""
    in_place: bool = False
    """A discussion's turn: the event keeps its place in the history."""
    trace: dict[str, str] | None = None


class ErrorRecord(BaseModel):
    """An exception as it crosses: its class's name, its message, what
    classifies it, and its causes in order (RFC §10.6.8)."""

    kind: str
    message: str
    args: list[Any] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    causes: list[ErrorRecord] = Field(default_factory=list)


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


# -- Control: owner to executor -----------------------------------------------


class Pull(BaseModel):
    type: Literal["pull"] = "pull"
    count: int = Field(default=1, ge=1)


class Close(BaseModel):
    type: Literal["close"] = "close"


class Cancel(BaseModel):
    type: Literal["cancel"] = "cancel"
    reason: str


Frame = Annotated[
    Accepted | Heartbeat | Output | Item | ToolArgs | ToolEnded | Record | Error | End,
    Field(discriminator="type"),
]
Control = Annotated[Pull | Close | Cancel | Heartbeat, Field(discriminator="type")]


class FrameEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Frame


class ControlEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Control
