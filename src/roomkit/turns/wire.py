"""The messages of a turn's link (RFC §10.6.3).

Every message a turn backend carries is one of these records, serialized to
JSON whatever the transport, the in-memory one included: no object of the turn
crosses between the room owner and the executor. A link carries an
:class:`Envelope` per message, numbered per direction, so a gap or a
duplicate is seen on arrival.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.steering import SteeringDirective
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AITool


class OwnerTool(BaseModel):
    """A tool the room owner serves a turn, as the request lists it (RFC §10.6.5)."""

    tool: AITool
    waits_by_design: bool = False
    """Bounded by its own bound, not the channel's default call bound (RFC §21.6)."""


class PreviousTurn(BaseModel):
    """The room's previous turn of the channel, the one the owner submitted
    last, ended or not (RFC §10.6.2); or its last channel call, which leaves
    the room's state as a turn does."""

    turn_id: str
    """The turn's id, or the channel call's."""
    executor_id: str | None = None
    """The executor that took it; ``None`` when none had said so yet."""


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
    room_hooks: list[str] = Field(default_factory=list)
    """The names of the room's hooks on the triggers the turn fires (RFC
    §10.6.6): the executor runs those it defines under them."""
    previous_turn: PreviousTurn | None = None
    """The room's previous turn of the channel; ``None`` for its first, or
    its first since the channel was detached from the room. An executor
    holding the room's state for the channel that this turn did not leave
    drops it before the turn runs (RFC §10.6.2)."""
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


class InputRequest(BaseModel):
    """A tool of the turn asked a human for input (RFC §10.6.5): sent before
    ``ON_USER_INPUT_REQUIRED`` fires, so the owner can answer it at once."""

    type: Literal["input_request"] = "input_request"
    request_id: str
    tool_name: str
    arguments: dict[str, Any]


class InputClosed(BaseModel):
    """A request settled on the executor, where its tool waits, by other than
    the owner's answer (an answer the channel's handler took, a hook's block,
    its bound, a release)."""

    type: Literal["input_closed"] = "input_closed"
    request_id: str
    reason: str


class InputTaken(BaseModel):
    """Whether the owner's ``input_answer`` settled its request: one settled
    first where its tool waits did not take it."""

    type: Literal["input_taken"] = "input_taken"
    request_id: str
    taken: bool


class Ephemeral(BaseModel):
    """An ephemeral event the turn published (thinking, a tool call's
    composition, a plan): the owner publishes it on its realtime backend
    (RFC §10.6.3)."""

    type: Literal["ephemeral"] = "ephemeral"
    channel: str
    """The realtime channel it was published to (``room:<room_id>``)."""
    event: dict[str, Any]
    """The event, as ``EphemeralEvent.to_dict()`` gives it."""


class FrameworkEventFrame(BaseModel):
    """A framework event the turn emitted (RFC §8.2): the owner emits it on
    its own bus (RFC §10.6.3)."""

    type: Literal["framework_event"] = "framework_event"
    event_type: str
    room_id: str | None = None
    channel_id: str | None = None
    event_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class SteerResult(BaseModel):
    """How many loops a ``steer`` reached; ``None`` for a channel without
    steering, whose owner then cuts the turn on a ``Cancel``."""

    type: Literal["steer_result"] = "steer_result"
    steer_id: str
    loops: int | None


# -- Calls: executor to owner, each answered once by its result --------------


class CallContext(BaseModel):
    """What a call's handler reads of the turn it serves (RFC §10.6.5, §21.4).

    The owner reads the room and the channel of the call from the request it
    submitted; a call that names others is refused. An intelligence turn
    serves no voice session, so none is carried.
    """

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


class HostCall(BaseModel):
    """A verb of the turn's ``TurnHost`` (``roomkit.turns.host.VERBS``), run
    on the owner with its arguments as data."""

    type: Literal["host_call"] = "host_call"
    call_id: str
    verb: str
    arguments: dict[str, Any]
    context: CallContext
    """What the verb reads of the turn: its chain depth, who asked; a call
    made from a hook names no tool call."""


class CallCancel(BaseModel):
    """The executor withdraws a call (a ``tool_call`` or a ``host_call``):
    its turn was cut or its bound expired."""

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


class Steer(BaseModel):
    """A steering directive for the turn's loop (RFC §21.3)."""

    type: Literal["steer"] = "steer"
    steer_id: str
    directive: SteeringDirective = Field(discriminator="type")


class InputAnswer(BaseModel):
    """The outcome the owner recorded for an ``input_request``."""

    type: Literal["input_answer"] = "input_answer"
    request_id: str
    outcome: Literal["answered", "rejected"]
    value: str | None = None
    reason: str | None = None


class ToolResult(BaseModel):
    """The owner's answer to one ``tool_call``."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    outcome: Literal["served", "refused", "failed", "unserved"]
    """``refused``: the handler refused the call (``ToolRefusedError``);
    ``unserved``: nothing on the owner serves it (RFC §21.4)."""
    result: dict[str, Any] | None = None
    """The handler's answer as the model reads it, when served."""
    error: ErrorRecord | None = None
    structured_content: dict[str, Any] | None = None
    """The structured copy the handler wrote on its call (RFC §21.4)."""


class HostResult(BaseModel):
    """The owner's answer to one ``host_call``: the verb's result as data,
    or what it raised."""

    type: Literal["host_result"] = "host_result"
    call_id: str
    result: Any = None
    error: ErrorRecord | None = None


# -- Channel control: owner to executor, outside any turn ----------------------


class RoomAttached(BaseModel):
    """The channel was attached to the room (RFC §10.6.3)."""

    type: Literal["room_attached"] = "room_attached"
    channel_id: str
    room_id: str
    binding: ChannelBinding


class RoomDetached(BaseModel):
    """The channel was detached from the room: its state for the room is
    dropped (RFC §10.6.3)."""

    type: Literal["room_detached"] = "room_detached"
    channel_id: str
    room_id: str


class CommittedEvent(BaseModel):
    """An event committed in the room that the channel's memory is handed (a
    discussion's, RFC §19.7.5 rule 3), answered by nothing (RFC §10.6.3)."""

    type: Literal["committed_event"] = "committed_event"
    channel_id: str
    room_id: str
    event: RoomEvent
    room_hooks: list[str] = Field(default_factory=list)
    """The room's hooks on executor triggers, as a turn's request names them:
    the event goes to an executor that would take the room's turns."""


class ChannelCall(BaseModel):
    """A per-room verb of the channel (an ACP session's ``set_config_option``),
    answered by a :class:`ChannelResult` (RFC §10.6.3)."""

    type: Literal["channel_call"] = "channel_call"
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex}")
    channel_id: str
    room_id: str
    verb: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    room_state: dict[str, Any] = Field(default_factory=dict)
    """What the room keeps for the channel, as the owner stores it: the
    channel takes it back when it holds none for the room."""
    previous_turn: PreviousTurn | None = None
    """The room's previous turn or call of the channel: the executor drops a
    state it did not leave before the verb runs, as before a turn."""


class ChannelResult(BaseModel):
    """The executor's answer to one ``channel_call``: the verb's result as
    data, or what it raised."""

    type: Literal["channel_result"] = "channel_result"
    result: Any = None
    error: ErrorRecord | None = None
    executor_id: str | None = None
    """The executor that ran the verb."""
    room_state: dict[str, Any] | None = None
    """What the room keeps for the channel once the verb ran, when it
    changed (empty: nothing); ``None`` when it did not."""


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
    | HostCall
    | CallCancel
    | SteerResult
    | InputRequest
    | InputClosed
    | InputTaken
    | Ephemeral
    | FrameworkEventFrame,
    Field(discriminator="type"),
]
Control = Annotated[
    Pull | Close | Cancel | Heartbeat | Steer | InputAnswer | ToolResult | HostResult,
    Field(discriminator="type"),
]


ChannelControl = Annotated[
    RoomAttached | RoomDetached | CommittedEvent | ChannelCall,
    Field(discriminator="type"),
]


class ChannelControlEnvelope(BaseModel):
    message: ChannelControl


# -- Outside a turn: executor to owner ---------------------------------------


class RoomCall(BaseModel):
    """A verb of a turn's host made after its turn ended (RFC §10.6.3,
    §10.6.5), answered by a :class:`HostResult`: the turn it belongs to names
    the room, and its chain depth bounds what it sends."""

    type: Literal["room_call"] = "room_call"
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex}")
    channel_id: str
    room_id: str
    executor_id: str
    turn_id: str
    """The turn the work making the call began in."""
    chain_depth: int = 0
    """The chain depth of that turn's response."""
    verb: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RoomPublication(BaseModel):
    """A realtime publication or a framework event made after its turn ended
    (RFC §10.6.3), answered by nothing."""

    type: Literal["room_publication"] = "room_publication"
    channel_id: str
    room_id: str
    executor_id: str
    turn_id: str
    frame: Annotated[Ephemeral | FrameworkEventFrame, Field(discriminator="type")]


OwnerMessage = Annotated[RoomCall | RoomPublication, Field(discriminator="type")]


class OwnerMessageEnvelope(BaseModel):
    message: OwnerMessage


class FrameEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Frame


class ControlEnvelope(BaseModel):
    turn_id: str
    seq: int
    message: Control
