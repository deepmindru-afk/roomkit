"""A host's error classes across the turn backend (RFC §10.6.8).

The classes a host declares come back as themselves, attributes and causes
kept, on every path an error crosses, read with the receiving side's
declaration. Any other comes back as a ``RemoteTurnError`` that keeps its
JSON attributes and reads as its nearest ancestor that side rebuilds. A name
read off the wire never imports anything.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels._tool_registry import _TURN_OWNER_TOOLS
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import RemoteTurnError, error_type_name
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.task import DelegatedTaskResult
from roomkit.providers.ai.base import AIContext, AITool, ProviderError, StreamEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import ToolCallContext, tool_turn_context
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, current_turn_host
from roomkit.turns._error_codec import ErrorClasses, error_record
from roomkit.turns.base import OwnerHost, OwnerTools
from roomkit.turns.wire import ErrorRecord, OwnerTool, RoomCall, ToolCall, TurnRequest
from tests.test_framework import SimpleChannel
from tests.test_turns_backend import ScriptedChannel, _read_all, _run
from tests.turn_errors import same_error


class HostProviderError(ProviderError):
    """A host's classified provider failure, as an application defines one."""

    def __init__(self, message: str, *, code: str, answered: bool = True) -> None:
        super().__init__(message, provider="vendor", status_code=402)
        self.code = code
        self._answered = answered

    @property
    def answered(self) -> bool:
        return self._answered


class HostTransportError(Exception):
    """A transport library's base error, which a host declares."""


class HostReadTimeoutError(HostTransportError):
    """One of that library's errors, declared through its base."""


class HostQuotaError(TimeoutError):
    """A host error deriving from one of Python's."""


class FormattedError(Exception):
    """An application error whose message is built from its arguments."""

    def __str__(self) -> str:
        return f"app error {self.args[0]}"


class RefusingBaseError(ValueError):
    """A declared base that refuses any subclass it did not register."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise ValueError(f"unregistered subclass {cls.__name__}")


class Billing:
    """A host's classes nested in a class, as Django's ``Model.DoesNotExist``."""

    class QuotaExceededError(Exception):
        def __init__(self, code: str) -> None:
            super().__init__(f"quota exceeded: {code}")
            self.code = code

    class PlanQuotaExceededError(QuotaExceededError):
        """Undeclared: read as the declared class it derives from."""


class _Lazy(types.ModuleType):
    """A module whose ``__getattr__`` would import what a name asks for."""

    def __init__(self) -> None:
        super().__init__("rmk838_lazy")
        self.asked: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.asked.append(name)
        raise AttributeError(name)


class _LazyProxy:
    """A module-level object that evaluates what it wraps when its class is
    read, as ``django.conf.settings`` or werkzeug's ``LocalProxy`` do."""

    def __init__(self, load: Callable[[], Any]) -> None:
        self._load = load

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return type(self._load())


def _host_error() -> HostProviderError:
    error = HostProviderError("payment required", code="payment_required", answered=False)
    error.__cause__ = HostReadTimeoutError("read timed out")
    return error


def _crossed(error: BaseException, *declared: type[Exception]) -> BaseException:
    return ErrorClasses(declared).rebuild(error_record(error))


# -- What a side rebuilds -------------------------------------------------------


def test_a_declared_class_comes_back_as_itself_with_its_attributes() -> None:
    raised = _host_error()
    rebuilt = _crossed(raised, HostProviderError, HostTransportError)
    assert same_error(rebuilt, raised)
    assert isinstance(rebuilt, HostProviderError)
    assert (rebuilt.code, rebuilt.answered, rebuilt.status_code) == (
        "payment_required",
        False,
        402,
    )


def test_a_subclass_of_a_declared_class_is_rebuilt_by_type() -> None:
    rebuilt = _crossed(HostReadTimeoutError("read timed out"), HostTransportError)
    assert type(rebuilt) is HostReadTimeoutError


def test_a_declared_class_nested_in_a_class_comes_back_as_itself() -> None:
    rebuilt = _crossed(Billing.QuotaExceededError("Q1"), Billing.QuotaExceededError)
    assert type(rebuilt) is Billing.QuotaExceededError
    assert (str(rebuilt), rebuilt.code) == ("quota exceeded: Q1", "Q1")


def test_a_declared_class_defined_in_a_function_comes_back_as_itself() -> None:
    class LocalError(Exception):
        pass

    rebuilt = _crossed(LocalError("x"), LocalError)
    assert type(rebuilt) is LocalError


def test_an_undeclared_subclass_of_a_declared_nested_class_reads_as_it() -> None:
    rebuilt = _crossed(Billing.PlanQuotaExceededError("Q2"), Billing.QuotaExceededError)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, Billing.QuotaExceededError)
    assert rebuilt.code == "Q2"


def test_a_dotted_name_off_the_wire_reaches_no_undeclared_nested_class() -> None:
    rebuilt = _crossed(Billing.QuotaExceededError("Q1"))
    assert type(rebuilt) is RemoteTurnError


def test_a_declared_class_in_the_cause_chain_comes_back_as_itself() -> None:
    rebuilt = _crossed(_host_error(), HostTransportError)
    assert isinstance(rebuilt, RemoteTurnError)
    assert type(rebuilt.__cause__) is HostReadTimeoutError
    assert str(rebuilt.__cause__) == "read timed out"


def test_an_undeclared_provider_error_subclass_reads_as_a_provider_error() -> None:
    raised = _host_error()
    rebuilt = _crossed(raised)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, ProviderError)
    assert rebuilt.kind.endswith(":HostProviderError")
    assert rebuilt.bases[0] == "roomkit.providers.ai.base:ProviderError"
    assert str(rebuilt) == str(raised)
    assert error_type_name(rebuilt) == "HostProviderError"
    assert (rebuilt.code, rebuilt.status_code, rebuilt.retryable) == (
        "payment_required",
        402,
        False,
    )
    assert isinstance(rebuilt.__cause__, RemoteTurnError)
    assert not isinstance(rebuilt.__cause__, HostTransportError)


def _raised(work: Callable[[], object]) -> Exception:
    try:
        work()
    except Exception as exc:
        return exc
    raise AssertionError("did not raise")


# Python's errors whose message lives in fields their ``__init__`` sets.
BUILT_IN_MESSAGES: dict[str, Callable[[], object]] = {
    "syntax_error": lambda: compile("def (", "<turn>", "exec"),
    "indentation_error": lambda: compile("if x:\npass", "<turn>", "exec"),
    "unicode_decode_error": lambda: b"\xff".decode(),
    "unicode_encode_error": lambda: chr(233).encode("ascii"),
    "file_not_found": lambda: open("/nonexistent/roomkit-turn"),  # noqa: SIM115
    "two_filenames": lambda: os.rename("/nonexistent/roomkit-a", "/nonexistent/roomkit-b"),
}


@pytest.mark.parametrize("case", BUILT_IN_MESSAGES)
def test_a_built_in_error_keeps_the_message_its_fields_make(case: str) -> None:
    raised = _raised(BUILT_IN_MESSAGES[case])
    wire = error_record(raised).model_dump_json()
    rebuilt = ErrorClasses().rebuild(ErrorRecord.model_validate_json(wire))
    assert isinstance(rebuilt, type(raised))
    assert str(rebuilt) == str(raised)


def test_a_built_in_error_built_from_its_arguments_keeps_its_fields() -> None:
    rebuilt = _crossed(_raised(BUILT_IN_MESSAGES["syntax_error"]))
    assert type(rebuilt) is SyntaxError
    assert (rebuilt.filename, rebuilt.lineno) == ("<turn>", 1)


def test_an_os_error_keeps_its_filenames_and_its_errno() -> None:
    rebuilt = _crossed(_raised(BUILT_IN_MESSAGES["two_filenames"]))
    assert type(rebuilt) is FileNotFoundError
    assert (rebuilt.errno, rebuilt.filename, rebuilt.filename2) == (
        2,
        "/nonexistent/roomkit-a",
        "/nonexistent/roomkit-b",
    )


def test_a_declared_class_that_cannot_give_its_message_back_still_reads_as_itself() -> None:
    class DetailedError(Exception):
        def __init__(self, response: object) -> None:
            super().__init__("detailed")
            self.response = response

        def __str__(self) -> str:
            return f"failed: {self.response!r}"

    raised = DetailedError(object())  # JSON carries no response
    rebuilt = _crossed(raised, DetailedError)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, DetailedError)
    assert str(rebuilt) == str(raised)


def test_an_undeclared_class_reads_as_its_nearest_python_ancestor() -> None:
    rebuilt = _crossed(HostQuotaError("over quota"))
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, TimeoutError)
    assert str(rebuilt) == "over quota"


def test_a_class_this_side_has_not_loaded_reads_as_a_declared_ancestor() -> None:
    record = ErrorRecord(
        kind="rmk838.not_loaded:ReadTimeout",
        message="read timed out",
        attributes={"retries": 3},
        bases=[f"{HostTransportError.__module__}:HostTransportError"],
    )
    rebuilt = ErrorClasses((HostTransportError,)).rebuild(record)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, HostTransportError)
    assert rebuilt.retries == 3  # type: ignore[attr-defined]


def test_an_ancestor_that_refuses_a_message_alone_is_passed_over() -> None:
    record = ErrorRecord(
        kind="rmk838.not_loaded:Grouped",
        message="several failed",
        bases=["builtins:ExceptionGroup", "builtins:ValueError"],
    )
    rebuilt = ErrorClasses().rebuild(record)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, ValueError)
    assert not isinstance(rebuilt, ExceptionGroup)


def test_an_error_rebuilt_as_remote_crosses_again_as_the_error_it_stands_for() -> None:
    raised = _host_error()
    on_owner = _crossed(raised)
    assert isinstance(on_owner, RemoteTurnError)

    on_executor = _crossed(on_owner, HostProviderError)

    assert type(on_executor) is HostProviderError
    assert on_executor.code == "payment_required"
    assert str(on_executor) == str(raised)
    again = _crossed(on_owner)
    assert isinstance(again, ProviderError)
    assert (again.kind, again.bases) == (on_owner.kind, on_owner.bases)


def test_an_error_crossing_twice_keeps_its_arguments_and_attributes() -> None:
    raised = FormattedError("X")
    raised.kind = "quota"  # type: ignore[attr-defined]
    raised.code = 7  # type: ignore[attr-defined]
    on_owner = _crossed(raised)
    assert str(on_owner) == "app error X"
    on_owner.reported = True  # type: ignore[attr-defined]

    on_executor = _crossed(on_owner, FormattedError)

    assert type(on_executor) is FormattedError
    assert on_executor.args == ("X",)
    assert str(on_executor) == "app error X"
    attributes = {k: v for k, v in vars(on_executor).items() if k != "__notes__"}
    assert attributes == {"kind": "quota", "code": 7, "reported": True}


def test_an_ancestor_whose_class_cannot_be_made_is_passed_over() -> None:
    record = ErrorRecord(
        kind="rmk838.not_loaded:Leaf",
        message="m",
        bases=[f"{RefusingBaseError.__module__}:RefusingBaseError", "builtins:ValueError"],
    )
    rebuilt = ErrorClasses((RefusingBaseError,)).rebuild(record)
    assert isinstance(rebuilt, RemoteTurnError)
    assert isinstance(rebuilt, ValueError)
    assert not isinstance(rebuilt, RefusingBaseError)


def test_a_frame_naming_the_remote_error_itself_crosses_again() -> None:
    record = ErrorRecord(kind="roomkit.core.exceptions:RemoteTurnError", message="m", args=["m"])
    rebuilt = ErrorClasses().rebuild(record)
    assert error_record(rebuilt).message == "m"
    assert error_type_name(rebuilt) == "RemoteTurnError"


def test_an_attribute_named_like_a_method_does_not_break_the_rebuild() -> None:
    record = ErrorRecord(
        kind="builtins:ValueError",
        message="m",
        args=["m"],
        attributes={"add_note": 1},
        traceback="tb",
    )
    rebuilt = ErrorClasses().rebuild(record)
    assert type(rebuilt) is ValueError
    assert any("tb" in note for note in rebuilt.__notes__)


def test_a_loaded_class_named_in_a_frame_and_not_declared_is_not_rebuilt() -> None:
    rebuilt = _crossed(HostReadTimeoutError("read timed out"), HostProviderError)
    assert type(rebuilt) is not HostReadTimeoutError
    assert isinstance(rebuilt, RemoteTurnError)


def test_a_name_off_the_wire_runs_no_module_code_and_imports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lazy = _Lazy()
    monkeypatch.setitem(sys.modules, "rmk838_lazy", lazy)
    absent = "rmk838_never_imported"
    record = ErrorRecord(
        kind="rmk838_lazy:Anything",
        message="m",
        bases=[f"{absent}:Base", "rmk838_lazy:Other"],
        causes=[ErrorRecord(kind=f"{absent}.sub:Cause", message="c")],
    )

    rebuilt = ErrorClasses((Exception,)).rebuild(record)

    assert type(rebuilt) is RemoteTurnError
    assert lazy.asked == []
    assert absent not in sys.modules
    assert f"{absent}.sub" not in sys.modules


def test_a_name_off_the_wire_does_not_load_a_lazily_imported_module(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = tmp_path / "ran"
    source = tmp_path / "rmk838_lazily_loaded.py"
    source.write_text(
        f"import pathlib\npathlib.Path({str(ran)!r}).write_text('ran')\n"
        "class Boom(Exception):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location("rmk838_lazily_loaded", source)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "rmk838_lazily_loaded", module)
    record = ErrorRecord(kind="rmk838_lazily_loaded:Boom", message="m")

    rebuilt = ErrorClasses((Exception,)).rebuild(record)

    assert type(rebuilt) is RemoteTurnError
    assert not ran.exists()


def test_a_name_off_the_wire_does_not_evaluate_a_lazy_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []
    module = types.ModuleType("rmk854_settings")
    module.settings = _LazyProxy(lambda: loaded.append("settings") or ValueError())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rmk854_settings", module)
    record = ErrorRecord(
        kind="rmk854_settings:settings", message="m", bases=["rmk854_settings:settings"]
    )

    rebuilt = ErrorClasses((Exception,)).rebuild(record)

    assert type(rebuilt) is RemoteTurnError
    assert loaded == []


async def test_a_declaration_given_once_reaches_both_sides_of_the_kit() -> None:
    raised = _host_error()
    once = (cls for cls in (HostProviderError, HostTransportError))

    result, _ = await _inbound(RoomKit(error_classes=once), raised)

    assert type(result.error) is HostProviderError
    assert type(result.error.__cause__) is HostReadTimeoutError


@pytest.mark.parametrize("entry", ["httpx", KeyboardInterrupt, object, ValueError("x")])
def test_a_declaration_takes_exception_classes_only(entry: Any) -> None:
    with pytest.raises(TypeError, match="Exception classes"):
        ErrorClasses((entry,))
    with pytest.raises(TypeError, match="Exception classes"):
        RoomKit(error_classes=(entry,))
    with pytest.raises(TypeError, match="Exception classes"):
        TurnExecutor({}.get, error_classes=(entry,))


# -- Every path an error crosses ------------------------------------------------

PROBE = AITool(name="probe", description="Probe.", parameters={"type": "object"})

Act = Callable[["_Acting"], Awaitable[ChannelOutput]]


class _Acting(ScriptedChannel):
    """An intelligence channel whose turn runs *act* on its executor."""

    def __init__(self, act: Act) -> None:
        super().__init__()
        self.act = act
        self.caught: BaseException | None = None
        self.later: asyncio.Future[BaseException | None] | None = None

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return await self.act(self)

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        raise self.raises or RuntimeError("no error set")


class _RaisingHost(OwnerHost):
    """The owner's host verbs, each raising *raised*; ``wait_task`` answers
    a task that failed with it."""

    def __init__(self, raised: Exception) -> None:
        self.raised = raised

    async def call(self, request: TurnRequest, verb: str, arguments: dict[str, Any]) -> Any:
        if verb != "wait_task":
            raise self.raised
        result = DelegatedTaskResult(
            task_id="t1", child_room_id="child", parent_room_id="room", agent_id="worker"
        )
        return {
            "result": result.model_dump(mode="json"),
            "exception": error_record(self.raised).model_dump(mode="json"),
        }

    async def call_between_turns(self, message: RoomCall) -> Any:
        raise self.raised


class _RaisingTools(OwnerTools):
    """The owner serves ``probe``, which raises *raised*."""

    def __init__(self, raised: Exception) -> None:
        self.raised = raised

    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        return [OwnerTool(tool=PROBE)]

    async def serve(self, request: TurnRequest, call: ToolCall) -> Any:
        raise self.raised


def _client(
    channel: ScriptedChannel,
    *,
    owner: tuple[type[Exception], ...] = (),
    executor: tuple[type[Exception], ...] = (),
    raised: Exception | None = None,
) -> TurnClient:
    """A client and its executor, each declaring its own classes; the
    owner's host verbs and tools raise *raised*."""
    channels = {channel.channel_id: channel}
    return TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__, error_classes=executor),
        host=None if raised is None else _RaisingHost(raised),
        tools=None if raised is None else _RaisingTools(raised),
        error_classes=owner,
    )


async def _caught(work: Coroutine[Any, Any, Any] | asyncio.Future[Any]) -> BaseException | None:
    try:
        await work
    except Exception as exc:
        return exc
    return None


async def _turn_output(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the owner: the turn's ``on_event`` raised."""
    return await _caught(_run(_client(ScriptedChannel(raises=raised), owner=declared)))


async def _turn_stream(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the owner: the turn's stream raised after an item."""

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "partial"
        raise raised

    output = await _run(_client(ScriptedChannel(script), owner=declared))
    return await _caught(_read_all(output))


async def _output_error(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the owner: the turn's output carried an error."""

    async def act(channel: _Acting) -> ChannelOutput:
        return ChannelOutput(error=raised)

    return (await _run(_client(_Acting(act), owner=declared))).error


async def _kit_channel_verb(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the owner (a kit): a channel's per-room verb raised."""

    async def act(channel: _Acting) -> ChannelOutput:
        return ChannelOutput.empty()

    channel = _Acting(act)
    channel.raises = raised
    kit = RoomKit(error_classes=declared)
    kit.register_channel(channel)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    try:
        return await _caught(kit.channel_call("agent", "room", "probe"))
    finally:
        await kit.close()


async def _on_executor(
    act: Act, declared: tuple[type[Exception], ...], raised: Exception
) -> BaseException | None:
    """What *act* caught on an executor declaring *declared*, the owner
    declaring nothing."""
    channel = _Acting(act)
    await _run(_client(channel, executor=declared, raised=raised))
    if channel.later is not None:
        return await channel.later
    return channel.caught


async def _host_verb(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the executor: a verb of the turn's host raised on the owner."""

    async def act(channel: _Acting) -> ChannelOutput:
        host = current_turn_host()
        assert host is not None
        channel.caught = await _caught(host.post_status("agent", "probe", "running"))
        return ChannelOutput.empty()

    return await _on_executor(act, declared, raised)


async def _owner_tool(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the executor: a tool the owner serves raised there."""

    async def act(channel: _Acting) -> ChannelOutput:
        tools = _TURN_OWNER_TOOLS.get()
        assert tools is not None
        serve = tools.entries["probe"].serve
        assert serve is not None
        with tool_turn_context(room_id="room", call=ToolCallContext("room", "c1", "agent")):
            channel.caught = await _caught(serve({}))
        return ChannelOutput.empty()

    return await _on_executor(act, declared, raised)


async def _between_turns(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the executor: a host verb sent after the turn ended raised on the owner."""

    async def act(channel: _Acting) -> ChannelOutput:
        host = current_turn_host()
        assert host is not None

        async def after_the_turn() -> BaseException | None:
            while host.open:
                await asyncio.sleep(0.01)
            return await _caught(host.post_status("agent", "probe", "done"))

        channel.later = asyncio.ensure_future(after_the_turn())
        return ChannelOutput.empty()

    return await _on_executor(act, declared, raised)


async def _delegated_task(declared: tuple[type[Exception], ...], raised: Exception) -> Any:
    """On the executor: the failure of a task the owner ran, read by ``wait_task``."""

    async def act(channel: _Acting) -> ChannelOutput:
        host = current_turn_host()
        assert host is not None
        result = await host.wait_task("t1")
        assert result is not None
        channel.caught = result.exception
        return ChannelOutput.empty()

    return await _on_executor(act, declared, raised)


PATHS = {
    "turn_output": _turn_output,
    "turn_stream": _turn_stream,
    "output_error": _output_error,
    "kit_channel_verb": _kit_channel_verb,
    "host_verb": _host_verb,
    "owner_tool": _owner_tool,
    "between_turns": _between_turns,
    "delegated_task": _delegated_task,
}


@pytest.mark.parametrize("path", PATHS)
async def test_a_declared_class_comes_back_as_itself_on_every_path(path: str) -> None:
    received = await PATHS[path]((HostProviderError, HostTransportError), _host_error())

    assert type(received) is HostProviderError
    assert (received.code, received.answered) == ("payment_required", False)
    assert type(received.__cause__) is HostReadTimeoutError


@pytest.mark.parametrize("path", PATHS)
async def test_an_undeclared_class_keeps_its_attributes_and_ancestor_on_every_path(
    path: str,
) -> None:
    raised = _host_error()

    received = await PATHS[path]((), raised)

    assert isinstance(received, RemoteTurnError)
    assert isinstance(received, ProviderError)
    assert str(received) == str(raised)
    assert (received.code, received.status_code) == ("payment_required", 402)


# -- What the room owner reads ----------------------------------------------------


class _Failing(MockAIProvider):
    """A streaming model whose stream raises *raised* before any item."""

    def __init__(self, raised: Exception) -> None:
        super().__init__(streaming=True)
        self.raised = raised

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        raise self.raised
        yield  # pragma: no cover - keeps this an async generator


async def _inbound(kit: RoomKit, raised: Exception) -> tuple[Any, list[RoomEvent]]:
    """One message to a room whose agent's stream raises *raised*: the
    inbound result, and the ``ON_ERROR`` events."""
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=_Failing(raised)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    errors: list[RoomEvent] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: RoomEvent, context: RoomContext) -> None:
        errors.append(event)

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)  # ON_ERROR runs after the room lock is released
    await kit.close()
    return result, errors


async def test_a_declared_class_a_turn_raised_reaches_the_inbound_result() -> None:
    raised = _host_error()

    result, errors = await _inbound(RoomKit(error_classes=(HostProviderError,)), raised)

    assert same_error(result.error, raised)
    assert result.error.answered is False
    assert [e.metadata["error_type"] for e in errors] == ["HostProviderError"]


async def test_a_declared_nested_class_a_turn_raised_reaches_the_inbound_result() -> None:
    result, _ = await _inbound(
        RoomKit(error_classes=(Billing.QuotaExceededError,)), Billing.QuotaExceededError("Q1")
    )

    assert type(result.error) is Billing.QuotaExceededError
    assert result.error.code == "Q1"


async def test_an_undeclared_provider_error_subclass_is_logged_as_a_provider_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raised = _host_error()

    with caplog.at_level(logging.DEBUG, logger="roomkit"):
        result, errors = await _inbound(RoomKit(), raised)

    assert isinstance(result.error, ProviderError)
    assert result.error.code == "payment_required"
    assert [e.metadata["error_type"] for e in errors] == ["HostProviderError"]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]
