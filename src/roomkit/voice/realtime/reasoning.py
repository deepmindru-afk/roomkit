"""Reasoning delegation backends for full-duplex providers (RFC §12.4.1).

A full-duplex model (OpenAI GPT-Live) holds the conversation and hands
reasoning and tool use to a backend. In the integrator mode that backend is
the application's: :class:`~roomkit.channels.realtime_voice.RealtimeVoiceChannel`
serves each delegation through a :class:`ReasoningBackend`, hands it the
transcript recorded since the previous one — the model sends no task text —
and returns every output to the model as spoken or silent context.

:class:`AgentReasoningBackend` serves the delegations with an agent like any
other, on the AI channel's tool loop, its calls executed through the channel's
own gate; :class:`AIProviderReasoningBackend` builds that agent from an
:class:`~roomkit.providers.ai.base.AIProvider`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from roomkit.channels._tool_eviction import REREAD_TOOL
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import RoomKitError, ToolRefusedError
from roomkit.models.channel import ChannelBinding
from roomkit.models.streaming import LoopEndMarker, ToolCallStartMarker
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext, current_tool_call
from roomkit.tools.result import tool_failure

if TYPE_CHECKING:
    from roomkit.providers.ai.base import AIProvider
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.voice.realtime.reasoning")

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
"""``(name, arguments) -> result`` — runs one tool call through the channel's gate."""


@dataclass(frozen=True)
class ToolCallResult:
    """One tool call's outcome, as a backend's model reads it (RFC §12.4.1).

    Attributes:
        text: What the model reads: the result, or why the call failed.
        is_error: The call was refused, failed, blocked, served by nothing or
            cancelled, as every tool loop marks such a call (RFC §9.3).
    """

    text: str
    is_error: bool = False


ToolCallExecutor = Callable[[str, dict[str, Any]], Awaitable[ToolCallResult]]
"""``(name, arguments) -> ToolCallResult`` — the same call, with its outcome."""

RefusalReporter = Callable[..., Awaitable[None]]
"""``(name, arguments, body, *, cancelled=False)`` — reports a call the backend
refused before the channel's gate to its ON_TOOL_CALL observers."""


class ReasoningCutShortError(RoomKitError):
    """A backend's turn ended before its answer: the round cap, the deadline or
    the budget cut it. The channel answers the delegation with its spoken
    fallback (RFC §12.4.1)."""

    def __init__(self, delegation_id: str, reason: str | None) -> None:
        super().__init__(f"Reasoning turn for delegation {delegation_id} ended {reason}")
        self.delegation_id = delegation_id
        self.reason = reason


DEFAULT_TRANSCRIPT_INSTRUCTION = "Act on the user's most recent request in the conversation above."


@dataclass(frozen=True)
class TranscriptLine:
    """One speaker's contribution in the transcript handed to a backend."""

    role: Literal["user", "assistant"]
    text: str


@dataclass
class ReasoningRequest:
    """What a backend receives for one delegation (RFC §12.4.1).

    Attributes:
        session: The voice session whose model delegated.
        delegation_id: Opaque provider identifier; returned unchanged with
            every output.
        transcript: User and assistant lines recorded since the previous
            request — the backend's only account of what was asked.
        first: Whether this is the session's first request, in which case
            the transcript is the whole conversation so far.
        tools: The channel's declared tool catalogue, as tool dicts. The
            backend offers these to its model.
        execute_tool: Runs one tool call through the channel's pre-execution
            gate (declared catalogue, argument schema, skill gating,
            ``BEFORE_TOOL_USE``, ``ON_TOOL_CALL``) and returns the result
            text. A backend MUST route its tool calls through it.
        execute_tool_call: The same call, returning a :class:`ToolCallResult`
            that also says whether it failed, so the backend's model reads a
            refused or failed call as one. A backend SHOULD prefer it.
        report_refusal: Reports a call the backend's own loop refused before
            the gate (its arguments did not read, a stop cut it) to the
            channel's ON_TOOL_CALL observers, as ``(name, arguments, body,
            cancelled=...)``: a backend's call is reported wherever it ends.
    """

    session: VoiceSession
    delegation_id: str
    transcript: list[TranscriptLine]
    first: bool
    tools: list[dict[str, Any]] = field(default_factory=list)
    execute_tool: ToolExecutor | None = None
    execute_tool_call: ToolCallExecutor | None = None
    report_refusal: RefusalReporter | None = None


@dataclass(frozen=True)
class ReasoningOutput:
    """One piece of a backend's output, on its way to the model.

    Attributes:
        text: What the backend produced.
        spoken: ``True`` asks the model to relay it to the user in its own
            words; ``False`` adds it as silent context the model may draw on.
        is_final: The answer to the request, as opposed to progress toward
            it.
    """

    text: str
    spoken: bool = True
    is_final: bool = False


class ReasoningBackend(ABC):
    """The backend a full-duplex model's integrator-side delegation goes to.

    Implementations work out the request from ``request.transcript``, answer
    it with their own model, context and tools, and yield
    :class:`ReasoningOutput` as they go. Tool calls MUST go through
    ``request.execute_tool`` so the channel's gate applies (RFC §12.4.1).
    """

    @abstractmethod
    def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        """Serve one delegation, yielding outputs as they are produced."""
        ...

    async def session_ended(self, session_id: str) -> None:  # noqa: B027
        """Release any state kept for a session that just ended."""

    async def close(self) -> None:  # noqa: B027
        """Release all resources."""

    def _adopt_telemetry(self, telemetry: TelemetryProvider) -> None:  # noqa: B027
        """Take the voice channel's telemetry for the spans of the backend's
        own turns; a backend that opens none has nothing to do with it."""


def render_transcript_request(
    transcript: list[TranscriptLine],
    *,
    first: bool,
    instruction: str = DEFAULT_TRANSCRIPT_INSTRUCTION,
) -> str:
    """Render the transcript as one labelled message for a text model.

    Flattening the conversation into one user message keeps the two
    conversations apart: the backend's own context holds only what the
    backend itself said as assistant messages, so it never mistakes the
    voice model's speech for its own.
    """
    lines: list[str] = []
    if transcript:
        lines.append(
            "Voice conversation so far:"
            if first
            else "Voice conversation since the previous delegation:"
        )
        lines.extend(f"{line.role.upper()}: {line.text}" for line in transcript if line.text)
        lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


_DELEGATION: ContextVar[ReasoningRequest | None] = ContextVar("_DELEGATION", default=None)
"""The delegation an agent backend is serving, read by its tool handler."""


class AgentReasoningBackend(ReasoningBackend):
    """A backend that is an agent like any other, driven by the voice (RFC §12.4.1).

    Each delegation runs on the agent's tool loop, the AI channel's, with
    everything that loop does: its round cap and timeout, its retry of an
    empty answer and of a call the provider could not parse, its refusal of a
    call whose arguments do not read, its span and its usage (RFC §6.4). The
    tools it offers are the voice session's catalogue, each call served
    through the voice channel's gate, which reports it; the agent's own tools,
    skills or sandbox would bypass that gate, so an agent carrying any is
    refused. The agent is the backend's: its tool handler and its refusal
    reports are taken over, so it serves no room besides.

    Each request becomes one user message carrying the transcript. Text the
    model writes before a tool round is yielded as progress, silent by default
    and spoken with ``spoken_progress=True``; its final answer is yielded
    spoken. A turn cut short (the round cap, the deadline, the budget) has no
    answer, and its narration is not passed off as one: the run raises
    :class:`ReasoningCutShortError`, as a provider error raises its own, and
    the channel's spoken fallback answers. The backend keeps its own conversation per voice
    session, so a later delegation sees what it worked out for an earlier one.

    Example:
        backend = AgentReasoningBackend(
            Agent("reasoner", provider=OpenAIAIProvider(...), system_prompt=PROMPT)
        )
        channel = RealtimeVoiceChannel(
            "voice", provider=live, transport=transport,
            tools=[check_flight, rebook_flight], reasoning_backend=backend,
        )
    """

    def __init__(self, agent: AIChannel, *, spoken_progress: bool = False) -> None:
        _refuse_own_tools(agent)
        self._agent = agent
        self._spoken_progress = spoken_progress
        self._histories: dict[str, list[AIMessage]] = {}
        # The session's calls go to the voice channel's gate; what the loop
        # refuses before them is reported there too.
        agent.tool_handler = self._serve_through_gate
        agent._tool_observer_hook = self._report_loop_refusal

    @property
    def agent(self) -> AIChannel:
        """The agent serving the delegations."""
        return self._agent

    @property
    def provider(self) -> AIProvider:
        """The model behind this backend."""
        return self._agent.provider

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        token = _DELEGATION.set(request)
        try:
            async for output in self._answer(request):
                yield output
        finally:
            _DELEGATION.reset(token)

    async def _answer(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        """Run the delegation on the agent's loop: progress, then the answer."""
        session_id = request.session.id
        messages = [
            *self._histories.get(session_id, []),
            AIMessage(
                role="user",
                content=render_transcript_request(request.transcript, first=request.first),
            ),
        ]
        context, loop_ctx = self._turn_context(request, messages)
        text: list[str] = []
        reason: str | None = None
        try:
            async for delta in self._agent._run_streaming_tool_loop(
                context, parent_loop_ctx=loop_ctx
            ):
                if isinstance(delta, str):
                    text.append(delta)
                elif isinstance(delta, ToolCallStartMarker):
                    if progress := "".join(text).strip():
                        yield ReasoningOutput(progress, spoken=self._spoken_progress)
                    text.clear()
                elif isinstance(delta, LoopEndMarker):
                    reason = delta.reason
        finally:
            # The loop extended the context with each round; whatever the
            # turn's end, the session's conversation keeps them, closed by
            # what the model said last.
            closing = AIMessage(role="assistant", content="".join(text).strip() or "(no answer)")
            self._histories[session_id] = [*context.messages, closing]
        if reason != "completed":
            # A turn cut short (its round cap, its deadline, its budget) has
            # no answer, and its narration is none: the request failed, which
            # the channel answers aloud (RFC §12.4.1).
            raise ReasoningCutShortError(request.delegation_id, reason)
        if answer := "".join(text).strip():
            yield ReasoningOutput(answer, spoken=True, is_final=True)

    def _turn_context(
        self, request: ReasoningRequest, messages: list[AIMessage]
    ) -> tuple[AIContext, _ToolLoopContext]:
        """The turn's context, with the agent's own settings, and its loop
        context: the session's catalogue as the resolved toolset."""
        tools = [
            AITool(
                name=t["name"],
                description=str(t.get("description", "")),
                parameters=dict(t.get("parameters") or {}),
            )
            for t in request.tools
            if isinstance(t, dict) and t.get("name")
        ]
        agent = self._agent
        room_id = getattr(request.session, "room_id", None)
        binding = ChannelBinding(
            channel_id=agent.channel_id, room_id=room_id or "", channel_type=agent.channel_type
        )
        # The agent's own settings, its system prompt among them, as a turn
        # with no room override starts from.
        settings = agent._turn_settings(binding, None)
        context = AIContext(messages=messages, tools=tools, **settings)
        loop_ctx = _ToolLoopContext(room_id=room_id)
        loop_ctx.all_context_tools = tools
        # Every result comes through the voice channel's gate, which bounds it
        # (RFC §21.5): the loop has nothing to store and page back.
        loop_ctx.withdrawn_tools = frozenset({REREAD_TOOL})
        return context, loop_ctx

    async def _serve_through_gate(self, name: str, arguments: dict[str, Any]) -> str:
        """Serve one of the agent's calls through the voice channel's gate.

        The gate serves, judges and reports the call: its one report is claimed
        before it goes, so the loop reports it no second time (RFC §9.3).
        """
        call = current_tool_call()
        loop_ctx = _current_loop_ctx.get()
        if call is not None and loop_ctx is not None:
            loop_ctx.claim_report(call.tool_call_id)
        done = await _execute(_DELEGATION.get(), name, arguments)
        if done.is_error:
            raise ToolRefusedError(done.text)
        return done.text

    async def _report_loop_refusal(self, event: ToolCallEvent) -> None:
        """Report a call the loop refused before the gate (its arguments did
        not read, a turn cut it) to the voice channel's observers."""
        request = _DELEGATION.get()
        if request is None or request.report_refusal is None:
            return
        await request.report_refusal(
            event.name, dict(event.arguments), str(event.result or ""), cancelled=event.cancelled
        )

    async def session_ended(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    async def close(self) -> None:
        self._histories.clear()

    def _adopt_telemetry(self, telemetry: TelemetryProvider) -> None:
        # An agent registered with a kit keeps the kit's.
        if getattr(self._agent, "_telemetry", None) is None:
            self._agent._telemetry = telemetry  # ty: ignore[unresolved-attribute]
            self._agent._propagate_telemetry()


class AIProviderReasoningBackend(AgentReasoningBackend):
    """Default backend: an :class:`~roomkit.providers.ai.base.AIProvider` as an
    agent of its own (:class:`AgentReasoningBackend`).

    Example:
        backend = AIProviderReasoningBackend(
            AnthropicAIProvider(AnthropicConfig(api_key="...")),
            system_prompt=BACKEND_PROMPT,
        )
    """

    def __init__(
        self,
        provider: AIProvider,
        *,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
        temperature: float | None = None,
        spoken_progress: bool = False,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must not be negative")
        agent = AIChannel(
            "reasoning-backend",
            provider=provider,
            system_prompt=system_prompt,
            max_tool_rounds=max_tool_rounds,
            # None keeps the default every turn starts from.
            temperature=AIContext().temperature if temperature is None else temperature,
            tool_search=False,
        )
        super().__init__(agent, spoken_progress=spoken_progress)


def _refuse_own_tools(agent: AIChannel) -> None:
    """Refuse an agent whose own tools would bypass the voice channel's gate."""
    own = {
        "tools": bool(agent._user_tools or agent._user_tool_handler),
        "skills": agent._skills is not None,
        "a sandbox": agent._sandbox is not None,
        "an external tool handler": agent._external_tool_handler is not None,
        "a human-input handler": agent._human_input_handler is not None,
        "planning": agent._planner is not None,
    }
    if carried := [what for what, has in own.items() if has]:
        raise ValueError(
            f"A reasoning backend's agent serves the voice session's tools only, through "
            f"the voice channel's gate (RFC §12.4.1); agent {agent.channel_id!r} carries "
            f"{', '.join(carried)}"
        )


async def _execute(
    request: ReasoningRequest | None, name: str, arguments: dict[str, Any]
) -> ToolCallResult:
    """One call through the channel's gate, read with its outcome."""
    try:
        if request is not None and request.execute_tool_call is not None:
            return await request.execute_tool_call(name, arguments)
        if request is not None and request.execute_tool is not None:
            return ToolCallResult(await request.execute_tool(name, arguments))
    except Exception as exc:
        # The class, never the message (RFC §9.3): it goes to the log.
        logger.exception("Reasoning backend tool %s failed", name)
        return ToolCallResult(tool_failure(name, exc), is_error=True)
    error = json.dumps({"error": f"No tool executor available for {name}"})
    return ToolCallResult(error, is_error=True)
