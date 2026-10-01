"""AIChannel mixin for tool execution, dispatch, and skill tool handlers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._sandbox_handlers import handle_sandbox_command
from roomkit.channels._served_tools import CollisionLog, declared_once
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    ALREADY_ACTIVE_NOTE,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
)
from roomkit.channels._skill_handlers import (
    activation_ack,
    handle_activate_skill,
    handle_read_reference,
    handle_run_script,
)
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tool_eviction import ToolEviction, kept_whole
from roomkit.channels._tool_registry import (
    ChannelRegistry,
    ToolSource,
    channel_tool,
    schema_tool,
)
from roomkit.channels._tool_search import (
    normalize_max_results,
    related_family_tools,
    render_find_payload,
    render_list_payload,
    search_catalogue,
    search_tool_defs,
)
from roomkit.channels._tool_search_constants import (
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)
from roomkit.core.exceptions import (
    ChannelRefusalError,
    ToolRefusedError,
    UnservedToolCallError,
)
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent, ToolCallVerdict
from roomkit.providers.ai.base import (
    AIImagePart,
    AIProvider,
    AITextPart,
    AITool,
    AIToolResultPart,
)
from roomkit.providers.ai.tool_calls import cut_call_error
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.redaction import redact
from roomkit.tools.context import ToolCallContext, _current_tool_call
from roomkit.tools.result import (
    as_tool_result,
    failure_detail,
    is_unknown_tool_answer,
    pre_execution_denial,
    tool_failure,
    unserved_tool_error,
)
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.models.tool_call import ToolCallCallback, ToolCallObserver
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.human_input import HumanInputToolHandler
    from roomkit.tools.policy import ToolPolicy

    ToolResult = str | list[AITextPart | AIImagePart]
    ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

logger = logging.getLogger("roomkit.channels.ai")


@dataclass(frozen=True)
class _HookOutcome:
    """ON_TOOL_CALL's verdict applied to one served call."""

    result: Any  # what the model reads, before eviction
    recorded: Any  # what the usage memory keeps
    failed: bool  # the hook blocked the call, or nothing served it
    structured: dict[str, Any] | None  # the structured copy the call keeps
    remember: bool = True  # the room's tool memory keeps it (not an unserved call)


def _log_answer(name: str, result: Any, started: float) -> None:
    elapsed = (time.monotonic() - started) * 1000
    if result is None:
        logger.info("Tool %s: nothing served it (%.0f ms)", name, elapsed)
        return
    size = len(result) if isinstance(result, str) else -1
    logger.info("Tool %s returned %d chars in %.0f ms", name, size, elapsed)
    logger.debug("Tool %s result: %s", name, redact(_preview(result)))


@runtime_checkable
class AIToolsHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolsMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider — read for the model id in fold diagnostics.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _user_tool_handler: User-provided tool handler for fallback dispatch.
        _skills: Skill registry for gated tool resolution.
        _script_executor: Script executor for skill scripts.
        _sandbox: Sandbox executor for ad-hoc command execution.
        _eviction: Tool result eviction / truncation strategy.
        _registry: The tools the channel serves, with their traits.
        _skill_activation: Per-room record of the skills active in a conversation.
        _planner: Optional task planner.
        _realtime: Realtime backend for ephemeral events.
        _tool_call_hook: Optional unified ON_TOOL_CALL hook callback.
        _tool_observer_hook: Optional ON_TOOL_CALL observer callback, fired for
            a call that failed or was refused.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        _effective_tool_policy: ``AIToolPolicyMixin`` property — resolved policy.
        _gated_tool_names: ``AIToolPolicyMixin`` property — gated tool names.
        _maybe_truncate_result: ``AIResilienceMixin`` — truncate large results.
        _get_loop_ctx: ``AISteeringMixin`` — returns current tool-loop context.
        _apply_tool_filters: ``AIToolPolicyMixin`` — policy / skill-gating /
            Tool Search visibility filter.
        _reachable_tools: ``AIToolPolicyMixin`` — the tools policy and skill
            gating admit, Tool Search's window aside.
        _gate_refusal: ``AIToolPolicyMixin`` — why policy or gating refuses a
            call, or ``None``.
    """

    _provider: AIProvider
    _tool_handler: Any
    _user_tool_handler: Any
    _user_tools: list[AITool]
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _human_input_handler: HumanInputToolHandler | None
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _realtime: RealtimeBackend | None
    _plan_updated_hook: Any  # ON_PLAN_UPDATED callback — injected by register_channel
    _tool_call_hook: ToolCallCallback | None
    _tool_observer_hook: ToolCallObserver | None
    _before_tool_call_hook: Any
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_miss_hint: str | None
    channel_id: str

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None: ...
    @property
    def _gated_tool_names(self) -> set[str]: ...

    def _maybe_truncate_result(
        self,
        result: str | list[AITextPart | AIImagePart],
        tool_call_id: str = ...,
    ) -> str | list[AITextPart | AIImagePart]: ...
    def _get_loop_ctx(self) -> _ToolLoopContext: ...
    def _orchestration_tool_names(self, room_id: str | None) -> set[str]: ...
    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]: ...
    def _reachable_tools(self, tools: Iterable[AITool]) -> list[AITool]: ...
    def _gate_refusal(self, name: str) -> dict[str, str] | None: ...
    def _reference_shown(self, loop_ctx: _ToolLoopContext) -> list[str]: ...


def _tool_name(tool: AITool) -> str:
    return tool.name


def _cut_call_error(tc: Any) -> dict[str, Any]:
    """What the model reads for a call cut before its arguments were complete."""
    logger.warning("Provider cut tool call %s (%s) before its arguments ended", tc.name, tc.id)
    return cut_call_error(tc.name)


class AIToolsMixin:
    """Parallel tool execution, skill tool definitions, and dispatch routing.

    Host contract: :class:`AIToolsHost`.
    """

    _provider: AIProvider
    _tool_handler: Any
    _user_tool_handler: Any
    _user_tools: list[AITool]
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _human_input_handler: HumanInputToolHandler | None
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _realtime: RealtimeBackend | None
    _plan_updated_hook: Any  # ON_PLAN_UPDATED callback — injected by register_channel
    _tool_call_hook: ToolCallCallback | None
    _tool_observer_hook: ToolCallObserver | None
    _before_tool_call_hook: Any
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_miss_hint: str | None
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing
    _effective_tool_policy: Any  # see AIToolsHost
    _gated_tool_names: Any  # see AIToolsHost
    _maybe_truncate_result: Any  # see AIToolsHost
    _get_loop_ctx: Any  # see AIToolsHost
    _apply_tool_filters: Any  # see AIToolsHost
    _reachable_tools: Any  # see AIToolsHost
    _never_deferred: Any  # AIToolPolicyMixin: what Tool Search never defers
    _gate_refusal: Any  # see AIToolsHost
    _reference_shown: Any  # AIToolPolicyMixin: held tools a result makes callable
    _orchestration_tools: Any  # AIChannel: the tools orchestration set up for a room
    _orchestration_tool_names: Any  # AIChannel: never deferred behind Tool Search

    def _tool_parameters(
        self, name: str, declared_tools: list[AITool] | None = None
    ) -> dict[str, Any] | None:
        """Return the declared JSON-Schema ``parameters`` for tool *name*.

        ``None`` when the tool's schema is not known to this channel (infra,
        skill, or sandbox tools) — those skip argument validation.
        """
        if declared_tools is None:
            room_id = self._get_loop_ctx().room_id
            declared_tools = [*self._user_tools, *self._orchestration_tools(room_id)]
        for tool in declared_tools:
            if tool.name == name:
                return tool.parameters
        return None

    def _recover_deferred_tool(self, name: str) -> AITool | None:
        """A find_tools reveal applied at call time, for an exact-name call.

        Small models routinely skip the two-step discovery protocol and call a
        catalogue tool they saw (via list_tools, or a prior turn) without
        revealing it first. The name being exact, the call is trivially
        recoverable: reveal the tool as find_tools would have and let the call
        proceed — provided it survives the same visibility filter a reveal is
        subject to (tool policy, glob-aware skill gating). The execution guard
        applies that rule again to the call, but a name the filter refuses
        must not even be revealed.

        Returns the catalogue tool (its schema keeps argument validation
        fail-closed) or ``None`` when the name is not recoverable.
        """
        loop_ctx = self._get_loop_ctx()
        if not loop_ctx.tool_search_active:
            # Inactive search declares the whole filtered catalogue — an
            # undeclared name is either filtered out or unknown, never deferred.
            return None
        tool = next((t for t in loop_ctx.all_context_tools or () if t.name == name), None)
        if tool is None:
            return None
        # Reveal first: while Tool Search is active the filter keeps only
        # pinned/revealed/sticky names, so the eligibility probe needs the name
        # in the reveal set. Rolled back when the probe fails.
        loop_ctx.revealed_tools.add(name)
        if not self._apply_tool_filters([tool]):
            loop_ctx.revealed_tools.discard(name)
            return None
        # Parity with _handle_find_tools: the reveal persists across turns.
        self._tool_usage.record_revealed(loop_ctx.room_id, {name})
        return tool

    def _declared_schema(
        self, name: str, declared_tools: list[AITool] | None
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        """The schema a call to *name* is validated against, or why it is undeclared.

        Once the turn's toolset is resolved, a call must name a tool the round
        declared, an empty declaration included, or one Tool Search recovers
        from the turn's catalogue (RFC §6.4); the channel's own tools answer
        for themselves. A loop built without context (``all_context_tools``
        is ``None``) has no declaration to hold the call to.
        """
        params = self._tool_parameters(name, declared_tools)
        # A tool held unseen that nothing referenced is not callable yet: it
        # goes through recovery like a tool Tool Search hides (RFC §6.4).
        referenced = self._get_loop_ctx().referenced
        declared_names = {
            tool.name
            for tool in declared_tools or []
            if not tool.defer_loading or tool.name in referenced
        }
        channel_managed = name in self._channel_tool_names()
        resolved = bool(declared_names) or self._get_loop_ctx().all_context_tools is not None
        if not resolved or name in declared_names or channel_managed:
            return params, None
        recovered = self._recover_deferred_tool(name)
        if recovered is None:
            logger.warning("Provider requested undeclared tool %s", name)
            return None, self._undeclared_tool_error(name)
        # The model skipped find_tools but named a real catalogue tool: the
        # reveal happened at call time instead of ahead of it, and every
        # guard after this one still applies.
        logger.info("Recovered deferred catalogue tool %s at call time", name)
        return recovered.parameters, None

    def _undeclared_tool_error(self, name: str) -> dict[str, str]:
        """Actionable payload for an undeclared call that could not be recovered."""
        loop_ctx = self._get_loop_ctx()
        if any(t.name == name for t in loop_ctx.all_context_tools or ()):
            # In the catalogue but filtered out (tool policy or skill gating):
            # a find_tools reveal would be dropped by the same filter, so no
            # retry hint — the refusal is the answer.
            return {
                "error": (
                    f"Tool '{name}' exists but is not available to this agent "
                    "(blocked by the tool policy or gated behind a skill)."
                )
            }
        if loop_ctx.tool_search_active:
            return {
                "error": f"Unknown tool '{name}': no tool by that name exists.",
                "hint": (
                    "Check the spelling, or call find_tools(query=<the task>) "
                    "to discover the right tool."
                ),
            }
        return {"error": f"Unknown tool '{name}': it is not declared"}

    async def _fire_tool_refusal(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: str,
        room_id: str | None,
        *,
        detail: str | None = None,
    ) -> None:
        """Fire ON_TOOL_CALL for a call that failed or was refused.

        *detail* is a raised call's full failure, or the error of a
        BEFORE_TOOL_USE hook that failed closed, for the observers only
        (``ToolCallEvent.error_detail``).

        The refusal paths below return before the handler runs, and a handler
        that raises jumps past the firing that follows it — so without this,
        ON_TOOL_CALL only ever reported the calls that worked. A host auditing
        tool use saw a denied tool as a tool that was never called, which reads
        the same as an agent that never tried.

        Observational by construction: it reaches the ASYNC observers of
        ON_TOOL_CALL and no further. A SYNC hook is the one that can *serve* a
        call, so a refused call must not reach one — otherwise the refusal
        would hide the side effect instead of preventing it. A hook that raises
        must not turn a refusal into a crash either: the refusal is already on
        its way to the model.
        """
        if self._tool_observer_hook is None:
            return
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tc.id,
            name=tc.name,
            arguments=arguments,
            result=result,
            room_id=room_id,
            is_error=True,
            error_detail=detail,
        )
        try:
            await self._tool_observer_hook(event)
        except Exception:
            logger.debug(
                "ON_TOOL_CALL observation failed for refused tool %s", tc.name, exc_info=True
            )

    async def _failed_call(
        self,
        tc: Any,
        arguments: dict[str, Any],
        room_id: str | None,
        body: str,
        *,
        detail: str | None = None,
    ) -> ToolResult:
        """The model's copy of a call that was refused or failed, its observers told.

        Fired on the body before eviction and the repeated-result note shape
        the model's copy of it.
        """
        await self._fire_tool_refusal(tc, arguments, body, room_id, detail=detail)
        return self._bound_tool_result(tc.name, body, tc.id)

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        declared_tools: list[AITool] | None = None,
        parent_span_id: str | None = None,
        executed_arguments: dict[str, dict[str, Any]] | None = None,
    ) -> list[_ContentPart]:
        """Execute tool calls concurrently and return result parts."""
        if self._tool_handler is None:
            raise RuntimeError("_execute_tools_parallel called without a tool handler")
        handler = self._tool_handler
        # Capture the invocation-scoped room once. The channel object is shared
        # across rooms, while the loop context is copied into every task spawned
        # by gather below.
        room_id = self._get_loop_ctx().room_id

        async def _run_one(tc: Any) -> AIToolResultPart:
            # INFO names the call and its argument keys; the values can carry
            # personal data: DEBUG shows them only with content logging on.
            logger.info(
                "Executing tool %s (call %s) with %s",
                tc.name,
                tc.id,
                ", ".join(sorted(tc.arguments)) or "no arguments",
            )
            logger.debug("Tool %s arguments: %s", tc.name, redact(_preview(tc.arguments)))

            async def rejected(
                error: dict[str, Any], detail: str | None = None
            ) -> AIToolResultPart:
                # Refusals never reach the handler's guard. Count their raw
                # attempts here; successful calls are counted only by the
                # handler, using the effective payload after folds and hooks.
                guard = self._repeated_call_guard(tc.name, tc.arguments)
                body = guard or json.dumps(error)
                await self._fire_tool_refusal(tc, tc.arguments, body, room_id, detail=detail)
                return AIToolResultPart(
                    tool_call_id=tc.id, name=tc.name, result=body, is_error=True
                )

            if getattr(tc, "partial", False):
                return await rejected(_cut_call_error(tc))
            # A tool BEFORE_AI_GENERATION withdrew is gone for the turn, the
            # channel's own included: no exemption below may bring it back.
            if tc.name in self._get_loop_ctx().withdrawn_tools:
                logger.warning("Provider called %s, withdrawn for this turn", tc.name)
                return await rejected(
                    {"error": f"Tool '{tc.name}' is not available in this turn."}
                )

            # Execution guard: argument validation against the declared schema
            # (fail-closed) — reject malformed calls before any other gate.
            params, undeclared = self._declared_schema(tc.name, declared_tools)
            if undeclared is not None:
                return await rejected(undeclared)
            call_arguments = tc.arguments
            if params is not None:
                # Repair before validating: a model that flattened a hub tool's
                # ``params`` gets its call folded back into shape instead of
                # spending a round on an error it can only fix by re-issuing.
                folded, fold_error = fold_hoisted_arguments(params, call_arguments)
                if fold_error is not None:
                    logger.warning("Tool %s arguments ambiguous: %s", tc.name, fold_error)
                    return await rejected(
                        {"error": f"Invalid arguments for '{tc.name}': {fold_error}"}
                    )
                if folded is not None:
                    logger.info(
                        "Tool %s: folded hoisted arguments %s into its container (model=%s)",
                        tc.name,
                        sorted(set(call_arguments) - set(folded)),
                        self._provider.model_name,
                    )
                    call_arguments = folded
                arg_error = validate_tool_arguments(params, call_arguments)
                if arg_error is not None:
                    logger.warning("Tool %s arguments rejected: %s", tc.name, arg_error)
                    return await rejected(
                        {"error": f"Invalid arguments for '{tc.name}': {arg_error}"}
                    )

            # Execution guard: policy and skill gating, the listing filter's
            # rule (RFC §21.1), re-checked on the call itself.
            refusal = self._gate_refusal(tc.name)
            if refusal is not None:
                return await rejected(refusal)

            # Pre-execution gate: BEFORE_TOOL_USE hook can deny the tool call,
            # or hand back rewritten arguments (a redaction hook putting real
            # values back before the tool acts on the model's tokenised text).
            # The handler and ON_TOOL_CALL read ``arguments``, so they report
            # what actually ran; the usage record keeps the model's own.
            arguments = call_arguments
            arguments_rewritten = False
            if self._before_tool_call_hook is not None:
                pre_event = ToolCallEvent(
                    channel_id=self.channel_id,
                    channel_type=ChannelType.AI,
                    tool_call_id=tc.id,
                    name=tc.name,
                    arguments=arguments,
                    result=None,
                    room_id=room_id,
                )
                decision = await self._before_tool_call_hook(pre_event)
                if not decision:
                    logger.info("Tool %s denied by BEFORE_TOOL_USE hook", tc.name)
                    return await rejected(
                        {"error": pre_execution_denial(tc.name)}, detail=decision.detail
                    )
                if decision.arguments is not None:
                    arguments = decision.arguments
                    arguments_rewritten = True

            # Validate the payload after every hook, even when it did not
            # explicitly return a replacement. ToolCallEvent is frozen but its
            # nested dict is mutable, so an in-place edit must not bypass this
            # fail-closed boundary either. No fold here, deliberately: these
            # arguments come from user code, and repairing a hook's output
            # would hide the hook's bug instead of naming it. The model's own
            # call was already folded above, so a hook that rewrites nothing
            # arrives here in the repaired shape.
            if params is not None:
                arg_error = validate_tool_arguments(params, arguments)
                if arg_error is not None:
                    qualifier = "rewritten " if arguments_rewritten else ""
                    logger.warning(
                        "Tool %s %sarguments rejected: %s", tc.name, qualifier, arg_error
                    )
                    return await rejected(
                        {"error": (f"Invalid {qualifier}arguments for '{tc.name}': {arg_error}")}
                    )

            tool_span_id = telemetry.start_span(
                SpanKind.LLM_TOOL_CALL,
                f"tool.{tc.name}",
                parent_id=parent_span_id,
                attributes={"tool.name": tc.name, "tool.id": tc.id},
            )
            structured_content: dict[str, Any] | None = None
            tool_failed = False
            # What the tool actually returned, before eviction swaps an
            # oversized body for a placeholder: the usage memory keeps the head
            # of the data, which is what a later turn asks about.
            recorded_result: Any = None
            remember = True  # see _remember_call
            if executed_arguments is not None:
                # Snapshot the post-hook payload before handing it to user
                # code. Persistence can then distinguish what the model
                # requested from what actually executed, in both loops.
                executed_arguments[tc.id] = dict(arguments)
            try:
                # Set contextvar so HumanInputToolHandler can read
                # room_id / tool_call_id / channel_id without protocol changes.
                _tc_ctx = ToolCallContext(
                    room_id=room_id or "",
                    tool_call_id=tc.id,
                    channel_id=self.channel_id,
                )
                started = time.monotonic()
                result = await self._serve_call(handler, tc.name, arguments, _tc_ctx)
                recorded_result = result
                _log_answer(tc.name, result, started)
                hook = await self._apply_tool_call_hook(tc, arguments, result, _tc_ctx, room_id)
                recorded_result, tool_failed = hook.recorded, hook.failed
                structured_content, remember = hook.structured, hook.remember
                result = self._bound_tool_result(tc.name, hook.result, tc.id)

                telemetry.end_span(tool_span_id)
            except asyncio.CancelledError:
                telemetry.end_span(tool_span_id, status="cancelled")
                raise
            except ToolRefusedError as refusal:
                # The branch below with the message kept. A handler that
                # declines a call has words for the model — a host tunes them
                # for a small one — and the generic wrapper would replace them
                # with its own sentence, which is how the reason gets lost.
                telemetry.end_span(tool_span_id, status="error", error_message=refusal.message)
                logger.info("Tool %s refused: %s", tc.name, refusal.message)
                recorded_result, tool_failed = refusal.message, True
                remember = not isinstance(refusal, ChannelRefusalError)
                result = await self._failed_call(tc, arguments, room_id, refusal.message)
            except Exception as exc:
                telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
                logger.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
                # The class, never the message (RFC §9.3): it goes to the log
                # above and to the observers, not to the model. The memory
                # records what the model saw, not a success the handler
                # returned before a hook raised.
                recorded_result, tool_failed = tool_failure(tc.name, exc), True
                result = await self._failed_call(
                    tc, arguments, room_id, recorded_result, detail=failure_detail(exc)
                )
            self._settle_activation(tc.id, served=not tool_failed)
            outcome = recorded_result if recorded_result is not None else result
            if remember:
                self._remember_call(room_id, tc.name, call_arguments, outcome)
            return self._model_part(
                tc,
                result,
                outcome,
                structured=structured_content,
                failed=tool_failed,
                references=[] if tool_failed else self._reference_shown(self._get_loop_ctx()),
            )

        tasks = [asyncio.create_task(_run_one(tc)) for tc in tool_calls]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # gather propagates a failed gate or a cancelled child immediately;
            # its siblings otherwise keep running after the loop has ended.
            # Own them through cleanup, without cancelling a finalizer twice.
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return list(results)

    def _model_part(
        self,
        tc: Any,
        result: ToolResult,
        outcome: Any,
        *,
        structured: dict[str, Any] | None,
        failed: bool,
        references: list[str],
    ) -> AIToolResultPart:
        """What the model reads of a call: its result, noted when this tool
        already gave that answer this turn, and the held tools it makes
        callable."""
        # The hash is taken on the recorded outcome, so the memory keeps the
        # tool's own output and only the model's copy carries the note, and
        # the hash stays stable: annotating before hashing would make every
        # repeat look new, and so would an evicted copy, whose placeholder id
        # is unique per call.
        if isinstance(result, str):
            hashed = outcome if isinstance(outcome, str) else result
            result = self._repeated_result_note(tc.name, result, outcome=hashed)
        return AIToolResultPart(
            tool_call_id=tc.id,
            name=tc.name,
            result=result,
            structured_content=structured,
            is_error=failed,
            references=references,
        )

    def _skill_tools(self) -> list[AITool]:
        """Build the list of AITool definitions for skill operations."""
        tools = [schema_tool(ACTIVATE_SKILL_SCHEMA), schema_tool(READ_REFERENCE_SCHEMA)]
        if self._script_executor:
            tools.append(schema_tool(RUN_SCRIPT_SCHEMA))
        return tools

    def _register_channel_tools(self) -> None:
        """Register the tools the channel serves itself, each with its traits.

        A handler may be sync or async: the dispatcher awaits what needs it.
        """
        served: list[tuple[AITool, Any]] = [
            (ToolEviction.tool_definition(), self._handle_read_tool_result)
        ]
        if self._planner is not None:
            served.append((TaskPlanner.tool_definition(), self._handle_plan_tasks))
        if self._skills:
            served += [
                (schema_tool(ACTIVATE_SKILL_SCHEMA), self._handle_activate_skill),
                (schema_tool(READ_REFERENCE_SCHEMA), self._handle_read_reference),
                (schema_tool(RUN_SCRIPT_SCHEMA), self._handle_run_script),
            ]
        # Tool Search discovery tools are channel-managed (they reshape the
        # visible tool surface, not the world). Registered unless explicitly
        # disabled; they are only ever injected into context when active.
        if self._tool_search is not False:
            find, inventory = search_tool_defs()
            served += [(find, self._handle_find_tools), (inventory, self._handle_list_tools)]
        for definition, serve in served:
            self._registry.register(channel_tool(definition, serve), owner=self)

    # Identical-call ceiling for regular tools: the 3rd repeat short-circuits.
    # Two identical executions can be legitimate (retry after a transient
    # failure); a model issuing the same call a third time is looping — the
    # observed failure mode is a small model re-running one find_tools query
    # for an entire turn and never answering.
    _REPEAT_CALL_LIMIT = 3
    # After the guard has BLOCKED the same call this many extra times and the
    # model still re-issues it, the advisory clearly isn't landing — force-stop
    # the loop. Small models otherwise ignore the error and hammer the same
    # call to the round limit (observed: sandbox_bash({}) called 37×).
    _REPEAT_FORCE_STOP_AT = 3
    # The same ceiling on the OTHER axis: how many identical RESULTS from one
    # tool before the model is told. Matched to ``_REPEAT_CALL_LIMIT`` for the
    # same reason — a second identical answer is ordinary (a retry, a poll, two
    # rows deleted), a third is a pattern.
    _REPEAT_RESULT_LIMIT = 3
    # Marker on this module's own advisory results, so a repeated advisory does
    # not get annotated as a repeated result. It already says what is wrong.
    _ADVISORY_MARKER = "these EXACT arguments"

    def _repeated_result_note(self, name: str, result: str, *, outcome: str | None = None) -> str:
        """Append a note when a tool returns an answer it already gave this turn.

        ``outcome`` is what the tool gave, when ``result`` (the model's copy)
        differs from it: an evicted copy carries a per-call id, so identical
        answers are recognised on the outcome. It defaults to ``result``.

        The blind spot in ``_repeated_call_guard``: it keys on the arguments, so
        a model that permutes them is never told anything. Measured on a stuck
        turn — 54 calls, 44 distinct argument sets, **25 distinct results**, one
        of them (`{"cards":[],"total":0}`) returned 23 times. The model narrated
        "let me confirm" at every round because nothing in what it read said the
        confirmation had already arrived, twenty-two times.

        **Annotates, never blocks**, and that asymmetry is deliberate. Identical
        results are not by themselves a fault: six deletions each answering
        ``{"success": true}`` are six correct operations with one result, and
        short-circuiting the sixth would destroy real work to save latency.
        Blocking stays with the argument guard, which cannot mistake legitimate
        work for a loop. This one only supplies the missing fact and lets the
        model act on it.
        """
        if self._ADVISORY_MARKER in result:
            return result
        hashed = result if outcome is None else outcome
        digest = hashlib.sha256(hashed.encode("utf-8", "replace")).hexdigest()
        counts = self._get_loop_ctx().repeated_results
        key = (name, digest)
        counts[key] = count = counts.get(key, 0) + 1
        if count < self._REPEAT_RESULT_LIMIT:
            return result
        # The only witness. The note rides on the tool result handed to the
        # model, which is downstream of the ON_TOOL_CALL hook the audit trail
        # listens on and absent from the turn-start context snapshot — so
        # neither of the two places an operator would look can show that this
        # fired. Logging it is what makes the guard observable at all.
        logger.warning(
            "Anti-loop: '%s' returned an identical result %d times this turn", name, count
        )
        return (
            f"{result}\n\n[identical result: '{name}' has now returned exactly this "
            f"{count} times this turn, for different arguments. Varying the arguments "
            f"is not finding anything new — this answer is settled. Use it and move "
            f"on, or answer with what you have.]"
        )

    def _repeated_call_guard(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Short-circuit a tool call repeated with identical arguments this turn."""
        try:
            key = (name, json.dumps(arguments or {}, sort_keys=True, default=str))
        except (TypeError, ValueError):
            return None
        loop_ctx = self._get_loop_ctx()
        counts = loop_ctx.repeated_calls
        counts[key] = count = counts.get(key, 0) + 1
        # A pure tool reads what cannot change within the turn (Tool Search's
        # fixed catalogue): an identical repeat never says anything new, so it
        # short-circuits at 2.
        traits = self._registry.traits(name, loop_ctx.room_id)
        limit = 2 if traits is not None and traits.pure else self._REPEAT_CALL_LIMIT
        if count < limit:
            return None
        # The model is ignoring the advisory and re-issuing anyway — pull the
        # ripcord so the loop force-ends with a plain-text answer.
        if count >= limit + self._REPEAT_FORCE_STOP_AT:
            loop_ctx.force_stop = True
        return json.dumps(
            {
                "error": (
                    f"You already called '{name}' with these EXACT arguments "
                    f"{count - 1} time(s) this turn — repeating it cannot yield "
                    "anything new."
                ),
                "hint": (
                    "STOP repeating this call. Use the results you already "
                    "have, try genuinely different arguments, or answer the "
                    "user now with what you know."
                ),
            }
        )

    def _sandbox_tool_names(self) -> frozenset[str]:
        """The names the attached sandbox declares, the ones the channel serves.

        By exact name, never by prefix: a host tool that merely starts with
        ``sandbox_`` is the host's (RFC §21.1).
        """
        if self._sandbox is None:
            return frozenset()
        return frozenset(
            tdef["name"]
            for tdef in self._sandbox.tool_definitions()
            if tdef["name"].startswith(SANDBOX_TOOL_PREFIX)
        )

    def _channel_tool_names(self) -> set[str]:
        """The tools this channel serves itself, before any host handler.

        Its own dispatch, its sandbox's commands and its human-input tools:
        a host tool under one of these names would be declared with the
        host's schema and served by the channel (RFC §21.1).
        """
        names = {e.name for e in self._registry.entries(None, source=ToolSource.CHANNEL)}
        names |= self._sandbox_tool_names()
        if self._human_input_handler is not None:
            names |= {tool.name for tool in self._human_input_handler.tools or ()}
        return names

    def _served_tool_names(self, room_id: str | None) -> set[str]:
        """The tools the channel and orchestration serve in *room_id*, before
        any host handler: a host tool under one of these is not declared."""
        return self._channel_tool_names() | self._registry.names(room_id)

    def _declared_once(self, tools: list[AITool], room_id: str | None) -> list[AITool]:
        """The host's part of a turn's toolset in *room_id*: no tool under a name
        the channel or orchestration serves there, and each name once (RFC
        §21.1, :func:`declared_once`)."""
        served = self._served_tool_names(room_id)
        return declared_once(tools, _tool_name, served, self._collisions)

    async def _channel_tool_handler(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Unified tool dispatcher: channel-managed -> sandbox -> skill -> user tools.

        The outcomes the channel decides itself (a repeat the guard stops, a
        tool outside the turn's toolset) are refusals, raised as
        :class:`ToolRefusedError` so they carry the failure marker (RFC §9.3).
        :class:`UnservedToolCallError` when nothing serves the call:
        ON_TOOL_CALL's hooks may then serve it.
        """
        guard = self._repeated_call_guard(name, arguments)
        if guard is not None:
            raise ChannelRefusalError(guard)
        entry = self._registry.lookup(name, self._get_loop_ctx().room_id)
        if entry is not None and entry.serve is not None:
            result = entry.serve(arguments)
            # Support both sync and async handlers
            if asyncio.iscoroutine(result):
                result = await result
            return as_tool_result(result)
        # The sandbox's own tools, by exact name, before user/MCP tools
        if self._sandbox is not None and name in self._sandbox_tool_names():
            return await handle_sandbox_command(name, arguments or {}, self._sandbox)
        # Provider responses are untrusted and may name a tool outside the
        # turn's resolved toolset. Once context construction has resolved
        # that invocation-scoped set, fail closed, before anything may serve
        # it, a hook included. ``None`` preserves direct internal loops built
        # without context; [] is a real deny-all set.
        context_tools = self._get_loop_ctx().all_context_tools
        if context_tools is not None and name not in {t.name for t in context_tools}:
            raise ChannelRefusalError(
                json.dumps({"error": f"Tool '{name}' is not available in the current turn."})
            )
        if self._user_tool_handler is None:
            raise UnservedToolCallError(name)
        return as_tool_result(await self._user_tool_handler(name, arguments))

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        loop_ctx = self._get_loop_ctx()
        skill = self._skills.get_skill(skill_name) if skill_name else None
        if skill_name and skill is None:
            # A known-but-unavailable skill already carries its reason in the
            # error — a "this is not a skill" tools hint would contradict it.
            if self._skills.get_unavailable_reason(skill_name) is None:
                # Small models routinely confuse skills with TOOLS ("activate the
                # Spotify skill" when SpotifySearch/... are tools). Turn the dead
                # end into the right outcome: reveal the matching tools and say so.
                wanted = skill_name.lower()
                reachable = self._reachable_tools(loop_ctx.all_context_tools or ())
                matching = sorted(t.name for t in reachable if wanted in t.name.lower())
                if matching:
                    loop_ctx.revealed_tools.update(matching)
                    data = json.loads(result_str)
                    data["tools_hint"] = (
                        f"{skill_name!r} is not a skill, but these TOOLS match and are "
                        f"now in your tool list — call one directly instead: "
                        f"{', '.join(matching[:8])}."
                    )
                    result_str = json.dumps(data)
            return result_str
        # Recorded once the call's outcome is known: an ON_TOOL_CALL hook that
        # blocks the call, or a failure, must open no gate (_settle_activation).
        already_active = self._skill_activation.is_active(loop_ctx.room_id, skill_name)
        self._defer_activation(loop_ctx, skill_name)
        if skill is None or not already_active:
            return result_str
        # Already active: _build_context put these very instructions in front of
        # the model before the turn started, so the body just built above would
        # be a second copy of rules it already holds. Ack instead.
        return activation_ack(skill, ALREADY_ACTIVE_NOTE, already_active=True)

    def _defer_activation(self, loop_ctx: _ToolLoopContext, skill_name: str) -> None:
        """Hold an activation until its call is served, or record it now.

        Inside the tool loop the call's outcome is not known yet: an
        ON_TOOL_CALL hook may still block it. Outside one (a direct call)
        nothing can, and the activation is recorded at once.
        """
        call = _current_tool_call.get()
        if call is not None and call.tool_call_id:
            loop_ctx.pending_activations[call.tool_call_id] = skill_name
        else:
            self._record_activation(loop_ctx, skill_name)

    def _settle_activation(self, tool_call_id: str, *, served: bool) -> None:
        """Record the activation a served call asked for; drop a refused one."""
        loop_ctx = self._get_loop_ctx()
        skill_name = loop_ctx.pending_activations.pop(tool_call_id, None)
        if skill_name is not None and served:
            self._record_activation(loop_ctx, skill_name)

    def _record_activation(self, loop_ctx: _ToolLoopContext, skill_name: str) -> None:
        # For this turn, so gated tools become visible on the next round...
        loop_ctx.activated_skills.add(skill_name)
        # ... and for the rest of the conversation, so the body can ride the
        # system prompt instead of being re-fetched every turn.
        self._skill_activation.activate(loop_ctx.room_id, skill_name)

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        return await handle_run_script(arguments, self._skills, self._script_executor)

    def _tool_search_catalogue(self, loop_ctx: _ToolLoopContext) -> list[dict[str, Any]]:
        """The turn's reachable tools as score-able dicts (name + description + tags).

        Only what the policy allows and no skill gates (RFC §21.1): a match
        the model can never call is a false promise, and listing it discloses
        what the policy hides.
        """
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "tags": getattr(t, "tags", []) or [],
            }
            for t in self._reachable_tools(loop_ctx.all_context_tools or ())
        ]

    async def _handle_find_tools(self, arguments: dict[str, Any]) -> str:
        """Reveal catalogue tools matching a query for the rest of the loop.

        Mutates ``loop_ctx.revealed_tools`` (swap window); the next round's
        tool re-filter exposes the matches. No ``provider.reconfigure`` — the
        text loop re-sends its tool list every round.
        """
        loop_ctx = self._get_loop_ctx()
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps(
                {
                    "error": "query is required",
                    "hint": "Pass a short natural-language description.",
                }
            )
        catalogue = self._tool_search_catalogue(loop_ctx)
        max_results = normalize_max_results(
            arguments.get("max_results"), self._tool_search_threshold
        )
        # Declared already, never named: every tool Tool Search never defers
        # (RFC §6.4, §21.1).
        exclude = self._never_deferred(loop_ctx)
        matches = search_catalogue(catalogue, query, max_results, exclude_names=exclude)
        loop_ctx.revealed_tools = {m["name"] for m in matches if m.get("name")}
        # Reveals persist across turns via ToolUsageMemory (the tool's own
        # description promises "the rest of the session") — a tool found in
        # turn N is often only called in turn N+1, after the user confirms.
        self._tool_usage.record_revealed(loop_ctx.room_id, loop_ctx.revealed_tools)
        # Compact result (name + short description). The matched tools' full
        # schemas reach the model via the next round's re-filtered tool list
        # (loop_ctx.revealed_tools), so inlining them here would only risk
        # overflowing the tool-result size limit on verbose tools.
        return render_find_payload(
            matches,
            miss_hint=self._tool_search_miss_hint,
            related=related_family_tools(catalogue, matches),
        )

    async def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        """List the turn's catalogue (name + short description). Reveals nothing."""
        loop_ctx = self._get_loop_ctx()
        category = str(arguments.get("category", "")).strip()
        catalogue = self._tool_search_catalogue(loop_ctx)
        return render_list_payload(catalogue, category, exclude_names=TOOL_SEARCH_INFRA_TOOL_NAMES)

    def _remember_call(
        self, room_id: str | None, name: str, call_arguments: dict[str, Any], outcome: Any
    ) -> None:
        """Remember a call (its final result, success or error) so later turns
        show "tools you've already used" and re-reveal it under Tool Search.

        With the model's own arguments, never a BEFORE_TOOL_USE rewrite: the
        digest goes back into the next turn's prompt, and a hook that
        de-tokenises (``<EMAIL_1>`` to the real address) would put there the
        very value it kept from the model. Not for a refusal the channel
        decided, nor a call nothing served: neither is the tool's answer, and
        either would stand in for an earlier, identical call's real result.
        Infra/discovery tools are filtered inside ``record()``.
        """
        self._tool_usage.record(room_id, name, call_arguments, outcome)

    async def _apply_tool_call_hook(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: ToolResult | None,
        call_ctx: ToolCallContext,
        room_id: str | None,
    ) -> _HookOutcome:
        """Run ON_TOOL_CALL on the outcome the model will read, whole.

        After a text-only model's flattening, so the hook sees the shape the
        model reads; before eviction, so everything read_stored_result can
        page back has passed through the hook and a redacting hook covers the
        full text, not a preview of it.

        The call's structured copy (MCP structuredContent, which the handler
        left on *call_ctx*) is read here, once the handler returned, and only
        the outcome carries it on: a call that fails before or during this
        step keeps none. The hook sees the copy and may replace it; a BLOCK
        withholds the result and drops the copy. Eviction never touches it:
        UI surfaces need the payload whole.

        A call nothing served (*result* ``None``) reaches the hooks with no
        result: one may serve it with the result it supplies, and if none does
        the call failed, reported once to the observers (RFC §9.3).
        """
        shaped = None if result is None else self._shape_for_model(tc.name, result, tc.id)
        structured = None if result is None else call_ctx.structured_content
        verdict = await self._tool_call_verdict(tc, arguments, shaped, structured, room_id)
        if verdict is not None and verdict.blocked:
            reason = verdict.result or json.dumps({"error": "blocked"})
            return _HookOutcome(result=reason, recorded=reason, failed=True, structured=None)
        if verdict is not None and verdict.replaces_structured:
            structured = verdict.structured_content
        if verdict is not None and verdict.result is not None:
            override = as_tool_result(verdict.result)
            return _HookOutcome(
                result=override, recorded=override, failed=False, structured=structured
            )
        if shaped is None:
            body = unserved_tool_error(tc.name)
            detail = verdict.error_detail if verdict is not None else None
            await self._fire_tool_refusal(tc, arguments, body, room_id, detail=detail)
            return _HookOutcome(
                result=body, recorded=body, failed=True, structured=None, remember=False
            )
        return _HookOutcome(result=shaped, recorded=result, failed=False, structured=structured)

    async def _tool_call_verdict(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: ToolResult | None,
        structured: dict[str, Any] | None,
        room_id: str | None,
    ) -> ToolCallVerdict | None:
        """ON_TOOL_CALL's SYNC chain on one call's outcome, as a verdict."""
        if self._tool_call_hook is None:
            return None
        verdict = await self._tool_call_hook(
            ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=tc.id,
                name=tc.name,
                arguments=arguments,
                result=result,
                room_id=room_id,
                structured_content=structured,
            )
        )
        if verdict is None or isinstance(verdict, ToolCallVerdict):
            return verdict
        return ToolCallVerdict(result=verdict)  # a bare override

    async def _serve_call(
        self,
        handler: Any,
        name: str,
        arguments: dict[str, Any],
        call_ctx: ToolCallContext,
    ) -> ToolResult | None:
        """The handler's answer to one call, run in its tool call context.

        ``None`` when nothing served it: the dispatcher found no handler, or
        the handlers all answered that the tool is not theirs (RFC §21.4), the
        answer a composition passes a call on for.
        """
        token = _current_tool_call.set(call_ctx)
        timeout = self._call_timeout(name, call_ctx.room_id or None)
        try:
            answer = await answer_within(timeout, name, handler(name, arguments))
        except UnservedToolCallError:
            return None
        finally:
            _current_tool_call.reset(token)
        return None if is_unknown_tool_answer(answer) else as_tool_result(answer)

    def _call_timeout(self, name: str, room_id: str | None) -> float | None:
        """The bound of one call to *name* (RFC §21.6): the channel's, unless
        the tool waits on another agent or on a person by design."""
        waits = self._registry.waits(name, room_id) or (
            self._human_input_handler is not None and name in self._human_input_handler.tool_names
        )
        return self._tool_timeouts.for_call(name, waits=waits)

    def _shape_for_model(self, name: str, result: ToolResult, tool_call_id: str) -> ToolResult:
        """A text-only model gets the text of a content-part result, the way it
        gets a message's (``_extract_content``): an image it cannot take would
        fail the request."""
        if isinstance(result, list) and not self._provider.supports_vision:
            return AIToolResultPart(tool_call_id=tool_call_id, name=name, result=result).as_text()
        return result

    def _bound_tool_result(self, name: str, result: ToolResult, tool_call_id: str) -> ToolResult:
        """The copy of a tool's outcome the model reads, evicted when oversized.

        Every outcome goes through it (a result, a hook's override, a refusal,
        an error): whichever path a 500 KB body takes, it must not reach the
        provider whole, save a result the model reads whole (``kept_whole``:
        a skill's instructions, which a 20 KB skill would otherwise see
        evicted). References are data and still evict.
        """
        result = self._shape_for_model(name, result, tool_call_id)
        if kept_whole(name):
            return result
        return self._maybe_truncate_result(result, tool_call_id)

    # -- Extracted tool handlers (delegate to focused modules) -----------------

    def _handle_read_tool_result(self, arguments: dict[str, Any]) -> str:
        """Delegate to ToolEviction."""
        return self._eviction.handle_read(arguments)

    async def _handle_plan_tasks(self, arguments: dict[str, Any]) -> str:
        """Delegate to TaskPlanner."""
        if self._planner is None:
            return json.dumps({"error": "Planning is not enabled"})
        room_id = self._get_loop_ctx().room_id
        return await self._planner.handle_plan_tasks(
            arguments,
            realtime=self._realtime,
            room_id=room_id,
            channel_id=self.channel_id,
            # RFC §9.2 ON_PLAN_UPDATED — the hook surface for the plan the
            # ephemeral event carries to live UIs.
            on_plan_updated=self._plan_updated_hook,
        )


_PREVIEW_CHARS = 500


def _preview(value: Any) -> str:
    """A bounded one-line rendering of a tool payload for DEBUG logs."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    if len(text) <= _PREVIEW_CHARS:
        return text
    return f"{text[:_PREVIEW_CHARS]}… ({len(text)} chars)"
