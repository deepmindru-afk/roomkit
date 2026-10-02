"""The callbacks the framework hands an AIChannel when it registers it.

``register_channel`` builds each one from the room's hooks and sets it on the
channel; the mixins that call them read these types. Aliases only: nothing
here runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.models.tool_call import AIGenerationEvent, ToolCallEvent
    from roomkit.tools.external import BeforeToolDecision

type BeforeGenerationHook = Callable[[AIGenerationEvent], Awaitable[SyncPipelineResult]]
"""BEFORE_AI_GENERATION for a turn's context: allowed or blocked, and why."""

type BeforeToolCallHook = Callable[[ToolCallEvent], Awaitable[BeforeToolDecision]]
"""BEFORE_TOOL_USE for one call: its decision."""

type ThinkingHook = Callable[[str, str, int], Awaitable[None]]
"""ON_AI_THINKING: ``(room_id, thinking, round_idx)``."""

type PlanUpdatedHook = Callable[[str, list[dict[str, Any]]], Awaitable[None]]
"""ON_PLAN_UPDATED: ``(room_id, tasks)``."""

type ToolUsageLoader = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""A room's stored tool calls, read to rebuild the channel's tool memory."""
