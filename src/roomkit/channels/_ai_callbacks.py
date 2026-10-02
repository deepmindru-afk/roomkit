"""The callbacks the framework hands an AIChannel when it registers it, for
those no public alias already names (``BeforeToolCallback``,
``PlanUpdatedCallback`` and ``AfterResponseCallback`` do).

``register_channel`` builds each one from the room's hooks and sets it on the
channel; the mixins that call them read these types.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.models.tool_call import AIGenerationEvent

BeforeGenerationHook = Callable[["AIGenerationEvent"], Awaitable["SyncPipelineResult"]]
"""BEFORE_AI_GENERATION for a turn's context: allowed or blocked, and why. The
public ``BeforeGenerationCallback`` leaves the result untyped, which models
cannot name without importing core."""

ThinkingHook = Callable[[str, str, int], Awaitable[None]]
"""ON_AI_THINKING: ``(room_id, thinking, round_idx)``."""

ToolUsageLoader = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""A room's stored tool calls, read to rebuild the channel's tool memory."""
