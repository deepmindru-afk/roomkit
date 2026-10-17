"""Register a hook on the side its trigger fires on (RFC §10.6.6), for tests
parametrized over triggers of both sides."""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit
from roomkit.core.hooks import TURN_TRIGGERS
from roomkit.models.enums import HookTrigger


def hook_on_its_side(kit: RoomKit, trigger: HookTrigger | str, *args: Any, **kwargs: Any) -> Any:
    """``kit.turn_executor.hook`` for a trigger a turn fires, ``kit.hook``
    for an owner's."""
    side = kit.turn_executor if HookTrigger(trigger) in TURN_TRIGGERS else kit
    return side.hook(HookTrigger(trigger), *args, **kwargs)
