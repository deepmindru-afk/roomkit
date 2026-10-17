"""Register a hook on the side its trigger fires on (RFC §10.6.6), for tests
parametrized over triggers of both sides."""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit
from roomkit.core.hooks import hook_decorator
from roomkit.models.enums import HookTrigger


def hook_on_its_side(kit: RoomKit, trigger: HookTrigger | str, *args: Any, **kwargs: Any) -> Any:
    """A hook decorator registering on the engine *trigger* fires on."""
    trigger = HookTrigger(trigger)
    return hook_decorator(kit.hooks_for(trigger).register, trigger, *args, **kwargs)
