"""A channel's updates of its room's metadata (RFC §10.1 step 15, §6.1).

A channel's output may carry ``metadata_updates``, which the framework
writes on the room once the delivery set has run. A key only the framework
writes is not a channel's to write: the implementation's namespace
(``roomkit:``, RFC §5.1) and a key of the room's private state (RFC §14.1,
:mod:`roomkit.core._private_state`). Each is dropped with a warning, the
rest applied.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from roomkit.core._private_state import PRIVATE_STATE_KEYS
from roomkit.turns.host import FRAMEWORK_KEYS

logger = logging.getLogger("roomkit.event_router")


def the_framework_s(key: str) -> bool:
    """Whether only the framework writes *key* of a room's metadata."""
    return key.startswith(FRAMEWORK_KEYS) or key in PRIVATE_STATE_KEYS


def channel_updates(
    updates: Mapping[str, Any], *, channel_id: str, room_id: str
) -> dict[str, Any]:
    """*updates*, a channel output's room metadata updates, without the keys
    only the framework writes, each dropped with a warning."""
    kept: dict[str, Any] = {}
    for key, value in updates.items():
        if the_framework_s(key):
            logger.warning(
                "Room %s drops the metadata update %r of channel %s: only the framework writes it",
                room_id,
                key,
                channel_id,
                extra={"room_id": room_id, "channel_id": channel_id},
            )
            continue
        kept[key] = value
    return kept
