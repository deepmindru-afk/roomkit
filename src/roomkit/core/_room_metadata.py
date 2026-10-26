"""The keys of a room's metadata only the framework writes (RFC §5.1, §14.1).

Two kinds:

- the implementation's namespace, ``roomkit:`` (RFC §5.1): what the
  framework keeps for the room, a detach's revocation
  (``roomkit:detached_channels``, §7.5 rule 7) and what the room keeps for a
  channel (``roomkit:channel_state:<id>``, §10.6.2);
- a key of the room's private state (RFC §14.1,
  :mod:`roomkit.core._private_state`): the register of authors, a
  discussion's results to post.

The kit's ``create_room()`` and ``update_room_metadata()``, called from a
hook's body too, refuse them (:func:`refuse_framework_keys`), and so does a
turn's ``TurnHost.update_room_metadata``; a channel's ``metadata_updates``
have them dropped with a warning, the rest applied (§10.1 step 15,
:func:`channel_updates`). The store's ``update_room()`` and
``patch_room_metadata()`` stay open: they sit below the kit's API, where
migrations and the runtime write these keys (a detach's revocation, a
channel's state, the base store's private state), and no channel output or
hook result reaches them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from roomkit.core._private_state import PRIVATE_STATE_KEYS

logger = logging.getLogger("roomkit.event_router")

FRAMEWORK_NAMESPACE = "roomkit:"
"""The prefix of the room-metadata keys the framework keeps for the room."""


def the_framework_s(key: str) -> bool:
    """Whether only the framework writes *key* of a room's metadata."""
    return key.startswith(FRAMEWORK_NAMESPACE) or key in PRIVATE_STATE_KEYS


def framework_keys_in(metadata: Mapping[str, object]) -> list[str]:
    """The keys of *metadata* only the framework writes, sorted."""
    return sorted(key for key in metadata if the_framework_s(key))


def refuse_framework_keys(metadata: Mapping[str, object]) -> None:
    """Refuse an integrator's write of *metadata* that names a key only the
    framework writes, before anything is written (RFC §5.1, §14.1). The
    store's own writes stay open (module docstring).

    Raises:
        ValueError: a key under ``roomkit:`` or of the room's private state.
    """
    reserved = framework_keys_in(metadata)
    if reserved:
        raise ValueError(f"Room metadata {reserved} is the framework's, written by it alone")


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
