"""The keys of a room's private state (RFC §14.1): what the runtime keeps for
itself, apart from the room's metadata, which clients may read.

Each key reserves the same key in the room's metadata: a store that keeps no
private state of its own keeps the value there
(:meth:`~roomkit.store.base.ConversationStore.get_private_state`), a room
stored before the key moved still holds it there, and the runtime merges what
it finds there (a register of authors, RFC §10.1 step 12). So no writer but
the runtime writes it there: :func:`refuse_private_state` refuses it to an
integrator's write, and a channel's metadata updates drop it (RMK-978).

A new private-state key is defined here, so that it is reserved with the
others.
"""

from __future__ import annotations

from collections.abc import Mapping

AUTHOR_REGISTER = "author_register"
"""The room's register of authors (RFC §10.1 step 12); the room metadata key
it was kept under before, which a store without private state of its own
keeps it under still."""

SPEAK_QUEUE_RESULTS = "_speak_queue_results"
"""A discussion's task results to post (RFC §19.7.5 rule 16)."""

PRIVATE_STATE_KEYS = frozenset({AUTHOR_REGISTER, SPEAK_QUEUE_RESULTS})
"""Every key of a room's private state, each reserved in its metadata."""


def private_state_keys_in(metadata: Mapping[str, object]) -> list[str]:
    """The keys of *metadata* a private-state key reserves, sorted."""
    return sorted(key for key in metadata if key in PRIVATE_STATE_KEYS)


def refuse_private_state(metadata: Mapping[str, object]) -> None:
    """Refuse *metadata* that writes a key a private-state key reserves: only
    the runtime writes it (RMK-978).

    Raises:
        ValueError: a key of :data:`PRIVATE_STATE_KEYS`.
    """
    reserved = private_state_keys_in(metadata)
    if reserved:
        raise ValueError(
            f"Room metadata {reserved} is the room's private state, written by the runtime alone"
        )
