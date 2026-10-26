"""The keys of a room's private state (RFC §14.1): what the runtime keeps for
itself, apart from the room's metadata, which clients may read.

Each key reserves the same key in the room's metadata: a store that keeps no
private state of its own keeps the value there
(:meth:`~roomkit.store.base.ConversationStore.get_private_state`), a room
stored before the key moved still holds it there, and the runtime merges what
it finds there (a register of authors, RFC §10.1 step 12). So no writer but
the runtime writes it there (:mod:`roomkit.core._room_metadata`, RMK-978).

A new private-state key is defined here, so that it is reserved with the
others.
"""

from __future__ import annotations

AUTHOR_REGISTER = "author_register"
"""The room's register of authors (RFC §10.1 step 12); the room metadata key
it was kept under before, which a store without private state of its own
keeps it under still."""

SPEAK_QUEUE_RESULTS = "_speak_queue_results"
"""A discussion's task results to post (RFC §19.7.5 rule 16)."""

PRIVATE_STATE_KEYS = frozenset({AUTHOR_REGISTER, SPEAK_QUEUE_RESULTS})
"""Every key of a room's private state, each reserved in its metadata."""
