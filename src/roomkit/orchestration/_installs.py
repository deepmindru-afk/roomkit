"""A strategy installed in several rooms wires its shared objects once (RFC §19.7).

A strategy is installed per room, but its agents, and a voice channel it wires,
serve every room they are attached to. Wrapping one of them again for a second
room stacks the wrappers: the outer one's work ends in the inner one's, which
does it again.
"""

from __future__ import annotations

import weakref
from typing import Any


def first_install(installed: weakref.WeakSet[Any], owner: Any) -> bool:
    """Whether *owner*, an object every room shares, is wired for the first
    time: a second room's install finds it in *installed* and wires nothing."""
    if owner in installed:
        return False
    installed.add(owner)
    return True
