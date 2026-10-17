"""What a turn's error keeps across the turn backend (RFC §10.6.8).

A turn runs on its executor and its error reaches the room owner as a record,
rebuilt there: the caller holds an error of the same class, message and
attributes, its causes chained, never the object the turn raised.
"""

from __future__ import annotations


def same_error(received: BaseException | None, raised: BaseException) -> bool:
    """Whether *received* is *raised* as it crosses the backend: same class,
    message and attributes."""
    return (
        received is not None
        and type(received) is type(raised)
        and str(received) == str(raised)
        and vars(received) == vars(raised)
    )
