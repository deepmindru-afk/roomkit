"""What an exception raised across the turn backend becomes on the wire, and
back (RFC §10.6.8).

Decoding builds new objects: nothing one side holds is handed to the other.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from roomkit.core.exceptions import RemoteTurnError
from roomkit.turns.wire import ErrorRecord

# The packages whose exception classes an owner rebuilds by type. Anything
# else comes back as a RemoteTurnError naming the class: a name read off the
# wire imports nothing and reaches only a class of an already loaded module.
_REBUILT_PACKAGES = ("roomkit", "builtins")


def error_record(exc: BaseException) -> ErrorRecord:
    """*exc* as it crosses, with the chain of its causes."""
    causes: list[ErrorRecord] = []
    cause = exc.__cause__
    while cause is not None and len(causes) < 16:
        causes.append(_one_record(cause))
        cause = cause.__cause__
    record = _one_record(exc)
    record.causes = causes
    record.traceback = "".join(traceback.format_exception(exc))
    return record


def _one_record(exc: BaseException) -> ErrorRecord:
    cls = type(exc)
    return ErrorRecord(
        kind=f"{cls.__module__}:{cls.__qualname__}",
        message=str(exc),
        args=[a if _is_json(a) else str(a) for a in exc.args],
        attributes={k: v for k, v in vars(exc).items() if _is_json(v)},
    )


def rebuild_error(record: ErrorRecord, *, where: str = "the turn's executor") -> BaseException:
    """One new exception per record, of the record's class when this side
    rebuilds it, with its causes chained; the note says *where* it was raised."""
    exc = _rebuild_one(record)
    current = exc
    for cause_record in record.causes:
        cause = _rebuild_one(cause_record)
        current.__cause__ = cause
        current = cause
    if record.traceback:
        exc.add_note(f"Raised on {where}:\n{record.traceback.rstrip()}")
    return exc


def _rebuild_one(record: ErrorRecord) -> BaseException:
    cls = _exception_class(record.kind)
    if cls is None:
        return RemoteTurnError(record.message, kind=record.kind)
    try:
        exc = cls.__new__(cls, *record.args)
    except Exception:
        return RemoteTurnError(record.message, kind=record.kind)
    exc.args = tuple(record.args)
    exc.__dict__.update(record.attributes)
    return exc


def _exception_class(kind: str) -> type[Exception] | None:
    """The class *kind* names: an ``Exception`` defined at the top of a
    loaded module of RoomKit or Python's builtins; ``None`` for any other
    (an interpreter exit or a cancellation is never raised from a name)."""
    module_name, _, name = kind.partition(":")
    if module_name.split(".", 1)[0] not in _REBUILT_PACKAGES or not name.isidentifier():
        return None
    target = getattr(sys.modules.get(module_name), name, None)
    if not isinstance(target, type) or not issubclass(target, Exception):
        return None
    return target if target.__module__ == module_name else None


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
