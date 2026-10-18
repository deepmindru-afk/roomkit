"""What an exception raised across the turn backend becomes on the wire, and
back (RFC §10.6.8).

Decoding builds new objects: nothing one side holds is handed to the other.
"""

from __future__ import annotations

import functools
import json
import sys
import traceback
from collections.abc import Iterable
from typing import Any

from roomkit.core.exceptions import RemoteTurnError
from roomkit.turns.wire import ErrorRecord

# The packages whose exception classes every side rebuilds by type; a host
# adds its own classes (ErrorClasses). A name read off the wire imports
# nothing and reaches only a class of an already loaded module.
_REBUILT_PACKAGES = ("roomkit", "builtins")

# At most this many causes, and this many ancestors, cross with an error.
_CHAIN_LIMIT = 16


def error_record(exc: BaseException) -> ErrorRecord:
    """*exc* as it crosses, with the chain of its causes."""
    causes: list[ErrorRecord] = []
    cause = exc.__cause__
    while cause is not None and len(causes) < _CHAIN_LIMIT:
        causes.append(_one_record(cause))
        cause = cause.__cause__
    record = _one_record(exc)
    record.causes = causes
    record.traceback = "".join(traceback.format_exception(exc))
    return record


def _one_record(exc: BaseException) -> ErrorRecord:
    """*exc* alone; an error rebuilt as a ``RemoteTurnError`` crosses again as
    the error it stands for."""
    if isinstance(exc, RemoteTurnError):
        kind, bases = exc.kind, list(exc.bases)
        own = {"kind", "bases"}
    else:
        cls = type(exc)
        kind, bases, own = _kind(cls), _ancestors(cls), set()
    return ErrorRecord(
        kind=kind,
        message=str(exc),
        args=[a if _is_json(a) else str(a) for a in exc.args],
        attributes={k: v for k, v in vars(exc).items() if k not in own and _is_json(v)},
        bases=bases,
    )


def _kind(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _ancestors(cls: type) -> list[str]:
    """The classes *cls* derives from below ``Exception``, nearest first."""
    below = [c for c in cls.__mro__[1:] if issubclass(c, Exception) and c is not Exception]
    return [_kind(c) for c in below[:_CHAIN_LIMIT]]


class ErrorClasses:
    """The exception classes one side of the turn backend rebuilds by type
    (RFC §10.6.8): RoomKit's, Python's, and those its host declared, each
    with its subclasses.

    Args:
        declared: The host's classes, beyond RoomKit's and Python's own.

    Raises:
        TypeError: An entry of *declared* is not an ``Exception`` class.
    """

    def __init__(self, declared: Iterable[type[Exception]] = ()) -> None:
        self._declared = tuple(declared)
        for cls in self._declared:
            if not isinstance(cls, type) or not issubclass(cls, Exception):
                raise TypeError(f"error_classes takes Exception classes, not {cls!r}")

    def rebuild(self, record: ErrorRecord, *, where: str = "the turn's executor") -> BaseException:
        """One new exception per record, its causes chained; the note says
        *where* it was raised."""
        exc = self._one(record)
        current = exc
        for cause_record in record.causes:
            cause = self._one(cause_record)
            current.__cause__ = cause
            current = cause
        if record.traceback:
            exc.add_note(f"Raised on {where}:\n{record.traceback.rstrip()}")
        return exc

    def _one(self, record: ErrorRecord) -> BaseException:
        """The record's error, of its class when this side rebuilds it; else a
        ``RemoteTurnError`` also of the nearest ancestor this side rebuilds."""
        cls = self._rebuilt(record.kind)
        if cls is not None:
            exc = _of_class(cls, record)
            if exc is not None:
                return exc
        return _remote(record, filter(None, map(self._rebuilt, record.bases)))

    def _rebuilt(self, kind: str) -> type[Exception] | None:
        """The class *kind* names when this side rebuilds it."""
        cls = _loaded_class(kind)
        if cls is None:
            return None
        if cls.__module__.split(".", 1)[0] in _REBUILT_PACKAGES or issubclass(cls, self._declared):
            return cls
        return None


ROOMKIT_ERRORS = ErrorClasses()
"""What a side whose host declared no class rebuilds."""


def _of_class(cls: type[Exception], record: ErrorRecord) -> BaseException | None:
    """A new *cls* carrying the record's arguments and attributes, built
    without its ``__init__``; ``None`` when the class refuses them."""
    try:
        exc = cls.__new__(cls, *record.args)
    except Exception:
        return None
    exc.args = tuple(record.args)
    exc.__dict__.update(record.attributes)
    return exc


def _remote(record: ErrorRecord, ancestors: Iterable[type[Exception]]) -> RemoteTurnError:
    """The record's error as a ``RemoteTurnError`` that is also the first of
    *ancestors* it can be built as."""
    built = (_new_remote(ancestor, record.message) for ancestor in ancestors)
    exc = next((e for e in built if e is not None), None)
    if exc is None:
        exc = RemoteTurnError.__new__(RemoteTurnError, record.message)
    exc.args = (record.message,)
    exc.__dict__.update(record.attributes)
    exc.kind = record.kind
    exc.bases = tuple(record.bases)
    return exc


def _new_remote(ancestor: type[Exception], message: str) -> RemoteTurnError | None:
    """A new ``RemoteTurnError`` that is also an *ancestor*; ``None`` when
    the ancestor's ``__new__`` refuses a message alone (``ExceptionGroup``)."""
    cls = _remote_class(ancestor)
    try:
        return cls.__new__(cls, message)
    except Exception:
        return None


@functools.cache
def _remote_class(ancestor: type[Exception]) -> type[RemoteTurnError]:
    """``RemoteTurnError``, made a subclass of *ancestor* too: one class per
    ancestor; ``RemoteTurnError`` itself when the two cannot share a layout."""
    if issubclass(RemoteTurnError, ancestor):
        return RemoteTurnError
    namespace = {
        "__module__": RemoteTurnError.__module__,
        "__qualname__": RemoteTurnError.__qualname__,
    }
    try:
        return type(RemoteTurnError.__name__, (RemoteTurnError, ancestor), namespace)
    except TypeError:
        return RemoteTurnError


def _loaded_class(kind: str) -> type[Exception] | None:
    """The class *kind* names: an ``Exception`` defined at the top of a
    loaded module, read from the module's namespace, so that no name off the
    wire imports anything (nor runs a module's ``__getattr__``); ``None`` for
    any other (an interpreter exit or a cancellation is never raised from a
    name)."""
    module_name, _, name = kind.partition(":")
    namespace = getattr(sys.modules.get(module_name), "__dict__", None)
    if not isinstance(namespace, dict) or not name.isidentifier():
        return None
    target = namespace.get(name)
    if not isinstance(target, type) or not issubclass(target, Exception):
        return None
    return target if target.__module__ == module_name else None


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
