"""The call signatures of roomkit's public names, and the changes that break a caller.

``tests/public_api_signatures.json`` holds a snapshot of every class and function
roomkit defines and a package exports in its ``__all__`` (a name re-exported
from another library is left out): a function's parameters, a class's
constructor and its public methods and properties, an enum's members. Each
parameter is written as ``inspect`` writes it, without its annotation:
``name``, ``name=default``, ``*args``, ``**kwargs``, and the ``/`` and ``*``
markers. A member's first item says what it is: ``def``, ``async def``,
``async gen def``, ``classmethod``, ``staticmethod`` (each possibly ``async``),
``property`` or ``member``.

``test_public_api.py`` compares the code with the snapshot. A change that can
break a caller fails: a name, member or parameter removed or renamed, a
parameter made required or moved, a default changed, a method made async, an
enum member's value changed. An addition a caller cannot notice passes and
needs no entry. The commit that breaks on purpose rewrites the snapshot with
``make api-snapshot`` and carries its BREAKING entry.

``tests/public_api_names.json`` holds every name each package exports in its
``__all__``, constants and re-exports included: ``test_public_api.py`` fails
on one dropped from its package, which a name the signatures resolve by
import alone would not show. ``make api-snapshot`` rewrites both files in an
environment of its own installed with ``--extra all``, so a name only an
extra of that set resolves (``roomkit.webrtc``, which needs fastapi) is
pinned too; the extras outside it (livekit, docs, pynacl, webtransport, the
local model runtimes neutts, qwen-*, pocket-tts, phonon and vui, and a few
others) are not installed there. Written elsewhere, a package or a name whose
optional dependency the environment lacks keeps what the files hold for it,
and the guard leaves it out; a roomkit module that is gone drops its names.

Annotations are left out: whether a type widens or narrows is not read
mechanically. What a call does is not here either; a change of meaning under
the same signature is the conformance scenarios' to catch.
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import inspect
import json
import re
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import pydantic

import roomkit

SNAPSHOT = Path(__file__).with_name("public_api_signatures.json")
NAMES = Path(__file__).with_name("public_api_names.json")

_PACKAGE_ROOT = Path(roomkit.__file__).parent
_P = inspect.Parameter
_POSITIONAL = (_P.POSITIONAL_ONLY, _P.POSITIONAL_OR_KEYWORD)
_VARIADIC = (_P.VAR_POSITIONAL, _P.VAR_KEYWORD)
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_ABSENT = object()  # the name is not there
_OPTIONAL = object()  # the name needs an optional dependency the environment lacks

Spec = list[str]


@dataclass(frozen=True)
class _Param:
    name: str
    kind: inspect._ParameterKind
    default: str | None  # None: the parameter is required


def build() -> dict[str, dict[str, Spec]]:
    """The snapshot of the code as it is, keyed by the shortest path a name is
    exported under."""
    paths: dict[int, str] = {}
    objects: dict[int, Any] = {}
    for package, module in _packages():
        for name in getattr(module, "__all__", ()):
            obj = _resolve(module, name)
            if not _is_roomkit(obj):
                continue
            path = f"{package}.{name}"
            known = paths.get(id(obj))
            if known is None or (path.count("."), path) < (known.count("."), known):
                paths[id(obj)] = path
                objects[id(obj)] = obj
    return {paths[key]: describe(objects[key]) for key in sorted(paths, key=paths.__getitem__)}


def breaks_since_snapshot() -> list[str]:
    """What the code changed since the snapshot that can break a caller."""
    found: list[str] = []
    for path, old in json.loads(SNAPSHOT.read_text(encoding="utf-8")).items():
        package, _, name = path.rpartition(".")
        try:
            module = importlib.import_module(package)
        except ModuleNotFoundError as missing:
            _raise_unless_optional(missing)
            continue
        obj = _resolve(module, name)
        if obj is not _OPTIONAL:
            found += object_breaks(path, old, None if obj is _ABSENT else describe(obj))
    return found


def describe(obj: Any) -> dict[str, Spec]:
    """A class's or a function's members, each as the snapshot writes it."""
    if not inspect.isclass(obj):
        spec = _callable_spec(obj, "def", drop_first=False)
        return {} if spec is None else {"__call__": spec}
    members: dict[str, Spec] = {}
    if issubclass(obj, enum.Enum):
        members.update({name: ["member", _repr(m.value)] for name, m in obj.__members__.items()})
    elif _builds_itself(obj):
        if issubclass(obj, pydantic.BaseModel) and not obj.__pydantic_complete__:
            # A model whose forward references wait for its first use reads
            # as `(**data)` until then: complete it, as that use would.
            obj.model_rebuild(raise_errors=False)
        with contextlib.suppress(TypeError, ValueError):
            members["__init__"] = ["def", *_tokens(inspect.signature(obj), drop_first=False)]
    for klass in obj.__mro__:
        if not _is_roomkit(klass):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in members:
                continue
            spec = _member_spec(value)
            if spec is not None:
                members[name] = spec
    return dict(sorted(members.items()))


def object_breaks(path: str, old: dict[str, Spec], new: dict[str, Spec] | None) -> list[str]:
    """The changes from ``old`` to ``new`` that can break a caller of ``path``."""
    if new is None:
        return [f"{path}: no longer exported"]
    found: list[str] = []
    for member, before in old.items():
        where = path if member == "__call__" else f"{path}.{member}"
        after = new.get(member)
        if after == before:
            continue
        if after is None:
            found.append(f"{where}: removed")
        elif after[0] != before[0]:
            found.append(f"{where}: {before[0]} became {after[0]}")
        elif before[0] == "member":
            if after[1] != before[1]:
                found.append(f"{where}: value {before[1]} became {after[1]}")
        elif before[0] != "property":
            found += [f"{where}: {m}" for m in call_breaks(before[1:], after[1:])]
    return found


def call_breaks(old: Spec, new: Spec) -> list[str]:
    """The changes from parameters ``old`` to ``new`` that break a call that
    worked: each parameter is matched by its position when a caller can pass
    it by position, by its name otherwise."""
    before, after = _params(old), _params(new)
    found: list[str] = []
    old_positional = [p for p in before if p.kind in _POSITIONAL]
    new_positional = [p for p in after if p.kind in _POSITIONAL]
    by_name = {q.name: q for q in after if q.kind not in _VARIADIC}
    for i, p in enumerate(old_positional):
        q = new_positional[i] if i < len(new_positional) else None
        if q is not None and (p.kind is _P.POSITIONAL_ONLY or q.name == p.name):
            if p.kind is _P.POSITIONAL_OR_KEYWORD and q.kind is _P.POSITIONAL_ONLY:
                found.append(f"{p.name!r} can no longer be passed by name")
            else:
                found += _default_breaks(p, q)
        elif p.name not in by_name:
            found.append(f"{p.name!r} removed")
        else:
            found.append(f"{p.name!r} is no longer positional parameter {i + 1}")
    for p in before:
        if p.kind is not _P.KEYWORD_ONLY:
            continue
        q = by_name.get(p.name)
        if q is None:
            found.append(f"{p.name!r} removed")
        elif q.kind is _P.POSITIONAL_ONLY:
            found.append(f"{p.name!r} can no longer be passed by name")
        else:
            found += _default_breaks(p, q)
    for kind, label in ((_P.VAR_POSITIONAL, "*args"), (_P.VAR_KEYWORD, "**kwargs")):
        if any(p.kind is kind for p in before) and not any(q.kind is kind for q in after):
            found.append(f"{label} removed")
    old_names = {p.name for p in before}
    matched = {id(q) for q in new_positional[: len(old_positional)]}
    for q in after:
        required = q.default is None and q.kind not in _VARIADIC
        if required and id(q) not in matched and q.name not in old_names:
            found.append(f"new required parameter {q.name!r}")
    return found


def write() -> None:
    """Rewrite the snapshot from the code, one member per line. A name whose
    optional dependency the environment lacks keeps the entry the snapshot
    holds for it."""
    snapshot = build()
    if SNAPSHOT.exists():
        for path, members in json.loads(SNAPSHOT.read_text(encoding="utf-8")).items():
            if path not in snapshot and _optional_here(path):
                snapshot[path] = members
    blocks = []
    for path, members in sorted(snapshot.items()):
        lines = [
            f"    {json.dumps(name)}: {json.dumps(spec, ensure_ascii=False)}"
            for name, spec in members.items()
        ]
        body = "{\n" + ",\n".join(lines) + "\n  }" if lines else "{}"
        blocks.append(f"  {json.dumps(path)}: {body}")
    SNAPSHOT.write_text("{\n" + ",\n".join(blocks) + "\n}\n", encoding="utf-8")


def exported_names() -> dict[str, list[str]]:
    """Every name each package exports, sorted, by package. A package whose
    optional dependency the environment lacks keeps the names
    :data:`NAMES` holds for it."""
    held = json.loads(NAMES.read_text(encoding="utf-8")) if NAMES.exists() else {}
    names: dict[str, list[str]] = {}
    for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
        package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
        try:
            module = importlib.import_module(package)
        except ModuleNotFoundError as missing:
            _raise_unless_optional(missing)
            if package in held:
                names[package] = held[package]
            continue
        exported = sorted(getattr(module, "__all__", ()))
        if exported:
            names[package] = exported
    return dict(sorted(names.items()))


def write_names() -> None:
    """Rewrite :data:`NAMES` from the code."""
    NAMES.write_text(json.dumps(exported_names(), indent=2) + "\n", encoding="utf-8")


def _default_breaks(p: _Param, q: _Param) -> list[str]:
    if p.default is None:
        return []
    if q.default is None:
        return [f"{p.name!r} lost its default {p.default}"]
    if q.default != p.default:
        return [f"{p.name!r} default {p.default} became {q.default}"]
    return []


def _params(spec: Spec) -> list[_Param]:
    params: list[_Param] = []
    kind = _P.POSITIONAL_OR_KEYWORD
    for token in spec:
        if token == "/":
            params = [_Param(p.name, _P.POSITIONAL_ONLY, p.default) for p in params]
        elif token == "*":
            kind = _P.KEYWORD_ONLY
        elif token.startswith("**"):
            params.append(_Param(token[2:], _P.VAR_KEYWORD, None))
        elif token.startswith("*"):
            params.append(_Param(token[1:], _P.VAR_POSITIONAL, None))
            kind = _P.KEYWORD_ONLY
        else:
            name, eq, default = token.partition("=")
            params.append(_Param(name, kind, default if eq else None))
    return params


def _tokens(signature: inspect.Signature, *, drop_first: bool) -> Spec:
    params = list(signature.parameters.values())
    if drop_first and params and params[0].kind in _POSITIONAL:
        params = params[1:]
    tokens: Spec = []
    keyword_only = False
    for i, p in enumerate(params):
        if p.kind is _P.VAR_POSITIONAL:
            tokens.append(f"*{p.name}")
            keyword_only = True
            continue
        if p.kind is _P.VAR_KEYWORD:
            tokens.append(f"**{p.name}")
            continue
        if p.kind is _P.KEYWORD_ONLY and not keyword_only:
            tokens.append("*")
            keyword_only = True
        tokens.append(p.name if p.default is _P.empty else f"{p.name}={_repr(p.default)}")
        following = params[i + 1].kind if i + 1 < len(params) else None
        if p.kind is _P.POSITIONAL_ONLY and following is not _P.POSITIONAL_ONLY:
            tokens.append("/")
    return tokens


def _member_spec(value: Any) -> Spec | None:
    if isinstance(value, property | cached_property):
        return ["property"]
    if isinstance(value, staticmethod):
        return _callable_spec(value.__func__, "staticmethod", drop_first=False)
    if isinstance(value, classmethod):
        return _callable_spec(value.__func__, "classmethod", drop_first=True)
    if inspect.isfunction(value):
        return _callable_spec(value, "def", drop_first=True)
    return None


def _callable_spec(fn: Any, binding: str, *, drop_first: bool) -> Spec | None:
    spec = _cached_spec(fn, binding, drop_first)
    return None if spec is None else list(spec)


@cache
def _cached_spec(fn: Any, binding: str, drop_first: bool) -> tuple[str, ...] | None:
    """A method a class inherits is read once, not once per subclass."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    flavour = (
        "async gen "
        if inspect.isasyncgenfunction(fn)
        else "async "
        if inspect.iscoroutinefunction(fn)
        else ""
    )
    return (flavour + binding, *_tokens(signature, drop_first=drop_first))


def _builds_itself(cls: type) -> bool:
    """Whether roomkit decides how ``cls`` is constructed: an ``__init__`` or
    ``__new__`` of its own or a pydantic model's fields. A constructor inherited
    from the standard library is left out; its signature varies between Python
    versions."""
    if issubclass(cls, pydantic.BaseModel):
        return True
    for klass in cls.__mro__:
        if "__init__" in vars(klass) or "__new__" in vars(klass):
            return _is_roomkit(klass)
    return False


def _repr(value: Any) -> str:
    if isinstance(value, set | frozenset):
        return f"{type(value).__name__}({sorted(map(repr, value))})"
    return _ADDRESS.sub("", repr(value))


def _packages() -> list[tuple[str, Any]]:
    found = []
    for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
        package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
        try:
            found.append((package, importlib.import_module(package)))
        except ModuleNotFoundError as missing:
            _raise_unless_optional(missing)
    return found


def _optional_here(path: str) -> bool:
    """Whether ``path`` needs an optional dependency the environment lacks. A
    roomkit module that is gone is no optional dependency: its names drop out
    of the snapshot."""
    package, _, name = path.rpartition(".")
    try:
        module = importlib.import_module(package)
        getattr(module, name)
    except ImportError as missing:
        return not (missing.name or "").startswith("roomkit")
    except AttributeError:
        return False
    return False


def _resolve(module: Any, name: str) -> Any:
    try:
        return getattr(module, name)
    except ImportError as missing:
        _raise_unless_optional(missing)
        return _OPTIONAL
    except AttributeError:
        return _ABSENT


def _is_roomkit(obj: Any) -> bool:
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        return False
    module = getattr(obj, "__module__", None) or ""
    return module == "roomkit" or module.startswith("roomkit.")


def _raise_unless_optional(missing: ImportError) -> None:
    if (missing.name or "").startswith("roomkit"):
        raise missing


if __name__ == "__main__":
    write()
    write_names()
