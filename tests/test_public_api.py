"""Tests for public API surface."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

import roomkit
from tests.api_signatures import breaks_since_snapshot, call_breaks, object_breaks

_PACKAGE_ROOT = Path(roomkit.__file__).parent
_EXPORTED = Path(__file__).with_name("public_api_names.json")


class TestPublicAPI:
    def test_version_string(self) -> None:
        assert isinstance(roomkit.__version__, str)
        # PEP 440 release identifier: ``MAJOR.MINOR.PATCH`` with an
        # optional pre-release suffix (``a``/``b``/``rc`` + digits) and
        # an optional post/dev tag. Pinning the exact string here would
        # tie every release commit to a test edit — keep this loose so
        # the assertion catches a missing or empty version, not a bump.
        assert re.match(
            r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$",
            roomkit.__version__,
        ), f"unexpected __version__ {roomkit.__version__!r}"

    def test_every_name_a_package_exports_resolves(self) -> None:
        """A name in ``__all__`` that does not resolve fails only whoever imports
        it, so the list of roomkit and of each of its subpackages is read here,
        once.

        A package, or a lazily imported name, that needs an optional dependency
        the environment lacks is left out; a missing roomkit module is a failure.
        """
        unresolved: dict[str, list[str]] = {}
        for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
            package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
            try:
                module = importlib.import_module(package)
            except ModuleNotFoundError as missing:
                _raise_unless_optional(missing)
                continue
            names = [n for n in getattr(module, "__all__", ()) if not _resolves(module, n)]
            if names:
                unresolved[package] = names

        assert unresolved == {}

    def test_no_name_a_package_exported_is_dropped(self) -> None:
        """Every name a package exported stays exported: an application imports
        it from there. tests/public_api_names.json lists them; the commit that
        breaks an import on purpose drops its name there, and a new export
        needs no entry.

        A package whose optional dependency the environment lacks is left out,
        as test_every_name_a_package_exports_resolves leaves it out.
        """
        dropped: dict[str, list[str]] = {}
        for package, names in json.loads(_EXPORTED.read_text()).items():
            try:
                module = importlib.import_module(package)
            except ModuleNotFoundError as missing:
                _raise_unless_optional(missing)
                continue
            gone = sorted(set(names) - set(getattr(module, "__all__", ())))
            if gone:
                dropped[package] = gone

        assert dropped == {}

    def test_no_signature_change_breaks_a_caller(self) -> None:
        """A call that worked on a public class or function still works:
        tests/public_api_signatures.json holds their signatures, read as
        tests/api_signatures.py says. The commit that breaks a call on purpose
        runs `make api-snapshot` and carries its BREAKING entry; an addition
        needs no entry.
        """
        assert breaks_since_snapshot() == []

    def test_channels_importable_from_channels_package(self) -> None:
        """Every channel resolves from ``roomkit.channels``, factories included.

        The transport factories always lived here; the channel classes only in
        the top-level package, so a reader had no rule telling one from the
        other and half the documented imports raised ImportError.
        """
        import roomkit.channels as channels

        for name in (
            "ACPChannel",
            "AIChannel",
            "CLIChannel",
            "EmailChannel",
            "HTTPChannel",
            "RCSChannel",
            "RealtimeVoiceChannel",
            "SMSChannel",
            "TransportChannel",
            "VoiceChannel",
            "WebSocketChannel",
        ):
            obj = getattr(channels, name, None)
            assert obj is not None, f"roomkit.channels.{name} is missing"
            assert obj is getattr(roomkit, name), f"{name} differs from the top-level export"


def _resolves(module: ModuleType, name: str) -> bool:
    """Whether ``name`` resolves on ``module``; an absent optional dependency counts."""
    try:
        getattr(module, name)
    except ModuleNotFoundError as missing:
        _raise_unless_optional(missing)
    except AttributeError:
        return False
    return True


def _raise_unless_optional(missing: ModuleNotFoundError) -> None:
    if (missing.name or "").startswith("roomkit"):
        raise missing


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        (["a", "b=1"], ["a", "b=1", "*", "c=None"], None),
        (["a"], ["a", "b=None"], None),
        (["a"], ["a=None"], None),
        (["*", "a"], ["a"], None),
        (["a", "/"], ["b", "/"], None),
        (["a"], ["a", "**kwargs"], None),
        (["a", "b=1"], ["a"], "'b'"),
        (["a"], ["b"], "'a'"),
        (["a=200"], ["a=50"], "'a'"),
        (["a=None"], ["a"], "'a'"),
        (["a"], ["a", "b"], "'b'"),
        (["a"], ["a", "*", "b"], "'b'"),
        (["a", "b=None"], ["a", "*", "b=None"], "'b'"),
        (["a", "b=None"], ["a", "c=None", "b=None"], "'b'"),
        (["a"], ["a", "/"], "'a'"),
        (["*", "a=None"], [], "'a'"),
        (["a", "**kwargs"], ["a"], "**kwargs"),
        (["*args"], [], "*args"),
    ],
    ids=[
        "keyword added with a default",
        "positional appended with a default",
        "required gains a default",
        "keyword-only becomes positional",
        "positional-only renamed",
        "kwargs added",
        "removed",
        "renamed",
        "default changed",
        "default dropped",
        "required added",
        "required keyword-only added",
        "made keyword-only",
        "inserted before another",
        "made positional-only",
        "keyword-only removed",
        "kwargs removed",
        "args removed",
    ],
)
def test_a_signature_change_breaks_only_a_call_that_worked(
    old: list[str], new: list[str], named: str | None
) -> None:
    """Each parameter is matched by its position when a caller can pass it by
    position, by its name otherwise; an addition no caller can notice passes."""
    found = call_breaks(old, new)
    if named is None:
        assert found == []
    else:
        assert len(found) == 1 and named in found[0], found


@pytest.mark.parametrize(
    ("old", "new", "broken"),
    [
        ({"run": ["def", "x"]}, {"run": ["def", "x"], "stop": ["def"]}, False),
        ({"A": ["member", "'a'"]}, {"A": ["member", "'a'"], "B": ["member", "'b'"]}, False),
        ({"run": ["def", "x"]}, {}, True),
        ({"run": ["def", "x"]}, {"run": ["async def", "x"]}, True),
        ({"size": ["property"]}, {"size": ["def"]}, True),
        ({"A": ["member", "'a'"]}, {"A": ["member", "'b'"]}, True),
        ({"run": ["def", "x"]}, None, True),
    ],
    ids=[
        "method added",
        "enum member added",
        "method removed",
        "method made async",
        "property made a method",
        "enum value changed",
        "no longer exported",
    ],
)
def test_a_member_change_breaks_a_caller(
    old: dict[str, list[str]], new: dict[str, list[str]] | None, broken: bool
) -> None:
    assert bool(object_breaks("roomkit.X", old, new)) is broken
