"""Tests for public API surface."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType

import roomkit

_PACKAGE_ROOT = Path(roomkit.__file__).parent


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
