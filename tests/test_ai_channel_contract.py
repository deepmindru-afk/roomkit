"""What AIChannel's mixins call on each other is implemented, of the declared
kind (RMK-315).

``_AIChannelContract`` declares those members for the type checker, which
checks each implementation's signature against it. What the type checker cannot
see is a member nothing implements, or one implemented as another kind (a
method for a property, a coroutine for a plain call): the contract would stand
in for it. This checks both at runtime, on AIChannel and its subclasses.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from roomkit.channels._ai_contract import _AIChannelContract
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel

_MEMBERS = {
    name: member
    for name, member in vars(_AIChannelContract).items()
    if not name.startswith("__") and (callable(member) or isinstance(member, property))
}


def _kind(member: Any) -> str:
    if isinstance(member, property):
        return "property"
    if inspect.iscoroutinefunction(member):
        return "coroutine"
    if inspect.isasyncgenfunction(member):
        return "async generator"
    return "call"


def _declared_kind(member: Any) -> str:
    """A plain ``def`` returning an async iterator declares an async generator."""
    kind = _kind(member)
    if kind != "call":
        return kind
    returns = str(inspect.signature(member).return_annotation)
    return "async generator" if returns.startswith(("AsyncIterator", "AsyncGenerator")) else kind


def test_the_contract_declares_members() -> None:
    assert len(_MEMBERS) >= 29


@pytest.mark.parametrize("channel", [AIChannel, Agent], ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", sorted(_MEMBERS))
def test_every_member_is_implemented_as_declared(channel: type, name: str) -> None:
    implementation = inspect.getattr_static(channel, name, None)

    assert implementation is not None, f"{channel.__name__} implements no {name}"
    assert implementation is not _MEMBERS[name], f"{name} resolves to the contract"
    assert _kind(implementation) == _declared_kind(_MEMBERS[name])


def test_no_mixin_derives_from_the_contract_at_runtime() -> None:
    assert _AIChannelContract not in AIChannel.__mro__
