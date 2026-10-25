"""Identity resolution: who sent an inbound message."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.identity.base import IdentityResolver

if TYPE_CHECKING:
    from roomkit.identity.mock import MockIdentityResolver
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"MockIdentityResolver": "roomkit.identity.mock"}
    )

__all__ = ["IdentityResolver", "MockIdentityResolver"]
