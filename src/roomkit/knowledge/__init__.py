"""Knowledge retrieval sources for AI context enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource
from roomkit.knowledge.mock import MockKnowledgeSource

if TYPE_CHECKING:
    from roomkit.knowledge.postgres import PostgresKnowledgeSource
else:
    __getattr__ = lazy_exports(__name__, {"PostgresKnowledgeSource": "roomkit.knowledge.postgres"})

__all__ = [
    "KnowledgeResult",
    "KnowledgeSource",
    "MockKnowledgeSource",
    "PostgresKnowledgeSource",
]
