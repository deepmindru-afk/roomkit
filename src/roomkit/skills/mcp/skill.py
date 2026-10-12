"""A skill an MCP server serves, once loaded and verified (RFC §24.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from roomkit.skills.errors import SkillPathError
from roomkit.skills.mcp.entry import SkillEntry
from roomkit.skills.models import Skill


@dataclass
class MCPSkill(Skill):
    """A skill whose ``SKILL.md`` an MCP server served and the registry verified.

    It holds the entry it was loaded from for as long as it is loaded: every
    file it reads is one that entry lists, checked against it. Nothing of it
    is on disk: its files are read through
    :meth:`~roomkit.skills.SkillRegistry.read_file`, never through ``path``,
    and it has no script a host can run.
    """

    origin: str = ""
    """The host's label for the server that serves it."""

    entry: SkillEntry | None = None
    """The entry it was loaded from."""

    files: dict[str, bytes] = field(default_factory=dict, repr=False)
    """The bytes of each file read so far, verified, by URI."""

    read_total: int = 0
    """Bytes read so far: what bounds a dynamic skill, which lists no sizes."""

    retirements: int = field(default=0, repr=False)
    """How many times a failed check had retired its name when it was loaded:
    a later one retires it in every registry holding it."""

    @property
    def held(self) -> SkillEntry:
        """The entry it was loaded from."""
        if self.entry is None:
            raise SkillPathError(f"Skill {self.name!r} holds no entry")
        return self.entry

    @property
    def has_scripts(self) -> bool:
        return False

    @property
    def has_references(self) -> bool:
        return bool(self.list_references())

    def _bundled_directory(self, name: str, *, kind: str) -> Path | None:
        return None

    def list_scripts(self) -> list[str]:
        """None: a served skill's files are never run (RFC §24.7)."""
        return []

    def list_references(self) -> list[str]:
        """Every file its entry lists but ``SKILL.md``, as a path inside the
        skill; none for a dynamic skill, which lists nothing."""
        entry = self.held
        if isinstance(entry.resources, str):
            return []
        return sorted(entry.relative(r.uri) for r in entry.resources if r.uri != entry.uri)

    def read_reference(self, filename: str) -> str:
        """Refused: a served skill's file is read from its server, verified,
        through :meth:`~roomkit.skills.SkillRegistry.read_file`."""
        raise SkillPathError(f"Skill {self.name!r} is read through SkillRegistry.read_file")

    def resolve_script(self, script_name: str) -> Path:
        """Refused: a served skill's files are never written to disk to be run."""
        raise SkillPathError(f"Skill {self.name!r} is served by an MCP server: it runs nothing")
