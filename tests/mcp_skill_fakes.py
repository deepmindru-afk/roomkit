"""An MCP server serving skills (SEP-2640), in memory, for the registry's tests.

It answers what a registry asks of a server (``MCPSkillServer``) and records
every request and read, so a test reads what crossed the wire rather than what
the registry believes it did.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from roomkit.skills.mcp import SKILLS_EXTENSION


def digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def skill_md(name: str, description: str, body: str = "Follow the steps.", **extra: str) -> str:
    """A SKILL.md with *name*, *description* and every *extra* field, in that order."""
    lines = [f"name: {name}", f"description: {description}"]
    lines += [f"{key}: {value}" for key, value in extra.items()]
    return "---\n" + "\n".join(lines) + "\n---\n\n" + body + "\n"


@dataclass
class ServedSkill:
    """One skill as a server holds it: its files by path inside the skill."""

    path: str
    files: dict[str, str | bytes]
    scheme: str = "skill"
    dynamic: bool = False

    @property
    def uri(self) -> str:
        return f"{self.scheme}://{self.path}/SKILL.md"

    def file_uri(self, relative: str) -> str:
        return f"{self.scheme}://{self.path}/{relative}"

    def raw(self, relative: str) -> bytes:
        content = self.files[relative]
        return content.encode("utf-8") if isinstance(content, str) else content

    def frontmatter(self) -> dict[str, Any]:
        """The SKILL.md frontmatter as the server renders it, read naively."""
        text = self.raw("SKILL.md").decode("utf-8")
        head = text.split("---", 2)[1]
        return dict(
            (key.strip(), value.strip())
            for key, value in (line.split(":", 1) for line in head.strip().splitlines())
        )

    def entry(self) -> dict[str, Any]:
        resources: Any = (
            "dynamic"
            if self.dynamic
            else [
                {
                    "uri": self.file_uri(rel),
                    "digest": digest(self.raw(rel)),
                    "size": len(self.raw(rel)),
                }
                for rel in self.files
            ]
        )
        return {"uri": self.uri, "frontmatter": self.frontmatter(), "resources": resources}


@dataclass
class FakeSkillServer:
    """A connected MCP server that declares the extension and serves *skills*.

    ``page_size`` pages the listing; ``listed`` (all by default) narrows what
    ``skills/list`` names while ``skills/get`` answers for every skill;
    ``served`` overrides the bytes a read returns, by URI, to tamper with them.
    """

    skills: list[ServedSkill] = field(default_factory=list)
    declares: bool = True
    page_size: int = 100
    listed: set[str] | None = None
    served: dict[str, str | bytes] = field(default_factory=dict)
    entry_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)

    @property
    def server_extensions(self) -> dict[str, dict[str, Any]]:
        return {SKILLS_EXTENSION: {}} if self.declares else {}

    def _entry(self, skill: ServedSkill) -> dict[str, Any]:
        return copy.deepcopy(self.entry_overrides.get(skill.uri) or skill.entry())

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        params = dict(params or {})
        self.requests.append((method, params))
        if method == "skills/list":
            listed = [s for s in self.skills if self.listed is None or s.uri in self.listed]
            start = int(params.get("cursor") or 0)
            end = start + self.page_size
            page: dict[str, Any] = {"skills": [self._entry(s) for s in listed[start:end]]}
            if end < len(listed):
                page["nextCursor"] = str(end)
            return page
        if method == "skills/get":
            for skill in self.skills:
                if skill.uri == params.get("uri"):
                    return {"skill": self._entry(skill)}
            raise LookupError(f"No skill is served at {params.get('uri')}")
        raise LookupError(f"unknown method {method}")

    async def read_resource(self, uri: str) -> dict[str, Any]:
        self.reads.append(uri)
        content: str | bytes | None = self.served.get(uri)
        if content is None:
            for skill in self.skills:
                for relative in skill.files:
                    if skill.file_uri(relative) == uri:
                        content = skill.files[relative]
        if content is None:
            raise LookupError(f"Unknown resource {uri}")
        if isinstance(content, bytes):
            return {"contents": [{"uri": uri, "blob": base64.b64encode(content).decode()}]}
        return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": content}]}
