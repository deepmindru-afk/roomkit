"""One skill an MCP server lists, read and checked on its own (RFC §24.7).

The entry is the skill's manifest under the MCP Skills extension (SEP-2640):
the URI of its ``SKILL.md``, its frontmatter verbatim, and every file it holds
with the digest and size of that file's bytes, or ``"dynamic"`` for a skill
whose content cannot carry stable digests. Everything the registry later
verifies is checked against it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from roomkit.skills.errors import SkillPathError, SkillValidationError
from roomkit.skills.parser import FRONTMATTER_MAX, valid_skill_name

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
"""The identifier a server declares the extension under."""

MAX_RESOURCES_PER_SKILL = 512
"""Files one skill may list, its ``SKILL.md`` included."""

MAX_TOTAL_SIZE = 16 * 1024 * 1024
"""Bytes one skill's files may hold in all."""

DYNAMIC = "dynamic"
"""The ``resources`` of a skill whose content has no stable digests."""

_SKILL_MD = "SKILL.md"
_DESCRIPTION_MAX = 1024
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://(.+)$")


@dataclass(frozen=True)
class SkillResource:
    """One file of a skill: its URI, and the SHA-256 digest and length of its bytes.

    Named as the MCP Python SDK names it, so this module gives way to the SDK's
    once RoomKit requires a release that carries it.
    """

    uri: str
    digest: str
    size: int


@dataclass(frozen=True)
class SkillEntry:
    """The entry for one skill, as ``skills/list`` and ``skills/get`` return it.

    Build it with :func:`parse_entry`, which refuses an entry the extension
    does not allow; a constructed entry is therefore consistent: its URI ends
    in ``/<name>/SKILL.md``, and a listed set holds its own ``SKILL.md`` and
    only files within the skill's directory, each once.
    """

    uri: str
    frontmatter: Mapping[str, Any]
    resources: tuple[SkillResource, ...] | Literal["dynamic"]
    _files: dict[str, SkillResource] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        files = {} if self.resources == DYNAMIC else {r.uri: r for r in self.resources}
        object.__setattr__(self, "_files", files)

    @property
    def name(self) -> str:
        """The frontmatter ``name``: the last segment of the skill's path."""
        return str(self.frontmatter["name"])

    @property
    def description(self) -> str:
        """The frontmatter ``description``, as the server wrote it."""
        return str(self.frontmatter["description"])

    @property
    def dynamic(self) -> bool:
        """Whether the skill lists no digests, so nothing of it can be verified."""
        return self.resources == DYNAMIC

    @property
    def root(self) -> str:
        """The URI of the skill's directory: its ``SKILL.md`` URI without the file."""
        return self.uri[: -len(_SKILL_MD) - 1]

    @property
    def path(self) -> str:
        """The skill's path on its server, as a registry names it after the label:
        the directory's URI, without the scheme when it is ``skill://``."""
        root = self.root
        return root[len("skill://") :] if root.startswith("skill://") else root

    @property
    def total_size(self) -> int:
        """Bytes the listed files hold in all; 0 for a dynamic skill."""
        return 0 if self.resources == DYNAMIC else sum(r.size for r in self.resources)

    @property
    def fingerprint(self) -> str | None:
        """The SHA-256 of the listed files' URIs and digests, one ``<uri> <digest>``
        line each, sorted: what an approval of this skill binds to (RFC §24.7).
        ``None`` for a dynamic skill, which no approval can bind."""
        if self.resources == DYNAMIC:
            return None
        lines = sorted(f"{r.uri} {r.digest}" for r in self.resources)
        return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    def resource(self, uri: str) -> SkillResource | None:
        """The listed file at *uri*, or ``None`` when the entry does not list it."""
        return self._files.get(uri)

    def resolve(self, relative: str) -> str:
        """The URI of *relative*, a path inside the skill's directory
        (``templates/invoice.md``), resolved against that directory.

        Raises:
            SkillPathError: The path is empty, absolute, or leaves the directory.
        """
        if not relative or relative.startswith("/") or "\\" in relative:
            raise SkillPathError(f"Not a path inside the skill: {relative!r}")
        if not _segments_ok(relative.split("/")):
            raise SkillPathError(f"Not a path inside the skill: {relative!r}")
        return f"{self.root}/{relative}"

    def relative(self, uri: str) -> str:
        """*uri*, a file of this skill, as a path inside its directory."""
        return uri[len(self.root) + 1 :]

    def limit_exceeded(self) -> str | None:
        """Why the skill is past the extension's limits, or ``None``."""
        if self.resources == DYNAMIC:
            return None
        if len(self.resources) > MAX_RESOURCES_PER_SKILL:
            return f"it lists {len(self.resources)} files, past {MAX_RESOURCES_PER_SKILL}"
        if self.total_size > MAX_TOTAL_SIZE:
            return f"its files hold {self.total_size} bytes, past {MAX_TOTAL_SIZE}"
        return None


def parse_entry(raw: object) -> SkillEntry:
    """The entry *raw* describes, as a server sent it.

    Raises:
        SkillValidationError: The entry is one the extension does not allow,
            with the reason; a host must not load it.
    """
    if not isinstance(raw, Mapping):
        raise SkillValidationError("the entry is not an object")
    uri = raw.get("uri")
    if not isinstance(uri, str):
        raise SkillValidationError("the entry has no URI")
    root = _skill_root(uri)
    frontmatter = raw.get("frontmatter")
    if not isinstance(frontmatter, Mapping):
        raise SkillValidationError(f"{uri}: the entry has no frontmatter")
    _check_frontmatter(uri, root, frontmatter)
    return SkillEntry(
        uri=uri,
        frontmatter=_json_copy(uri, frontmatter),
        resources=_resources(uri, root, raw.get("resources")),
    )


def _json_copy(uri: str, frontmatter: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *frontmatter* through its JSON, within :data:`FRONTMATTER_MAX`
    characters: what is kept is plain JSON, sized before anything else reads it."""
    try:
        text = json.dumps(frontmatter)
    except (RecursionError, TypeError, ValueError) as exc:
        raise SkillValidationError(f"{uri}: the frontmatter is not plain JSON") from exc
    if len(text) > FRONTMATTER_MAX:
        raise SkillValidationError(f"{uri}: the frontmatter is past {FRONTMATTER_MAX} characters")
    copied: dict[str, Any] = json.loads(text)
    return copied


def _segments_ok(segments: list[str]) -> bool:
    """Whether path *segments* stay where they are: none empty, ``.`` or ``..``."""
    return all(segment not in ("", ".", "..") for segment in segments)


def _skill_root(uri: str) -> str:
    """The directory URI of the skill whose ``SKILL.md`` is at *uri*."""
    match = _SCHEME.match(uri)
    suffix = "/" + _SKILL_MD
    if match is None or not uri.endswith(suffix):
        raise SkillValidationError(f"{uri}: not the URI of a SKILL.md")
    path = match.group(2)[: -len(suffix)]
    if not path or not _segments_ok(path.split("/")):
        raise SkillValidationError(f"{uri}: the skill's path is not a plain path")
    return uri[: -len(suffix)]


def _check_frontmatter(uri: str, root: str, frontmatter: Mapping[str, Any]) -> None:
    """Refuse a frontmatter whose name or description the extension does not allow."""
    name = frontmatter.get("name")
    if not isinstance(name, str) or not valid_skill_name(name):
        raise SkillValidationError(f"{uri}: the frontmatter name is not a skill name")
    if root.rsplit("/", 1)[-1] != name:
        raise SkillValidationError(f"{uri}: the frontmatter name {name!r} is not the URI's")
    description = frontmatter.get("description")
    if not isinstance(description, str) or not 1 <= len(description) <= _DESCRIPTION_MAX:
        raise SkillValidationError(f"{uri}: the frontmatter description is missing or too long")


def _resources(uri: str, root: str, raw: object) -> tuple[SkillResource, ...] | Literal["dynamic"]:
    """The entry's files, each within the skill's directory and listed once."""
    if raw == DYNAMIC:
        return DYNAMIC
    if not isinstance(raw, list):
        raise SkillValidationError(f"{uri}: resources is neither a list nor {DYNAMIC!r}")
    resources: list[SkillResource] = []
    seen: set[str] = set()
    for item in raw:
        resource = _resource(uri, item)
        relative = resource.uri[len(root) + 1 :]
        if not resource.uri.startswith(root + "/") or not _segments_ok(relative.split("/")):
            raise SkillValidationError(f"{uri}: {resource.uri} is outside the skill")
        if resource.uri in seen:
            raise SkillValidationError(f"{uri}: {resource.uri} is listed twice")
        seen.add(resource.uri)
        resources.append(resource)
    if uri not in seen:
        raise SkillValidationError(f"{uri}: resources does not list the skill's own SKILL.md")
    return tuple(resources)


def _resource(uri: str, item: object) -> SkillResource:
    """One listed file, its digest and size well formed."""
    if not isinstance(item, Mapping):
        raise SkillValidationError(f"{uri}: a listed file is not an object")
    file_uri, digest, size = item.get("uri"), item.get("digest"), item.get("size")
    if not isinstance(file_uri, str):
        raise SkillValidationError(f"{uri}: a listed file has no URI")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise SkillValidationError(f"{uri}: {file_uri} has no sha256 digest")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SkillValidationError(f"{uri}: {file_uri} has no size")
    return SkillResource(uri=file_uri, digest=digest, size=size)
