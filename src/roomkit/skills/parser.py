"""SKILL.md frontmatter parsing and validation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from roomkit.skills.errors import SkillParseError, SkillValidationError
from roomkit.skills.models import Skill, SkillMetadata

__all__ = [
    "SkillParseError",
    "SkillValidationError",
    "find_skill_md",
    "parse_frontmatter",
    "parse_skill",
    "parse_skill_metadata",
    "read_frontmatter_yaml",
    "split_frontmatter",
    "valid_skill_name",
    "validate_metadata",
]

logger = logging.getLogger("roomkit.skills")

FrontmatterValue = str | list[str]
"""A SKILL.md frontmatter value: a scalar, or the YAML list form ``allowed_tools``
uses in the RFC's own example."""

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX_LEN = 64
_DESC_MAX_LEN = 1024
_SKILL_FILENAMES = ("SKILL.md", "skill.md")

# Known frontmatter keys that map to SkillMetadata fields
_KNOWN_KEYS = {"name", "description", "license", "compatibility", "allowed_tools"}

FRONTMATTER_MAX = 64 * 1024
"""Characters a skill's frontmatter may take, in its entry's JSON as in its
``SKILL.md``'s YAML: far above any real one (a name, a description of 1024
characters, some metadata), and a bound on what reading and comparing it costs."""

_FLOW_DEPTH_MAX = 64
"""Brackets and braces open at once, read at most: pure-Python YAML takes time
quadratic in that depth (5000 levels, 3.5 s)."""

_DEPTH_MAX = 64
"""Collections a frontmatter may nest, in flow or block style: the C loader
builds a node by recursion, and 31000 nested ``- `` overflow its stack and
end the process."""


class _FrontmatterNotReadError(SkillParseError):
    """A frontmatter that was not read as a YAML mapping: PyYAML is missing,
    the text is not YAML or not a mapping, or it holds a value JSON cannot
    hold. Unlike a bound, it says nothing against the text, which a caller
    may read another way."""


def _read_skill_md(skill_dir: Path) -> tuple[Path, str]:
    """Resolve and decode one SKILL.md using the public error hierarchy."""
    try:
        resolved_dir = skill_dir.resolve()
        md_path = find_skill_md(resolved_dir)
        if md_path is None:
            raise SkillParseError(f"No SKILL.md found in {resolved_dir}")
        resolved_md = md_path.resolve()
        if resolved_dir not in resolved_md.parents:
            raise SkillParseError(f"SKILL.md escapes its skill directory: {md_path}")
        return resolved_dir, resolved_md.read_text(encoding="utf-8")
    except SkillParseError:
        raise
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise SkillParseError(f"Unable to read SKILL.md in {skill_dir}: {exc}") from exc


def find_skill_md(skill_dir: Path) -> Path | None:
    """Find SKILL.md in a directory (case-insensitive)."""
    for name in _SKILL_FILENAMES:
        candidate = skill_dir / name
        if candidate.is_file():
            return candidate
    # Fallback: scan for any case variation
    for child in skill_dir.iterdir():
        if child.is_file() and child.name.lower() == "skill.md":
            return child
    return None


def valid_skill_name(name: str) -> bool:
    """Whether *name* is a skill name as SKILL.md spells one (RFC §24.1):
    lowercase letters and digits in hyphen-separated runs, at most 64
    characters."""
    return bool(_NAME_PATTERN.match(name)) and len(name) <= _NAME_MAX_LEN


def split_frontmatter(content: str) -> tuple[str, str]:
    """Split SKILL.md into its frontmatter text and its body, both stripped.

    Raises:
        SkillParseError: If content has no valid frontmatter delimiters.
    """
    stripped = content.lstrip("\ufeff")  # strip BOM
    if not stripped.startswith("---"):
        raise SkillParseError("SKILL.md must start with '---' frontmatter delimiter")

    # Find closing delimiter
    end_idx = stripped.find("\n---", 3)
    if end_idx == -1:
        raise SkillParseError("SKILL.md missing closing '---' frontmatter delimiter")

    return stripped[3:end_idx].strip(), stripped[end_idx + 4 :].strip()  # skip past \n---


def read_frontmatter_yaml(text: str) -> dict[str, Any]:
    """The frontmatter *text* read as YAML, as JSON renders it (a date as its
    ISO string).

    An alias is refused before anything is built: YAML keeps an aliased node
    shared, so a few hundred bytes of aliases of aliases parse at once and
    then expand exponentially into JSON. Collections nested past
    :data:`_DEPTH_MAX` are refused from the same pass, before the loader
    recurses into them, and so is a frontmatter past :data:`FRONTMATTER_MAX`
    or nesting brackets past :data:`_FLOW_DEPTH_MAX`.

    Raises:
        SkillParseError: The text is past one of those bounds, or was not
            read as a YAML mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise _FrontmatterNotReadError(
            "PyYAML is needed to read a frontmatter as YAML: "
            "install it with pip install roomkit[mcp]"
        ) from exc
    if len(text) > FRONTMATTER_MAX:
        raise SkillParseError(f"the frontmatter is past {FRONTMATTER_MAX} characters")
    if _flow_depth(text) > _FLOW_DEPTH_MAX:
        raise SkillParseError("the frontmatter nests too deep")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        _refuse_unsafe_events(yaml.parse(text, loader), yaml)
        if loader is yaml.SafeLoader:
            data = yaml.safe_load(text)
        else:
            data = yaml.load(text, Loader=yaml.CSafeLoader)
        rendered = json.loads(json.dumps(data, default=str))
    except yaml.YAMLError as exc:
        raise _FrontmatterNotReadError("the frontmatter is not YAML") from exc
    except RecursionError as exc:
        raise SkillParseError("the frontmatter nests too deep") from exc
    except (OverflowError, TypeError, ValueError) as exc:
        # An integer past Python's digit limit, a value JSON cannot render.
        raise _FrontmatterNotReadError("the frontmatter holds a value JSON cannot hold") from exc
    if not isinstance(rendered, dict):
        raise _FrontmatterNotReadError("the frontmatter is not a mapping")
    return rendered


def _refuse_unsafe_events(events: Iterable[Any], yaml: ModuleType) -> None:
    """Refuse an alias, or collections nested past :data:`_DEPTH_MAX`, from
    the parse events alone: nothing is built yet, and the pass stops at the
    first one found."""
    depth = 0
    for event in events:
        if isinstance(event, yaml.AliasEvent):
            raise SkillParseError("the frontmatter holds a YAML alias")
        if isinstance(event, yaml.CollectionStartEvent):
            depth += 1
            if depth > _DEPTH_MAX:
                raise SkillParseError("the frontmatter nests too deep")
        elif isinstance(event, yaml.CollectionEndEvent):
            depth -= 1


def _flow_depth(text: str) -> int:
    """The most brackets and braces *text* holds open at once, quotes not
    read: an upper bound on its YAML flow nesting."""
    depth = deepest = 0
    for char in text:
        if char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}" and depth:
            depth -= 1
    return deepest


def parse_frontmatter(content: str) -> tuple[dict[str, FrontmatterValue], str]:
    """Split SKILL.md into frontmatter dict and body string.

    The frontmatter is read as YAML within the bounds of
    :func:`read_frontmatter_yaml`, and line by line (``key: value``) when it
    cannot be read as YAML or PyYAML is not installed.

    Raises:
        SkillParseError: If content has no valid frontmatter delimiters, or
            its YAML holds an alias, nests past 64 levels or is past
            :data:`FRONTMATTER_MAX` characters.
    """
    fm_text, body = split_frontmatter(content)
    data = _parse_yaml_or_fallback(fm_text)
    return data, body


def _parse_yaml_or_fallback(fm_text: str) -> dict[str, FrontmatterValue]:
    """Read the frontmatter as bounded YAML, or line by line when it is not
    read as YAML at all."""
    try:
        result = read_frontmatter_yaml(fm_text)
    except _FrontmatterNotReadError:
        logger.debug("YAML parse failed, falling back to regex parser")
        return _parse_lines(fm_text)
    # A list value keeps its items. Stringifying it produced
    # "['search_*', 'fetch_*']" — the literal repr — which then split
    # on commas into "['search_*'" and "'fetch_*']", gating nothing.
    # The RFC's own SKILL.md example uses the list form.
    return {
        str(k): [str(item) for item in v] if isinstance(v, list) else str(v)
        for k, v in result.items()
        if v is not None
    }


def _parse_lines(fm_text: str) -> dict[str, FrontmatterValue]:
    """Read the ``key: value`` lines of a frontmatter, quotes stripped."""
    data: dict[str, FrontmatterValue] = {}
    for line in fm_text.splitlines():
        match = re.match(r"^(\w[\w_-]*)\s*:\s*(.+)$", line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip().strip("\"'")
            data[key] = value
    return data


def _scalar(value: FrontmatterValue | None) -> str:
    """Read a frontmatter value that is meant to be a scalar.

    ``allowed_tools`` is the one field whose list form carries meaning. A list
    written where a scalar belongs is joined rather than refused — a skill with
    an odd ``license:`` line should still load.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def validate_metadata(data: dict[str, FrontmatterValue], skill_dir: Path) -> list[str]:
    """Validate parsed frontmatter data. Returns list of error strings."""
    errors: list[str] = []

    name = _scalar(data.get("name"))
    if not name:
        errors.append("Missing required field: name")
    elif not _NAME_PATTERN.match(name):
        errors.append(
            f"Invalid name {name!r}: must be kebab-case (lowercase alphanumeric + hyphens)"
        )
    elif len(name) > _NAME_MAX_LEN:
        errors.append(f"Name too long: {len(name)} chars (max {_NAME_MAX_LEN})")

    # Per spec: name must match directory name
    if name and name != skill_dir.name:
        errors.append(f"Name {name!r} does not match directory name {skill_dir.name!r}")

    description = _scalar(data.get("description"))
    if not description:
        errors.append("Missing required field: description")
    elif len(description) > _DESC_MAX_LEN:
        errors.append(f"Description too long: {len(description)} chars (max {_DESC_MAX_LEN})")

    return errors


def parse_skill_metadata(skill_dir: Path) -> SkillMetadata:
    """Parse SKILL.md frontmatter only (level 1 — lightweight).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    skill_dir, content = _read_skill_md(skill_dir)
    data, _ = parse_frontmatter(content)

    errors = validate_metadata(data, skill_dir)
    if errors:
        raise SkillValidationError(f"Validation failed for {skill_dir.name}: {'; '.join(errors)}")

    extra = {k: _scalar(v) for k, v in data.items() if k not in _KNOWN_KEYS}
    return SkillMetadata(
        name=_scalar(data["name"]),
        description=_scalar(data["description"]),
        license=_scalar(data.get("license")) or None,
        compatibility=_scalar(data.get("compatibility")) or None,
        allowed_tools=data.get("allowed_tools"),
        extra_metadata=extra,
    )


def parse_skill(skill_dir: Path) -> Skill:
    """Parse full SKILL.md including instructions body (level 2).

    Raises:
        SkillParseError: If SKILL.md cannot be found or parsed.
        SkillValidationError: If metadata fails validation.
    """
    skill_dir, content = _read_skill_md(skill_dir)
    data, body = parse_frontmatter(content)

    errors = validate_metadata(data, skill_dir)
    if errors:
        raise SkillValidationError(f"Validation failed for {skill_dir.name}: {'; '.join(errors)}")

    extra = {k: _scalar(v) for k, v in data.items() if k not in _KNOWN_KEYS}
    metadata = SkillMetadata(
        name=_scalar(data["name"]),
        description=_scalar(data["description"]),
        license=_scalar(data.get("license")) or None,
        compatibility=_scalar(data.get("compatibility")) or None,
        allowed_tools=data.get("allowed_tools"),
        extra_metadata=extra,
    )

    return Skill(metadata=metadata, instructions=body, path=skill_dir)
