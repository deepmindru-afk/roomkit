"""Content an MCP server serves for a skill, checked against its entry (RFC §24.7).

A match proves the content consistent with an entry from the same server,
never trustworthy: the content stays text from outside (RFC §6.4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from roomkit.skills.errors import SkillParseError, SkillVerificationError
from roomkit.skills.mcp.entry import FRONTMATTER_MAX, SkillEntry, SkillResource
from roomkit.skills.parser import split_frontmatter

_FLOW_DEPTH_MAX = 64
"""Brackets and braces open at once, read at most: pure-Python YAML takes time
quadratic in that depth (5000 levels, 3.5 s)."""


def verify_skill_resource(resource: SkillResource, content: bytes) -> None:
    """Check *content* against the listed file: its length, then its digest.

    Named as the MCP Python SDK names it.

    Raises:
        SkillVerificationError: The content is not what the entry describes.
    """
    if len(content) != resource.size:
        raise SkillVerificationError(
            f"{resource.uri}: {len(content)} bytes where the entry lists {resource.size}"
        )
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if digest != resource.digest:
        raise SkillVerificationError(f"{resource.uri}: its digest is not the entry's")


def verify_frontmatter(entry: SkillEntry, skill_md: str) -> str:
    """Check the frontmatter of *skill_md* against the entry's, field by field,
    and return the instructions that follow it.

    The YAML is read as JSON renders it (a date as its ISO string), and two
    values match only when their JSON is the same.

    Raises:
        SkillVerificationError: The frontmatter cannot be read, or a field
            differs, is missing or is extra.
    """
    try:
        frontmatter_text, body = split_frontmatter(skill_md)
    except SkillParseError as exc:
        raise SkillVerificationError(f"{entry.uri}: {exc}") from exc
    written = _yaml_as_json(entry.uri, frontmatter_text)
    try:
        differing = sorted(
            key
            for key in set(written) | set(entry.frontmatter)
            if key not in written
            or key not in entry.frontmatter
            or _canonical(written[key]) != _canonical(entry.frontmatter[key])
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise SkillVerificationError(f"{entry.uri}: its frontmatter cannot be compared") from exc
    if differing:
        raise SkillVerificationError(
            f"{entry.uri}: its frontmatter differs from the entry's on {', '.join(differing)}"
        )
    return body


def _yaml_as_json(uri: str, text: str) -> dict[str, Any]:
    """The frontmatter *text* as a JSON object.

    An alias is refused before anything is built: YAML keeps an aliased node
    shared, so a few hundred bytes of aliases of aliases parse at once and
    then expand exponentially into JSON. A server controls the content and
    its digest alike, so the digest does not stand in the way. Nesting too
    deep to read is refused as well, and so is a frontmatter past
    :data:`~roomkit.skills.mcp.entry.FRONTMATTER_MAX` or nesting brackets past
    :data:`_FLOW_DEPTH_MAX`.
    """
    if len(text) > FRONTMATTER_MAX:
        raise SkillVerificationError(
            f"{uri}: its frontmatter is past {FRONTMATTER_MAX} characters"
        )
    if _flow_depth(text) > _FLOW_DEPTH_MAX:
        raise SkillVerificationError(f"{uri}: its frontmatter nests too deep")
    try:
        import yaml
    except ImportError as exc:
        raise SkillVerificationError(
            "PyYAML is needed to verify a served skill's frontmatter: "
            "install it with pip install roomkit[mcp]"
        ) from exc
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text, loader)):
            raise SkillVerificationError(f"{uri}: its frontmatter holds a YAML alias")
        if loader is yaml.SafeLoader:
            data = yaml.safe_load(text)
        else:
            data = yaml.load(text, Loader=yaml.CSafeLoader)
        if not isinstance(data, Mapping):
            raise SkillVerificationError(f"{uri}: its frontmatter is not a mapping")
        rendered = json.loads(json.dumps(data, default=str))
    except yaml.YAMLError as exc:
        raise SkillVerificationError(f"{uri}: its frontmatter is not YAML") from exc
    except RecursionError as exc:
        raise SkillVerificationError(f"{uri}: its frontmatter nests too deep") from exc
    except (OverflowError, TypeError, ValueError) as exc:
        # An integer past Python's digit limit, a value JSON cannot render.
        raise SkillVerificationError(
            f"{uri}: its frontmatter holds a value JSON cannot hold"
        ) from exc
    if not isinstance(rendered, dict):
        raise SkillVerificationError(f"{uri}: its frontmatter is not a mapping")
    return rendered


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


def _canonical(value: Any) -> str:
    """*value* as one JSON text, keys sorted: what two equal values share."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
