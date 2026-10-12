"""Content an MCP server serves for a skill, checked against its entry (RFC §24.7).

A match proves the content consistent with an entry from the same server,
never trustworthy: the content stays text from outside (RFC §6.4).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from roomkit.skills.errors import SkillParseError, SkillVerificationError
from roomkit.skills.mcp.entry import SkillEntry, SkillResource
from roomkit.skills.parser import read_frontmatter_yaml, split_frontmatter


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
    """The frontmatter *text* as a JSON object, read within the bounds of
    :func:`~roomkit.skills.parser.read_frontmatter_yaml`: a server controls
    the content and its digest alike, so the digest does not stand in the way
    of an alias or a nesting that would exhaust the host."""
    try:
        return read_frontmatter_yaml(text)
    except SkillParseError as exc:
        raise SkillVerificationError(f"{uri}: {exc}") from exc


def _canonical(value: Any) -> str:
    """*value* as one JSON text, keys sorted: what two equal values share."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
