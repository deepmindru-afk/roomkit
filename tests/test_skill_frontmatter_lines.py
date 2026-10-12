"""A local SKILL.md frontmatter read line by line keeps what the skill gates
and requires (RFC §24.1), whichever reason sends it there: no PyYAML
installed, or a frontmatter that is not YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from roomkit.skills.errors import SkillParseError
from roomkit.skills.parser import parse_frontmatter
from roomkit.skills.registry import SkillRegistry

RFC_EXAMPLE = """---
name: "web-search"
description: "Search the web for current information"
license: "MIT"
compatibility: "Needs network access"
allowed_tools:
  - "search_*"
  - "fetch_*"
requires:
  - search_web
---

## Instructions
"""


@pytest.fixture(params=["without PyYAML", "not YAML"])
def head(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """The first lines of a frontmatter the line reader reads: PyYAML is
    gone, or the description's colon makes the text invalid YAML."""
    if request.param == "without PyYAML":
        monkeypatch.setitem(sys.modules, "yaml", None)
        return "---\nname: refunds\ndescription: Process refund requests\n"
    return "---\nname: refunds\ndescription: Refunds: process them\n"


def _write_skill(root: Path, text: str) -> None:
    skill_dir = root / "refunds"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(text + "Body\n", encoding="utf-8")


class TestALineByLineReadingKeepsItsLists:
    def test_a_key_followed_by_items(self, head: str) -> None:
        text = head + (
            'allowed_tools:\n  - "issue_refund"\n  - void_*  # partial refunds\n'
            "requires:\n- lookup_order\n---\nBody\n"
        )
        data, _ = parse_frontmatter(text)
        assert data["allowed_tools"] == ["issue_refund", "void_*"]
        assert data["requires"] == ["lookup_order"]

    def test_a_list_on_the_key_line(self, head: str) -> None:
        text = head + "allowed_tools: [issue_refund, 'void_*']\nrequires: [lookup_order]\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["allowed_tools"] == ["issue_refund", "void_*"]
        assert data["requires"] == ["lookup_order"]

    def test_a_key_with_no_items_is_left_out(self, head: str) -> None:
        data, _ = parse_frontmatter(head + "allowed_tools:\nlicense: MIT\n---\n")
        assert "allowed_tools" not in data
        assert data["license"] == "MIT"

    def test_the_registry_keeps_the_tool_closed_until_activation(
        self, head: str, tmp_path: Path
    ) -> None:
        _write_skill(
            tmp_path, head + "allowed_tools:\n  - issue_refund\nrequires:\n  - lookup_order\n---\n"
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        assert registry.gated_tool_names() == {"issue_refund"}
        assert registry.gated_tool_names({"refunds"}) == set()
        metadata = registry.get_metadata("refunds")
        assert metadata is not None
        assert metadata.required_tool_names == ["lookup_order"]


@pytest.mark.parametrize("key", ["allowed_tools", "requires"])
def test_a_line_it_cannot_place_under_a_list_key_refuses_the_skill(
    head: str, tmp_path: Path, key: str
) -> None:
    # Read without that line, the skill would leave open what it gates.
    _write_skill(tmp_path, head + f"{key}:\n  - issue_refund\n  scope: refunds\n---\n")
    with pytest.raises(
        SkillParseError, match=f"Parsing failed for refunds: .*{key} cannot be read line by line"
    ):
        SkillRegistry().discover(tmp_path)


def test_the_rfc_example_reads_the_same_without_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    as_yaml, _ = parse_frontmatter(RFC_EXAMPLE)
    monkeypatch.setitem(sys.modules, "yaml", None)
    by_lines, _ = parse_frontmatter(RFC_EXAMPLE)
    assert by_lines == as_yaml
    assert by_lines["allowed_tools"] == ["search_*", "fetch_*"]
