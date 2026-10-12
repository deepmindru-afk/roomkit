"""A local SKILL.md frontmatter read line by line keeps what the skill gates
and requires (RFC §24.1), whichever reason sends it there: no PyYAML
installed, or a frontmatter that is not YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from roomkit.skills.errors import SkillParseError
from roomkit.skills.models import parse_tool_names
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


@pytest.mark.parametrize(
    "lines",
    [
        "allowed_tools: [\n  issue_refund,\n  void_*\n]",
        "allowed_tools: >-\n  issue_refund,\n  void_*",
        "allowed_tools: |\n  issue_refund",
        "allowed_tools:\nissue_refund",
        "allowed_tools: issue_refund\n  - void_*",
        "allowed_tools: [issue_refund, [void_*]]",
        "allowed_tools: {issue_refund: true}",
        "allowed_tools:\n  - &money issue_refund",
        "requires:\n  - !tool lookup_order",
    ],
    ids=[
        "flow over lines",
        "folded",
        "literal",
        "bare line",
        "item after a value",
        "nested flow",
        "mapping",
        "anchor",
        "tag",
    ],
)
def test_a_gating_key_a_line_cannot_hold_whole_refuses_the_skill(head: str, lines: str) -> None:
    with pytest.raises(SkillParseError, match="cannot be read line by line"):
        parse_frontmatter(head + lines + "\n---\nBody\n")


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ('allowed_tools:\n  - "issue_refund"\n  - void_*', ["issue_refund", "void_*"]),
        ("allowed_tools:\n- issue_refund  # money", ["issue_refund"]),
        ("allowed_tools: [issue_refund, 'void_*']", ["issue_refund", "void_*"]),
        ("allowed_tools: [issue_refund, void_*]  # money", ["issue_refund", "void_*"]),
        ("allowed_tools: issue_refund, void_*", "issue_refund, void_*"),
        ('allowed_tools: "issue_refund, void_*"', "issue_refund, void_*"),
    ],
    ids=["block", "block with comment", "flow", "flow with comment", "text", "quoted text"],
)
def test_each_form_reads_as_yaml_reads_it(
    monkeypatch: pytest.MonkeyPatch, lines: str, expected: object
) -> None:
    text = "---\nname: refunds\ndescription: d\n" + lines + "\n---\n"
    as_yaml, _ = parse_frontmatter(text)
    monkeypatch.setitem(sys.modules, "yaml", None)
    by_lines, _ = parse_frontmatter(text)
    assert by_lines == as_yaml
    assert by_lines["allowed_tools"] == expected


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ("allowed_tools: *", "*"),
        ("allowed_tools:\n  - *_admin\n  - search_*", ["*_admin", "search_*"]),
        ("allowed_tools: [*_admin, search_*]", ["*_admin", "search_*"]),
    ],
    ids=["every tool", "block", "flow"],
)
def test_a_glob_starting_with_a_star_is_a_pattern(head: str, lines: str, expected: object) -> None:
    # YAML reads a leading * as an alias and fails; no line defines an
    # anchor, so the line reader takes it as the glob it is.
    data, _ = parse_frontmatter(head + lines + "\n---\n")
    assert data["allowed_tools"] == expected


def test_a_bracketed_description_stays_whole() -> None:
    # Only allowed_tools and requires read [a, b] as a list; this one is not
    # YAML either, so it goes to the line reader.
    text = "---\nname: refunds\ndescription: [Beta] Refunds, returns: see [docs]\n---\n"
    data, _ = parse_frontmatter(text)
    assert data["description"] == "[Beta] Refunds, returns: see [docs]"


@pytest.mark.parametrize(
    "lines",
    ["allowed_tools: {issue_refund: true}", "allowed_tools:\n  - [a, b]", "requires: [a, [b]]"],
    ids=["mapping", "nested list", "nested flow"],
)
def test_yaml_refuses_a_gating_key_that_is_not_a_list_of_texts(tmp_path: Path, lines: str) -> None:
    # As one text, it would gate nothing (RFC §24.1).
    _write_skill(tmp_path, "---\nname: refunds\ndescription: d\n" + lines + "\n---\n")
    with pytest.raises(
        SkillParseError, match="Parsing failed for refunds: .*neither a text nor a list of texts"
    ):
        SkillRegistry().discover(tmp_path)


@pytest.mark.parametrize(
    "lines",
    [
        "allowed_tools: [issue_refund]\nmetadata:\n  allowed_tools:\n",
        "allowed_tools: [issue_refund]\nmetadata:\n  allowed_tools: other\n",
        "requires:\n  - lookup_order\nmetadata:\n  requires: []\n",
        '"allowed_tools": [issue_refund]\n',
        "'allowed_tools':\n  - issue_refund\n",
        'allowed_tools: "issue_refund #1, void_*"\n',
        "allowed_tools:\n  - 'issue_refund # money'\n",
        'allowed_tools: ["issue,refund", void_*]\n',
        "allowed_tools:\n  - - issue_refund\n",
        "allowed_tools:\n  - ? issue_refund\n",
        "allowed_tools:\n  - issue_refund\n  - #void_*\n",
        "allowed_tools: issue_refund # money, void_*\n",
        " allowed_tools: [issue_refund]\n name: refunds\n",
        "allowed_tools: [issue_refund]\nmetadata: {a: 1,\nallowed_tools: x}\n",
        "allowed_tools: [issue_refund]\nnotes: [a,\nallowed_tools: x]\n",
        'allowed_tools: [issue_refund]\nnotes: "a\nallowed_tools:\nlicense: MIT"\n',
        "allowed_tools: [issue_refund]\nnotes: 'a\nallowed_tools: x'\n",
        "? allowed_tools\n: [issue_refund]\n",
        "!!str allowed_tools: [issue_refund]\n",
        "&key allowed_tools: [issue_refund]\n",
        "<<: {allowed_tools: [issue_refund]}\n",
        '"allowed\\x5ftools": [issue_refund]\n',
    ],
    ids=[
        "nested empty key",
        "nested value",
        "nested requires",
        "double-quoted key",
        "single-quoted key",
        "quoted value with a hash",
        "quoted item with a hash",
        "quoted item with a comma",
        "nested sequence",
        "complex key",
        "comment item",
        "comment after a value",
        "indented document",
        "flow mapping over lines",
        "flow sequence over lines",
        "double-quoted text over lines",
        "single-quoted text over lines",
        "explicit key",
        "tagged key",
        "anchored key",
        "merge key",
        "escaped key",
    ],
)
def test_the_line_reader_never_gates_less_than_yaml(
    monkeypatch: pytest.MonkeyPatch, lines: str
) -> None:
    # RFC §24.1: of a frontmatter that is also YAML, the line-by-line reading
    # gates and requires at least what YAML does, or refuses the skill.
    text = "---\nname: refunds\ndescription: d\n" + lines + "---\n"
    try:
        as_yaml, _ = parse_frontmatter(text)
    except SkillParseError:
        as_yaml = None
    monkeypatch.setitem(sys.modules, "yaml", None)
    try:
        by_lines, _ = parse_frontmatter(text)
    except SkillParseError:
        return
    assert as_yaml is not None
    for key in ("allowed_tools", "requires"):
        missing = set(parse_tool_names(as_yaml.get(key))) - set(
            parse_tool_names(by_lines.get(key))
        )
        assert not missing, key


def test_the_rfc_example_reads_the_same_without_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    as_yaml, _ = parse_frontmatter(RFC_EXAMPLE)
    monkeypatch.setitem(sys.modules, "yaml", None)
    by_lines, _ = parse_frontmatter(RFC_EXAMPLE)
    assert by_lines == as_yaml
    assert by_lines["allowed_tools"] == ["search_*", "fetch_*"]
