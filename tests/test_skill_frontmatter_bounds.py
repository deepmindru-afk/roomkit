"""A SKILL.md frontmatter is read as YAML within the same bounds whichever door
it comes through: a local skill's (``parse_frontmatter``) or one an MCP server
serves (``verify_frontmatter``, RFC §24.7)."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from roomkit.skills.errors import SkillError, SkillParseError, SkillVerificationError
from roomkit.skills.mcp import parse_entry, verify_frontmatter
from roomkit.skills.parser import parse_frontmatter
from roomkit.skills.registry import SkillRegistry
from tests.mcp_skill_fakes import ServedSkill, skill_md

REFUNDS = ServedSkill("refunds", {"SKILL.md": skill_md("refunds", "d")})


def _local(text: str) -> object:
    return parse_frontmatter(text)


def _served(text: str) -> object:
    return verify_frontmatter(parse_entry(REFUNDS.entry()), text)


DOORS: dict[str, tuple[Callable[[str], object], type[SkillError]]] = {
    "local": (_local, SkillParseError),
    "served": (_served, SkillVerificationError),
}


def _aliases_of_aliases() -> str:
    # A few hundred bytes YAML keeps shared, which reading would expand to
    # 9**7 strings (254 MB on the local door before it was bounded).
    lines = ["a: &a [x, x, x, x, x, x, x, x, x]"]
    for prev, cur in zip("abcdef", "bcdefg", strict=True):
        lines.append(f"{cur}: &{cur} [" + ", ".join([f"*{prev}"] * 9) + "]")
    return "---\nname: refunds\ndescription: d\n" + "\n".join(lines) + "\n---\nBody\n"


@pytest.mark.parametrize("door", DOORS)
class TestABoundRefusesTheFrontmatter:
    def test_an_alias_is_refused_before_it_expands(self, door: str) -> None:
        read, refusal = DOORS[door]
        with pytest.raises(refusal, match="YAML alias"):
            read(_aliases_of_aliases())

    def test_a_frontmatter_too_large_is_refused(self, door: str) -> None:
        read, refusal = DOORS[door]
        text = "---\nname: refunds\ndescription: " + "d" * 70_000 + "\n---\nBody\n"
        with pytest.raises(refusal, match="past 65536 characters"):
            read(text)

    def test_flow_nesting_too_deep_is_refused(self, door: str) -> None:
        read, refusal = DOORS[door]
        text = "---\nname: refunds\ndescription: d\ndeep: " + "[" * 5000 + "]" * 5000
        with pytest.raises(refusal, match="nests too deep"):
            read(text + "\n---\nBody\n")

    def test_block_nesting_too_deep_is_refused_without_crashing(self, door: str) -> None:
        # 31000 nested "- " fit under the size bound and overflow the C
        # loader's stack, ending the process (exit 139). Run apart, so a
        # regression fails this test, not the run.
        code = textwrap.dedent(
            f"""
            from roomkit.skills.errors import SkillError
            from tests.test_skill_frontmatter_bounds import DOORS

            read, _ = DOORS[{door!r}]
            text = "---\\nname: refunds\\ndescription: d\\nk:\\n" + "- " * 31000 + "v\\n---\\n"
            try:
                read(text)
            except SkillError as exc:
                print(type(exc).__name__, exc)
            """
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 0, done.stderr[-500:]
        assert DOORS[door][1].__name__ in done.stdout
        assert "nests too deep" in done.stdout


class TestTheLocalDoorStillReadsByLines:
    def test_a_frontmatter_that_is_not_yaml(self) -> None:
        # A colon inside a plain value is not YAML; the line reader takes it.
        text = "---\nname: refunds\ndescription: Use it: for refunds\n---\nBody\n"
        data, body = parse_frontmatter(text)
        assert data == {"name": "refunds", "description": "Use it: for refunds"}
        assert body == "Body"

    def test_without_pyyaml_no_bound_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "yaml", None)
        text = "---\nname: refunds\ndescription: " + "d" * 70_000 + "\n---\nBody\n"
        data, _ = parse_frontmatter(text)
        assert data["description"] == "d" * 70_000


def test_the_served_door_refuses_a_frontmatter_without_pyyaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(SkillVerificationError, match=r"PyYAML is needed.*roomkit\[mcp\]"):
        _served(skill_md("refunds", "d"))


def _write_skill(root: Path, name: str, frontmatter: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(frontmatter + "Body\n", encoding="utf-8")


class TestDiscoveryOfALocalSkillPastABound:
    def test_strict_discovery_fails_on_it(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "refunds", _aliases_of_aliases().replace("Body\n", ""))
        registry = SkillRegistry()
        with pytest.raises(SkillParseError, match="YAML alias"):
            registry.discover(tmp_path)
        assert registry.skill_names == []

    def test_lenient_discovery_skips_it_and_says_why(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_skill(tmp_path, "refunds", _aliases_of_aliases().replace("Body\n", ""))
        _write_skill(tmp_path, "orders", "---\nname: orders\ndescription: d\n---\n")
        registry = SkillRegistry()
        with caplog.at_level(logging.WARNING, logger="roomkit.skills"):
            assert registry.discover(tmp_path, strict=False) == 1
        assert registry.skill_names == ["orders"]
        assert "refunds" in caplog.text
        assert "YAML alias" in caplog.text
