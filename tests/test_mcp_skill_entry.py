"""A skill an MCP server lists: its entry read on its own, its content checked
against it (RFC §24.7, SEP-2640)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from roomkit.skills.errors import SkillPathError, SkillValidationError, SkillVerificationError
from roomkit.skills.mcp import (
    MAX_RESOURCES_PER_SKILL,
    MAX_TOTAL_SIZE,
    SkillResource,
    parse_entry,
    verify_frontmatter,
    verify_skill_resource,
)
from tests.mcp_skill_fakes import ServedSkill, digest, skill_md

REFUNDS = ServedSkill(
    "acme/billing/refunds",
    {
        "SKILL.md": skill_md("refunds", "Process refund requests"),
        "examples/email.md": "# Email\n",
    },
)


def _entry(**changes: Any) -> dict[str, Any]:
    raw = REFUNDS.entry()
    raw.update(changes)
    return raw


def _file(uri: str, content: bytes = b"x") -> dict[str, Any]:
    return {"uri": uri, "digest": digest(content), "size": len(content)}


class TestAValidEntry:
    def test_reads_its_name_path_and_files(self) -> None:
        entry = parse_entry(REFUNDS.entry())
        assert entry.name == "refunds"
        assert entry.description == "Process refund requests"
        assert entry.path == "acme/billing/refunds"
        assert entry.root == "skill://acme/billing/refunds"
        assert not entry.dynamic
        assert entry.resource("skill://acme/billing/refunds/examples/email.md") is not None
        assert entry.resource("skill://acme/billing/refunds/other.md") is None

    def test_a_path_under_another_scheme_keeps_it(self) -> None:
        skill = ServedSkill("owner/repo/skills/refunds", REFUNDS.files, scheme="github")
        assert parse_entry(skill.entry()).path == "github://owner/repo/skills/refunds"

    def test_a_dynamic_entry_lists_nothing_and_has_no_fingerprint(self) -> None:
        entry = parse_entry(_entry(resources="dynamic"))
        assert entry.dynamic
        assert entry.fingerprint is None
        assert entry.resource(REFUNDS.uri) is None

    def test_extra_frontmatter_passes_through_unchanged(self) -> None:
        raw = _entry()
        raw["frontmatter"] = {**raw["frontmatter"], "license": "MIT", "metadata": {"a": [1]}}
        entry = parse_entry(raw)
        raw["frontmatter"]["metadata"]["a"].append(2)
        assert entry.frontmatter["metadata"] == {"a": [1]}
        assert entry.frontmatter["license"] == "MIT"


class TestFingerprint:
    def test_binds_every_uri_and_digest_whatever_the_order(self) -> None:
        raw = REFUNDS.entry()
        shuffled = {**raw, "resources": list(reversed(raw["resources"]))}
        assert parse_entry(raw).fingerprint == parse_entry(shuffled).fingerprint
        lines = sorted(f"{r['uri']} {r['digest']}" for r in raw["resources"])
        expected = "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()
        assert parse_entry(raw).fingerprint == expected

    def test_changes_when_a_file_changes(self) -> None:
        changed = ServedSkill(REFUNDS.path, {**REFUNDS.files, "examples/email.md": "# Mail\n"})
        assert parse_entry(changed.entry()).fingerprint != parse_entry(REFUNDS.entry()).fingerprint


class TestARefusedEntry:
    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("not an object", "not an object"),
            ({"frontmatter": {}}, "no URI"),
            (_entry(uri="skill://acme/billing/refunds/README.md"), "not the URI of a SKILL.md"),
            (_entry(uri="refunds/SKILL.md"), "not the URI of a SKILL.md"),
            (_entry(uri="skill://acme/../refunds/SKILL.md"), "not a plain path"),
            (_entry(frontmatter=None), "no frontmatter"),
        ],
    )
    def test_without_a_skill_uri_or_frontmatter(self, raw: object, reason: str) -> None:
        with pytest.raises(SkillValidationError, match=reason):
            parse_entry(raw)

    def test_whose_name_is_not_the_uris(self) -> None:
        raw = _entry(frontmatter={"name": "chargebacks", "description": "d"})
        with pytest.raises(SkillValidationError, match="is not the URI's"):
            parse_entry(raw)

    @pytest.mark.parametrize("name", ["Refunds", "re_funds", "", None, "a" * 65])
    def test_whose_name_is_not_a_skill_name(self, name: object) -> None:
        with pytest.raises(SkillValidationError, match="not a skill name"):
            parse_entry(_entry(frontmatter={"name": name, "description": "d"}))

    @pytest.mark.parametrize("description", ["", None, "d" * 1025])
    def test_without_a_description(self, description: object) -> None:
        with pytest.raises(SkillValidationError, match="description"):
            parse_entry(_entry(frontmatter={"name": "refunds", "description": description}))

    @pytest.mark.parametrize("resources", [None, "static", {"uri": "x"}, 3])
    def test_whose_resources_are_neither_a_list_nor_dynamic(self, resources: object) -> None:
        raw = _entry()
        if resources is None:
            del raw["resources"]
        else:
            raw["resources"] = resources
        with pytest.raises(SkillValidationError, match="neither a list"):
            parse_entry(raw)

    def test_that_omits_its_own_skill_md(self) -> None:
        raw = _entry(resources=[_file("skill://acme/billing/refunds/examples/email.md")])
        with pytest.raises(SkillValidationError, match="own SKILL.md"):
            parse_entry(raw)

    @pytest.mark.parametrize(
        "outside",
        [
            "skill://acme/billing/other/SKILL.md",
            "skill://acme/billing/refunds-old/x.md",
            "skill://acme/billing/refunds/../other/x.md",
            "skill://acme/billing/refunds//x.md",
            "github://acme/billing/refunds/x.md",
        ],
    )
    def test_that_lists_a_file_outside_the_skill(self, outside: str) -> None:
        raw = _entry(resources=[*REFUNDS.entry()["resources"], _file(outside)])
        with pytest.raises(SkillValidationError, match="outside the skill"):
            parse_entry(raw)

    def test_that_lists_a_file_twice(self) -> None:
        resources = REFUNDS.entry()["resources"]
        with pytest.raises(SkillValidationError, match="listed twice"):
            parse_entry(_entry(resources=[*resources, resources[0]]))

    @pytest.mark.parametrize(
        "bad",
        [
            {"digest": "sha256:ABC"},
            {"digest": "md5:" + "0" * 64},
            {"size": -1},
            {"size": True},
            {"size": "3"},
            {"uri": None},
        ],
    )
    def test_whose_file_is_malformed(self, bad: dict[str, Any]) -> None:
        resources = REFUNDS.entry()["resources"]
        with pytest.raises(SkillValidationError):
            parse_entry(_entry(resources=[{**resources[0], **bad}, *resources[1:]]))


class TestLimits:
    def test_an_entry_past_the_file_count_says_so(self) -> None:
        files = [
            _file(f"skill://acme/billing/refunds/f{i}.md") for i in range(MAX_RESOURCES_PER_SKILL)
        ]
        entry = parse_entry(_entry(resources=[*REFUNDS.entry()["resources"][:1], *files]))
        assert "513 files" in (entry.limit_exceeded() or "")

    def test_an_entry_at_the_limits_is_within_them(self) -> None:
        resources = REFUNDS.entry()["resources"][:1]
        files = [
            _file(f"skill://acme/billing/refunds/f{i}.md")
            for i in range(MAX_RESOURCES_PER_SKILL - 1)
        ]
        assert parse_entry(_entry(resources=[*resources, *files])).limit_exceeded() is None

    def test_an_entry_past_the_total_size_says_so(self) -> None:
        big = {
            "uri": "skill://acme/billing/refunds/big.bin",
            "digest": digest(b""),
            "size": MAX_TOTAL_SIZE,
        }
        entry = parse_entry(_entry(resources=[*REFUNDS.entry()["resources"], big]))
        assert "bytes, past" in (entry.limit_exceeded() or "")


class TestResolve:
    def test_a_relative_path_resolves_against_the_skill(self) -> None:
        entry = parse_entry(REFUNDS.entry())
        assert (
            entry.resolve("examples/email.md") == "skill://acme/billing/refunds/examples/email.md"
        )
        assert entry.relative(entry.resolve("examples/email.md")) == "examples/email.md"

    @pytest.mark.parametrize(
        "path", ["", "/etc/passwd", "../other/SKILL.md", "a/../../x", "a//b", "./x", "a\\b"]
    )
    def test_a_path_leaving_the_skill_is_refused(self, path: str) -> None:
        with pytest.raises(SkillPathError):
            parse_entry(REFUNDS.entry()).resolve(path)


class TestVerifyResource:
    RESOURCE = SkillResource(uri="skill://x/SKILL.md", digest=digest(b"hello"), size=5)

    def test_matching_content_passes(self) -> None:
        verify_skill_resource(self.RESOURCE, b"hello")

    def test_a_length_that_differs_fails_before_the_digest(self) -> None:
        with pytest.raises(SkillVerificationError, match="6 bytes where the entry lists 5"):
            verify_skill_resource(self.RESOURCE, b"hello!")

    def test_a_digest_that_differs_fails(self) -> None:
        with pytest.raises(SkillVerificationError, match="digest"):
            verify_skill_resource(self.RESOURCE, b"hullo")


class TestVerifyFrontmatter:
    def test_matching_frontmatter_returns_the_instructions(self) -> None:
        entry = parse_entry(REFUNDS.entry())
        assert verify_frontmatter(entry, REFUNDS.files["SKILL.md"]) == "Follow the steps."

    def test_nested_values_compare_as_json(self) -> None:
        text = "---\nname: refunds\ndescription: d\nmetadata:\n  tags: [a, b]\n---\nBody\n"
        raw = _entry(
            frontmatter={"name": "refunds", "description": "d", "metadata": {"tags": ["a", "b"]}}
        )
        assert verify_frontmatter(parse_entry(raw), text) == "Body"

    @pytest.mark.parametrize(
        ("text", "field"),
        [
            ("---\nname: refunds\ndescription: other\n---\nBody\n", "description"),
            (
                "---\nname: refunds\ndescription: Process refund requests\n"
                "allowed_tools: [bash]\n---\n",
                "allowed_tools",
            ),
            ("---\nname: refunds\n---\nBody\n", "description"),
            (
                "---\nname: refunds\ndescription: Process refund requests\nversion: 1\n---\n",
                "version",
            ),
        ],
    )
    def test_a_field_that_differs_is_named(self, text: str, field: str) -> None:
        with pytest.raises(SkillVerificationError, match=f"differs from the entry's on {field}"):
            verify_frontmatter(parse_entry(REFUNDS.entry()), text)

    def test_a_number_and_its_string_differ(self) -> None:
        raw = _entry(frontmatter={"name": "refunds", "description": "d", "version": "1"})
        with pytest.raises(SkillVerificationError, match="version"):
            verify_frontmatter(
                parse_entry(raw), "---\nname: refunds\ndescription: d\nversion: 1\n---\n"
            )

    @pytest.mark.parametrize(
        "text",
        ["no frontmatter", "---\nname: [unclosed\n---\n", "---\n- a list\n---\n"],
    )
    def test_unreadable_frontmatter_fails(self, text: str) -> None:
        with pytest.raises(SkillVerificationError):
            verify_frontmatter(parse_entry(REFUNDS.entry()), text)

    def test_an_alias_is_refused_before_it_expands(self) -> None:
        # Aliases of aliases: a few hundred bytes YAML keeps shared, which JSON
        # would expand to 9**7 strings. The server controls the digest too.
        lines = ["a: &a [x, x, x, x, x, x, x, x, x]"]
        for prev, cur in zip("abcdef", "bcdefg", strict=True):
            lines.append(f"{cur}: &{cur} [" + ", ".join([f"*{prev}"] * 9) + "]")
        text = "---\nname: refunds\ndescription: d\n" + "\n".join(lines) + "\n---\n"
        with pytest.raises(SkillVerificationError, match="YAML alias"):
            verify_frontmatter(parse_entry(REFUNDS.entry()), text)

    def test_a_frontmatter_too_large_is_refused(self) -> None:
        text = "---\nname: refunds\ndescription: " + "d" * 70_000 + "\n---\n"
        with pytest.raises(SkillVerificationError, match="past 65536 characters"):
            verify_frontmatter(parse_entry(REFUNDS.entry()), text)

    def test_nesting_too_deep_is_refused(self) -> None:
        text = "---\nname: refunds\ndescription: d\ndeep: " + "[" * 5000 + "]" * 5000 + "\n---\n"
        with pytest.raises(SkillVerificationError, match="nests too deep"):
            verify_frontmatter(parse_entry(REFUNDS.entry()), text)

    def test_block_nesting_too_deep_is_refused_without_crashing(self) -> None:
        # Found by a pre-release review: 31000 nested "- " fit under the size
        # bound and overflowed the C loader's stack, ending the process (exit
        # 139). Run apart, so a regression fails this test, not the run.
        code = textwrap.dedent(
            """
            from roomkit.skills.errors import SkillVerificationError
            from roomkit.skills.mcp import parse_entry, verify_frontmatter
            from tests.mcp_skill_fakes import ServedSkill, skill_md

            skill = ServedSkill("refunds", {"SKILL.md": skill_md("refunds", "d")})
            text = "---\\nname: refunds\\ndescription: d\\nk:\\n" + "- " * 31000 + "v\\n---\\n"
            try:
                verify_frontmatter(parse_entry(skill.entry()), text)
            except SkillVerificationError as exc:
                print(exc)
            """
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 0, done.stderr[-500:]
        assert "nests too deep" in done.stdout


def test_an_entry_whose_frontmatter_nests_too_deep_is_refused() -> None:
    deep: dict[str, object] = {}
    for _ in range(100_000):
        deep = {"d": deep}
    raw = _entry(frontmatter={"name": "refunds", "description": "d", "deep": deep})
    with pytest.raises(SkillValidationError, match="not plain JSON|past 65536"):
        parse_entry(raw)


def test_an_entry_whose_frontmatter_is_too_large_is_refused() -> None:
    raw = _entry(frontmatter={"name": "refunds", "description": "d", "notes": "n" * 70_000})
    with pytest.raises(SkillValidationError, match="past 65536 characters"):
        parse_entry(raw)


def test_an_integer_past_pythons_digit_limit_fails_verification() -> None:
    text = "---\nname: refunds\ndescription: d\nbig: 1" + "0" * 5000 + "\n---\n"
    with pytest.raises(SkillVerificationError, match="JSON cannot hold"):
        verify_frontmatter(parse_entry(REFUNDS.entry()), text)
