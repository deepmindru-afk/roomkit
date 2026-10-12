"""A registry holding the skills MCP servers serve (RFC §24.7, SEP-2640)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from roomkit.skills import SkillRegistry
from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillError,
    SkillPathError,
    SkillValidationError,
    SkillVerificationError,
)
from roomkit.skills.mcp.catalog import SkillChangedError
from roomkit.skills.mcp.entry import MAX_RESOURCES_PER_SKILL
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.skills.models import Skill, SkillMetadata
from tests.mcp_skill_fakes import FakeSkillServer, ServedSkill, digest, skill_md

PDF = ServedSkill(
    "pdf-processing",
    {
        "SKILL.md": skill_md(
            "pdf-processing", "Extract and fill PDF documents", "Use templates/."
        ),
        "templates/invoice.md": "# Invoice\n",
        "scripts/extract.py": "print('extract')\n",
    },
)
REFUNDS = ServedSkill(
    "acme/billing/refunds",
    {"SKILL.md": skill_md("refunds", "Process refund requests", "Be kind.")},
)


async def _registry(
    *skills: ServedSkill, label: str = "hf", **kwargs: Any
) -> tuple[SkillRegistry, FakeSkillServer]:
    server = FakeSkillServer(list(skills or (PDF, REFUNDS)))
    registry = SkillRegistry()
    await registry.add_mcp_server(server, label, **kwargs)
    return registry, server


class TestRegistration:
    async def test_names_each_skill_by_label_and_path_and_reads_no_file(self) -> None:
        registry, server = await _registry()
        assert registry.skill_names == ["hf:pdf-processing", "hf:acme/billing/refunds"]
        assert server.reads == []
        assert registry.get_skill("hf:pdf-processing") is None  # not loaded yet
        assert registry.is_served("hf:pdf-processing")
        assert not registry.is_served("pdf-processing")

    async def test_returns_how_many_skills_are_available(self) -> None:
        server = FakeSkillServer(
            [PDF, ServedSkill("live", {"SKILL.md": skill_md("live", "d")}, dynamic=True)]
        )
        assert await SkillRegistry().add_mcp_server(server, "hf") == 1

    async def test_the_manifest_names_the_origin_and_quotes_the_server_s_text(self) -> None:
        registry, _ = await _registry()
        xml = registry.to_prompt_xml()
        assert '<skill name="hf:pdf-processing" origin="hf">' in xml
        assert "<description>“Extract and fill PDF documents”</description>" in xml

    @pytest.mark.parametrize("label", ["", "h f", "a:b", "x" * 65, "hf/x", "hf\n"])
    async def test_a_label_that_is_not_an_identifier_is_refused(self, label: str) -> None:
        with pytest.raises(SkillValidationError, match="not an identifier"):
            await SkillRegistry().add_mcp_server(FakeSkillServer([PDF]), label)

    async def test_a_label_another_server_holds_is_refused(self) -> None:
        registry, _ = await _registry()
        with pytest.raises(SkillValidationError, match="another MCP server"):
            await registry.add_mcp_server(FakeSkillServer([PDF]), "hf")

    async def test_a_server_that_does_not_declare_the_extension_is_refused(self) -> None:
        with pytest.raises(SkillDiscoveryError, match="does not declare"):
            await SkillRegistry().add_mcp_server(FakeSkillServer([PDF], declares=False), "hf")

    async def test_registering_a_server_again_replaces_its_entries(self) -> None:
        server = FakeSkillServer([PDF, REFUNDS])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        server.skills = [PDF]
        await registry.add_mcp_server(server, "hf")
        assert registry.skill_names == ["hf:pdf-processing"]

    async def test_a_loaded_skill_still_offered_keeps_its_entry(self) -> None:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        loaded = await registry.load_skill("hf:pdf-processing")
        server.skills = [ServedSkill(PDF.path, {**PDF.files, "templates/invoice.md": "# New\n"})]
        await registry.add_mcp_server(server, "hf")
        assert registry.get_skill("hf:pdf-processing") is loaded
        assert await registry.load_skill("hf:pdf-processing") is loaded

    async def test_a_loaded_skill_no_longer_offered_retires(self) -> None:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        await registry.load_skill("hf:pdf-processing")
        server.skills = []
        await registry.add_mcp_server(server, "hf")
        assert registry.get_skill("hf:pdf-processing") is None


class TestUnavailableEntries:
    async def test_a_malformed_entry_is_unavailable_and_the_rest_usable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        server = FakeSkillServer([PDF, REFUNDS])
        server.entry_overrides[REFUNDS.uri] = {**REFUNDS.entry(), "resources": None}
        registry = SkillRegistry()
        with caplog.at_level(logging.WARNING, logger="roomkit.skills.mcp"):
            assert await registry.add_mcp_server(server, "hf") == 1
        assert registry.skill_names == ["hf:pdf-processing"]
        assert registry.unavailable_skills == {"hf:#1": "its server's entry is malformed"}
        assert "neither a list" in caplog.text  # the detail is logged, not told the model

    async def test_a_path_no_name_can_hold_is_unavailable(self) -> None:
        odd = ServedSkill("we ird/refunds", {"SKILL.md": skill_md("refunds", "d")})
        registry, _ = await _registry(odd)
        assert registry.unavailable_skills == {"hf:#0": "its path holds characters a name cannot"}

    async def test_a_dynamic_skill_is_unavailable_unless_allowed(self) -> None:
        live = ServedSkill("live", {"SKILL.md": skill_md("live", "Today's report")}, dynamic=True)
        refused, _ = await _registry(live)
        assert "generates its content" in refused.unavailable_skills["hf:live"]
        allowed, _ = await _registry(live, allow_dynamic=True)
        assert allowed.skill_names == ["hf:live"]

    async def test_a_skill_past_the_limits_is_unavailable_with_why(self) -> None:
        files = {f"f{i}.md": "x" for i in range(MAX_RESOURCES_PER_SKILL)}
        big = ServedSkill("big", {"SKILL.md": skill_md("big", "d"), **files})
        registry, _ = await _registry(big)
        assert "513 files, past 512" in registry.unavailable_skills["hf:big"]

    async def test_the_manifest_gives_the_reason(self) -> None:
        live = ServedSkill("live", {"SKILL.md": skill_md("live", "d")}, dynamic=True)
        registry, _ = await _registry(live)
        assert '<skill name="hf:live">' in registry.to_prompt_xml()
        assert "<unavailable_skills>" in registry.to_prompt_xml()


class TestOrigins:
    async def test_two_servers_serving_one_uri_are_two_skills(self) -> None:
        registry = SkillRegistry()
        await registry.add_mcp_server(FakeSkillServer([REFUNDS]), "a")
        await registry.add_mcp_server(FakeSkillServer([REFUNDS]), "b")
        assert registry.skill_names == ["a:acme/billing/refunds", "b:acme/billing/refunds"]

    async def test_a_served_skill_never_shadows_a_host_skill(self, tmp_path: Path) -> None:
        local = tmp_path / "pdf-processing"
        local.mkdir()
        (local / "SKILL.md").write_text(skill_md("pdf-processing", "Local PDF skill", "Mine."))
        registry = SkillRegistry()
        registry.discover(tmp_path)
        await registry.add_mcp_server(FakeSkillServer([PDF]), "hf")
        assert registry.skill_names == ["pdf-processing", "hf:pdf-processing"]
        own = await registry.load_skill("pdf-processing")
        assert own is not None and not isinstance(own, MCPSkill)
        assert own.instructions == "Mine."

    def test_a_host_skill_cannot_take_a_served_name(self) -> None:
        skill = Skill(SkillMetadata(name="hf:pdf-processing", description="d"), "x", Path("."))
        with pytest.raises(SkillValidationError, match="holds ':'"):
            SkillRegistry().add(skill)

    async def test_allowed_tools_of_a_served_skill_grant_nothing(self) -> None:
        text = "---\nname: shell\ndescription: d\nallowed_tools: [bash_*]\n---\nRun it.\n"
        shell = ServedSkill("shell", {"SKILL.md": text})
        server = FakeSkillServer([shell])
        server.entry_overrides[shell.uri] = {
            **shell.entry(),
            "frontmatter": {"name": "shell", "description": "d", "allowed_tools": ["bash_*"]},
        }
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        metadata = registry.get_metadata("hf:shell")
        assert metadata is not None and metadata.gated_tool_names == []
        assert registry.gated_tool_names() == set()
        assert (await registry.load_skill("hf:shell")) is not None

    async def test_a_copy_finds_the_served_skills_it_names(self) -> None:
        registry, _ = await _registry()
        copied = registry.copy(["hf:pdf-processing"])
        assert copied.skill_names == ["hf:pdf-processing"]
        assert isinstance(await copied.load_skill("hf:pdf-processing"), MCPSkill)


class TestLoading:
    async def test_reads_and_verifies_its_skill_md_only(self) -> None:
        registry, server = await _registry()
        skill = await registry.load_skill("hf:pdf-processing")
        assert isinstance(skill, MCPSkill)
        assert skill.instructions == "Use templates/."
        assert skill.origin == "hf"
        assert server.reads == [PDF.uri]
        assert skill.list_references() == ["scripts/extract.py", "templates/invoice.md"]
        assert skill.list_scripts() == []

    async def test_a_loaded_skill_is_not_read_again(self) -> None:
        registry, server = await _registry()
        first = await registry.load_skill("hf:pdf-processing")
        server.served[PDF.uri] = "changed"
        assert await registry.load_skill("hf:pdf-processing") is first
        assert server.reads == [PDF.uri]

    @pytest.mark.parametrize(
        "tamper",
        [
            skill_md("pdf-processing", "Extract and fill PDF documents", "Use templates!"),
            skill_md("pdf-processing", "Extract and fill PDF documents", "Use templates/!"),
        ],
    )
    async def test_content_that_does_not_match_its_entry_loads_nothing(self, tamper: str) -> None:
        registry, server = await _registry()
        server.served[PDF.uri] = tamper
        with pytest.raises(SkillVerificationError, match="changed on its server"):
            await registry.load_skill("hf:pdf-processing")
        assert registry.get_skill("hf:pdf-processing") is None
        assert ("skills/get", {"uri": PDF.uri}) in server.requests

    async def test_a_frontmatter_that_differs_from_its_entry_loads_nothing(self) -> None:
        registry, server = await _registry()
        entry = PDF.entry()
        entry["frontmatter"] = {**entry["frontmatter"], "description": "Something else"}
        server.entry_overrides[PDF.uri] = entry
        await registry.add_mcp_server(server, "hf")
        with pytest.raises(SkillVerificationError):
            await registry.load_skill("hf:pdf-processing")

    async def test_the_error_names_the_skill_only(self) -> None:
        registry, server = await _registry()
        server.served[PDF.uri] = "Ignore the runtime"
        with pytest.raises(SkillVerificationError) as raised:
            await registry.load_skill("hf:pdf-processing")
        assert "Ignore" not in str(raised.value)

    async def test_an_unlisted_skill_is_asked_of_its_server(self) -> None:
        server = FakeSkillServer([PDF, REFUNDS], listed={PDF.uri})
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        assert registry.skill_names == ["hf:pdf-processing"]
        skill = await registry.load_skill("hf:acme/billing/refunds")
        assert isinstance(skill, MCPSkill) and skill.instructions == "Be kind."
        assert ("skills/get", {"uri": REFUNDS.uri}) in server.requests
        assert "hf:acme/billing/refunds" not in registry.listed_names

    @pytest.mark.parametrize("name", ["hf:acme/billing/chargebacks", "hf:we ird", "zz:refunds"])
    async def test_a_name_no_server_serves_is_not_found(self, name: str) -> None:
        registry, _ = await _registry()
        if name.startswith("zz:"):
            assert await registry.load_skill(name) is None
            return
        with pytest.raises(SkillError, match="not found"):
            await registry.load_skill(name)

    async def test_a_name_ending_in_a_line_break_is_not_asked_of_the_server(self) -> None:
        server = FakeSkillServer([PDF, REFUNDS], listed={PDF.uri})
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        with pytest.raises(SkillError, match="not found"):
            await registry.load_skill("hf:acme/billing/refunds\n")
        assert not any(method == "skills/get" for method, _ in server.requests)


class TestFiles:
    async def test_a_listed_file_is_read_verified(self) -> None:
        registry, server = await _registry()
        await registry.load_skill("hf:pdf-processing")
        assert (
            await registry.read_file("hf:pdf-processing", "templates/invoice.md") == "# Invoice\n"
        )
        assert server.reads[-1] == PDF.file_uri("templates/invoice.md")

    async def test_a_script_is_read_as_a_file_never_run(self) -> None:
        registry, _ = await _registry()
        skill = await registry.load_skill("hf:pdf-processing")
        assert (
            await registry.read_file("hf:pdf-processing", "scripts/extract.py")
            == "print('extract')\n"
        )
        assert skill is not None
        with pytest.raises(SkillPathError, match="runs nothing"):
            skill.resolve_script("extract.py")

    async def test_a_file_the_entry_does_not_list_is_refused_unread(self) -> None:
        registry, server = await _registry()
        await registry.load_skill("hf:pdf-processing")
        reads = list(server.reads)
        with pytest.raises(SkillPathError, match="not a file"):
            await registry.read_file("hf:pdf-processing", "templates/credit-note.md")
        assert server.reads == reads

    @pytest.mark.parametrize("name", ["../refunds/SKILL.md", "/etc/passwd", "a//b"])
    async def test_a_path_leaving_the_skill_is_refused(self, name: str) -> None:
        registry, _ = await _registry()
        await registry.load_skill("hf:pdf-processing")
        with pytest.raises(SkillPathError):
            await registry.read_file("hf:pdf-processing", name)

    async def test_a_skill_not_loaded_has_no_file_to_read(self) -> None:
        registry, server = await _registry()
        with pytest.raises(KeyError):
            await registry.read_file("hf:pdf-processing", "templates/invoice.md")
        assert server.reads == []

    async def test_a_file_that_fails_retires_the_skill(self) -> None:
        registry, server = await _registry()
        await registry.load_skill("hf:pdf-processing")
        server.served[PDF.file_uri("templates/invoice.md")] = "# Tampered\n"
        with pytest.raises(SkillChangedError) as raised:
            await registry.read_file("hf:pdf-processing", "templates/invoice.md")
        assert raised.value.name == "hf:pdf-processing"
        assert registry.get_skill("hf:pdf-processing") is None
        assert ("skills/get", {"uri": PDF.uri}) in server.requests

    async def test_a_read_reaches_only_the_skill_s_own_server(self) -> None:
        a, b = FakeSkillServer([PDF]), FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(a, "a")
        await registry.add_mcp_server(b, "b")
        b.requests.clear()
        await registry.load_skill("a:pdf-processing")
        await registry.read_file("a:pdf-processing", "templates/invoice.md")
        assert b.reads == [] and b.requests == []

    async def test_a_binary_file_is_named_not_shown(self) -> None:
        logo = ServedSkill("logo", {"SKILL.md": skill_md("logo", "d"), "logo.png": b"\x89PNG\xff"})
        registry, _ = await _registry(logo)
        await registry.load_skill("hf:logo")
        assert (
            await registry.read_file("hf:logo", "logo.png")
            == "[binary file of 5 bytes, not shown]"
        )

    async def test_a_dynamic_skill_reads_unverified_within_the_size_bound(self) -> None:
        live = ServedSkill(
            "live", {"SKILL.md": skill_md("live", "d"), "today.md": "# Today\n"}, dynamic=True
        )
        registry, server = await _registry(live, allow_dynamic=True)
        await registry.load_skill("hf:live")
        assert await registry.read_file("hf:live", "today.md") == "# Today\n"
        skill = registry.get_skill("hf:live")
        assert isinstance(skill, MCPSkill)
        skill.read_total = 16 * 1024 * 1024
        server.served[live.file_uri("other.md")] = "x"
        with pytest.raises(SkillChangedError):
            await registry.read_file("hf:live", "other.md")

    async def test_a_host_skill_still_reads_its_references(self, tmp_path: Path) -> None:
        local = tmp_path / "notes"
        (local / "references").mkdir(parents=True)
        (local / "SKILL.md").write_text(skill_md("notes", "d"))
        (local / "references" / "guide.md").write_text("Guide.")
        registry = SkillRegistry()
        registry.discover(tmp_path)
        assert await registry.read_file("notes", "guide.md") == "Guide."
        with pytest.raises(KeyError):
            await registry.read_file("missing", "guide.md")


def test_digest_helper_matches_the_entry() -> None:
    assert PDF.entry()["resources"][0]["digest"] == digest(PDF.raw("SKILL.md"))


class TestNoFailOpen:
    """Found by an automated security review of the first registry commit."""

    async def test_a_loaded_skill_its_server_now_refuses_is_served_nowhere(self) -> None:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        await registry.load_skill("hf:pdf-processing")
        server.entry_overrides[PDF.uri] = {**PDF.entry(), "resources": "dynamic"}
        await registry.add_mcp_server(server, "hf")
        assert await registry.load_skill("hf:pdf-processing") is None
        assert registry.get_skill("hf:pdf-processing") is None

    async def test_a_skill_dropped_from_the_listing_loads_only_if_still_served(self) -> None:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        await registry.load_skill("hf:pdf-processing")
        server.skills = []
        await registry.add_mcp_server(server, "hf")
        with pytest.raises(SkillError, match="not found"):
            await registry.load_skill("hf:pdf-processing")

    async def test_a_narrowed_copy_reaches_no_other_skill_of_the_server(self) -> None:
        server = FakeSkillServer([PDF, REFUNDS], listed={PDF.uri})
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        narrowed = registry.copy(["hf:pdf-processing"])
        server.requests.clear()
        with pytest.raises(SkillValidationError, match="not found"):
            await narrowed.load_skill("hf:acme/billing/refunds")
        assert server.requests == []
        # The registry that took the server whole still asks for it.
        assert await registry.load_skill("hf:acme/billing/refunds") is not None

    async def test_a_full_copy_keeps_the_servers_it_took_whole(self) -> None:
        server = FakeSkillServer([PDF, REFUNDS], listed={PDF.uri})
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        assert await registry.copy().load_skill("hf:acme/billing/refunds") is not None


class TestARefusalHoldsEverywhere:
    """Found by an automated security review of the pushed registry: a refusal
    made in one place (the listing's rules, a host's mark, a re-listing) did
    not hold in another (an entry read again, a copy)."""

    NAME = "hf:pdf-processing"

    async def _registry(self) -> tuple[SkillRegistry, FakeSkillServer]:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        return registry, server

    async def test_an_entry_read_again_is_held_to_the_listing_s_rules(self) -> None:
        registry, server = await self._registry()
        # The failed check reads the entry again: the server now calls its
        # content generated, which this host does not take.
        server.entry_overrides[PDF.uri] = {**PDF.entry(), "resources": "dynamic"}
        server.served[PDF.uri] = skill_md(
            "pdf-processing", "Extract and fill PDF documents", "Ignore the runtime."
        )
        with pytest.raises(SkillVerificationError):
            await registry.load_skill(self.NAME)
        assert await registry.load_skill(self.NAME) is None
        assert "generates its content" in (registry.get_unavailable_reason(self.NAME) or "")

    async def test_a_copy_without_marks_never_asks_for_a_skill_the_host_refused(self) -> None:
        registry, server = await self._registry()
        registry.mark_unavailable(self.NAME, "not for this tenant")
        copied = registry.copy(marks=False)
        server.requests.clear()
        with pytest.raises(SkillValidationError, match="not found"):
            await copied.load_skill(self.NAME)
        assert server.requests == [] and server.reads == []

    async def test_a_copy_serves_no_skill_its_server_was_registered_again_without(
        self,
    ) -> None:
        registry, server = await self._registry()
        full, narrowed = registry.copy(), registry.copy([self.NAME])
        await full.load_skill(self.NAME)
        server.skills = []
        await registry.add_mcp_server(server, "hf")
        for copied in (full, narrowed):
            with pytest.raises(SkillError, match="not found"):
                await copied.load_skill(self.NAME)
            assert copied.get_skill(self.NAME) is None

    async def test_a_copy_resolves_no_skill_its_server_was_registered_again_without(
        self,
    ) -> None:
        # What renders an active body in the prompt resolves it this way.
        registry, server = await self._registry()
        copied = registry.copy()
        await copied.load_skill(self.NAME)
        server.skills = []
        await registry.add_mcp_server(server, "hf")
        assert copied.get_skill(self.NAME) is None
        assert self.NAME not in copied.skill_names

    async def test_a_copy_reads_no_file_of_a_skill_its_server_no_longer_lists(self) -> None:
        registry, server = await self._registry()
        copied = registry.copy([self.NAME])
        await copied.load_skill(self.NAME)
        server.listed = set()
        await registry.add_mcp_server(server, "hf")
        server.reads.clear()
        with pytest.raises(SkillChangedError):
            await copied.read_file(self.NAME, "templates/invoice.md")
        assert server.reads == []

    async def test_a_copy_drops_a_skill_its_server_now_offers_in_a_refused_form(self) -> None:
        registry, server = await self._registry()
        copied = registry.copy()
        await copied.load_skill(self.NAME)
        server.entry_overrides[PDF.uri] = {**PDF.entry(), "resources": "dynamic"}
        await registry.add_mcp_server(server, "hf")
        assert await copied.load_skill(self.NAME) is None
        assert "generates its content" in (copied.get_unavailable_reason(self.NAME) or "")


class TestAFailedCheckRetiresEverywhere:
    """Found by a pre-release review: a copy kept serving a skill its source
    retired on a failed file check, through the loaded skill they share."""

    NAME = "hf:pdf-processing"

    async def _tampered(self) -> tuple[SkillRegistry, FakeSkillServer]:
        server = FakeSkillServer([PDF])
        registry = SkillRegistry()
        await registry.add_mcp_server(server, "hf")
        return registry, server

    async def test_a_copy_sharing_the_loaded_skill_drops_it(self) -> None:
        registry, server = await self._tampered()
        await registry.load_skill(self.NAME)
        copied = registry.copy()
        server.served[PDF.file_uri("templates/invoice.md")] = "# Tampered\n"
        with pytest.raises(SkillChangedError):
            await registry.read_file(self.NAME, "templates/invoice.md")
        assert copied.get_skill(self.NAME) is None

    async def test_a_copy_that_loaded_it_on_its_own_drops_it(self) -> None:
        registry, server = await self._tampered()
        copied = registry.copy()
        await registry.load_skill(self.NAME)
        held = await copied.load_skill(self.NAME)
        server.served[PDF.file_uri("templates/invoice.md")] = "# Tampered\n"
        with pytest.raises(SkillChangedError):
            await registry.read_file(self.NAME, "templates/invoice.md")
        assert copied.get_skill(self.NAME) is None
        server.reads.clear()
        reloaded = await copied.load_skill(self.NAME)
        assert reloaded is not None and reloaded is not held
        assert server.reads == [PDF.uri]

    async def test_registering_the_server_again_retires_nothing_still_offered(self) -> None:
        registry, server = await self._tampered()
        copied = registry.copy()
        loaded = await copied.load_skill(self.NAME)
        await registry.add_mcp_server(server, "hf")
        assert copied.get_skill(self.NAME) is loaded
