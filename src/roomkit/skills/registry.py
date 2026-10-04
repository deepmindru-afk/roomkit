"""Skill registry for discovering and managing Agent Skills."""

from __future__ import annotations

import logging
from collections.abc import Container, Iterable
from html import escape
from pathlib import Path

from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillParseError,
    SkillValidationError,
)
from roomkit.skills.models import RequiresMatch, Skill, SkillMetadata, serves_exactly
from roomkit.skills.parser import (
    find_skill_md,
    parse_skill,
    parse_skill_metadata,
)

logger = logging.getLogger("roomkit.skills")


class SkillRegistry:
    """Discover, load, and manage Agent Skills.

    Lightweight metadata is parsed on discover/register. Full skill
    instructions are loaded lazily on first ``get_skill()`` call and
    cached for subsequent access.
    """

    def __init__(self, *, requires_match: RequiresMatch = serves_exactly) -> None:
        """*requires_match* says whether a skill's ``requires`` name is served
        by the tools a conversation offers (RFC §24.3): an exact tool name by
        default, a host's own reading otherwise (a hub served by any tool
        named ``<hub>_*``, say). Every door checks ``requires`` through it."""
        self.requires_match: RequiresMatch = requires_match
        self._metadata: dict[str, SkillMetadata] = {}
        self._skills: dict[str, Skill] = {}
        self._paths: dict[str, Path] = {}
        self._unavailable: dict[str, str] = {}
        # The patterns a skill marked unavailable gated: it can never be
        # activated here, so what it gates stays closed (RFC §24.2).
        self._closed_gates: dict[str, list[str]] = {}
        self._unlisted: set[str] = set()

    def discover(self, *directories: str | Path, strict: bool = True) -> int:
        """Scan directories for subdirectories containing SKILL.md.

        A malformed skill is a deployment error, not a runtime condition, so
        the default is to stop. Skipping it instead removes the skill from the
        catalogue while the agent keeps answering: the model is never told the
        capability is missing, and neither is anyone reading the conversation.
        The failure surfaces hours later as an agent that quietly cannot do
        something it was configured to do.

        Discovery commits only once every candidate has parsed, so a strict
        failure leaves the registry exactly as it was rather than half filled.

        Args:
            directories: Directories to scan. Only immediate subdirectories
                containing a SKILL.md are considered.
            strict: Stop on the first unreadable directory or invalid skill.
                Set False to log and skip each instead — appropriate when
                skills come from a source you do not control.

        Returns:
            The number of skills registered.

        Raises:
            SkillDiscoveryError: In strict mode, when a directory cannot be
                scanned or a candidate escapes it through a symlink.
            SkillParseError: In strict mode, when a SKILL.md cannot be parsed.
            SkillValidationError: In strict mode, when metadata is invalid.
        """
        parsed: list[tuple[Path, SkillMetadata]] = []
        for skill_dir in self._candidates(directories, strict=strict):
            try:
                parsed.append((skill_dir, parse_skill_metadata(skill_dir)))
            except (SkillParseError, SkillValidationError) as exc:
                if strict:
                    raise
                logger.warning("Skipping invalid skill %s: %s", skill_dir.name, exc)
        for skill_dir, metadata in parsed:
            self._commit(skill_dir, metadata)
        return len(parsed)

    def _candidates(self, directories: tuple[str | Path, ...], *, strict: bool) -> list[Path]:
        """Skill directories found under ``directories``, in a stable order."""
        candidates: list[Path] = []
        for directory in directories:
            try:
                dir_path = Path(directory).resolve()
            except (OSError, RuntimeError) as exc:
                message = f"Skill directory cannot be resolved: {directory}"
                if strict:
                    raise SkillDiscoveryError(message) from exc
                logger.warning("%s: %s", message, exc)
                continue
            if not dir_path.is_dir():
                if strict:
                    raise SkillDiscoveryError(f"Skill directory not found: {dir_path}")
                logger.warning("Skill directory not found: %s", dir_path)
                continue
            try:
                children = sorted(dir_path.iterdir())
            except (OSError, RuntimeError) as exc:
                message = f"Skill directory cannot be scanned: {dir_path}"
                if strict:
                    raise SkillDiscoveryError(message) from exc
                logger.warning("%s: %s", message, exc)
                continue

            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    resolved_child = child.resolve()
                    if dir_path not in resolved_child.parents:
                        raise SkillDiscoveryError(
                            f"Skill candidate escapes discovery directory: {child}"
                        )
                    if find_skill_md(resolved_child) is not None:
                        candidates.append(resolved_child)
                except SkillDiscoveryError:
                    if strict:
                        raise
                    logger.warning("Skipping skill candidate outside %s: %s", dir_path, child)
                except (OSError, RuntimeError) as exc:
                    message = f"Skill candidate cannot be inspected: {child}"
                    if strict:
                        raise SkillDiscoveryError(message) from exc
                    logger.warning("%s: %s", message, exc)
        return candidates

    def register(self, skill_dir: str | Path) -> SkillMetadata:
        """Register a single skill directory.

        Parses frontmatter only (lightweight). Replaces any existing
        skill with the same name.

        Raises:
            SkillParseError: If SKILL.md cannot be found or parsed.
            SkillValidationError: If metadata fails validation.
        """
        try:
            skill_path = Path(skill_dir).resolve()
        except (OSError, RuntimeError) as exc:
            raise SkillDiscoveryError(f"Skill directory cannot be resolved: {skill_dir}") from exc
        metadata = parse_skill_metadata(skill_path)
        self._commit(skill_path, metadata)
        return metadata

    def _commit(
        self, skill_path: Path, metadata: SkillMetadata, *, log_level: int = logging.INFO
    ) -> None:
        """Record a parsed skill, replacing any earlier one of the same name."""
        self._metadata[metadata.name] = metadata
        self._paths[metadata.name] = skill_path
        # Invalidate cached full skill if re-registering
        self._skills.pop(metadata.name, None)
        # Registering makes the skill usable again — drop any stale mark
        self._unavailable.pop(metadata.name, None)
        self._closed_gates.pop(metadata.name, None)
        self._unlisted.discard(metadata.name)
        logger.log(log_level, "Registered skill: %s", metadata.name)

    def add(self, skill: Skill) -> None:
        """Register a skill built in memory (a store, a marketplace), replacing
        any earlier one of the same name (RFC §24.3).

        Its ``path`` is where its scripts and references are read from. Adding
        it clears its marks, as registering a directory does. Logged at debug:
        a store adds its whole catalogue at once, and says so itself.
        """
        self._commit(skill.path, skill.metadata, log_level=logging.DEBUG)
        self._skills[skill.metadata.name] = skill

    def copy(self, names: Iterable[str] | None = None, *, marks: bool = True) -> SkillRegistry:
        """A registry of *names*, every registered skill when ``None`` (RFC §24.3).

        It finds every skill the source finds, one discovered but not yet
        loaded included: each name keeps its path and, once loaded, its skill.
        With *marks*, the copy keeps the source's unlisted marks and the
        unavailable reasons of the names it copies; without, every skill it
        copies is available and listed (a hand-picked set, say), and a name
        the source marks unavailable is not copied at all.
        """
        wanted = set(self._metadata) | set(self._unavailable) if names is None else set(names)
        copied = SkillRegistry(requires_match=self.requires_match)
        for name, metadata in self._metadata.items():
            if name not in wanted:
                continue
            copied._metadata[name] = metadata
            if name in self._paths:
                copied._paths[name] = self._paths[name]
            if name in self._skills:
                copied._skills[name] = self._skills[name]
        if marks:
            copied._unlisted = self._unlisted & set(copied._metadata)
            copied._unavailable = {
                name: reason for name, reason in self._unavailable.items() if name in wanted
            }
            copied._closed_gates = {
                name: gates for name, gates in self._closed_gates.items() if name in wanted
            }
        return copied

    def get_metadata(self, name: str) -> SkillMetadata | None:
        """Get metadata for a skill by name."""
        return self._metadata.get(name)

    def get_skill(self, name: str) -> Skill | None:
        """Get full skill (with instructions), loading lazily if needed."""
        if name in self._skills:
            return self._skills[name]

        if name not in self._paths:
            return None

        try:
            skill = parse_skill(self._paths[name])
            self._skills[name] = skill
            return skill
        except (SkillParseError, SkillValidationError) as exc:
            logger.error("Failed to load skill %s: %s", name, exc)
            return None

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Record a skill that exists but cannot be used in this context.

        The skill leaves the available set entirely — ``skill_names``,
        ``all_metadata()`` and ``get_skill()`` no longer see it — and only
        surfaces through ``unavailable_skills``, the prompt manifest and
        the skill-tool error paths, so callers can say WHY it is missing
        instead of leaving a silent gap (e.g. a ``requires`` gate dropping
        a skill whose tools are not granted in this execution context).

        What it gated stays closed: no activation can open it any more
        (:meth:`gated_tool_names`).
        """
        metadata = self._metadata.pop(name, None)
        if metadata is not None and metadata.gated_tool_names:
            self._closed_gates[name] = metadata.gated_tool_names
        self._skills.pop(name, None)
        self._paths.pop(name, None)
        self._unavailable[name] = reason

    def gated_tool_names(self, activated: Container[str] = ()) -> set[str]:
        """The tool patterns skills keep closed (RFC §24.2): those of every
        registered skill not in *activated*, and the closed ones
        (:meth:`closed_tool_names`). Both doors read skill gating through
        this one rule."""
        gated = {
            pattern
            for metadata in self._metadata.values()
            if metadata.name not in activated
            for pattern in metadata.gated_tool_names
        }
        return gated | self.closed_tool_names(activated)

    def closed_tool_names(self, activated: Container[str] = ()) -> set[str]:
        """The tool patterns only skills marked unavailable gate, which
        nothing can open (RFC §24.2). A pattern a skill in *activated* gates is
        that skill's to open: an unavailable skill gating the same pattern does
        not hold it closed."""
        opened = {
            pattern
            for metadata in self._metadata.values()
            if metadata.name in activated
            for pattern in metadata.gated_tool_names
        }
        return {
            pattern
            for gates in self._closed_gates.values()
            for pattern in gates
            if pattern not in opened
        }

    def unopenable_tool_names(self) -> set[str]:
        """The closed patterns no available skill gates as well: no
        activation can open them (RFC §24.2)."""
        return self.closed_tool_names(self._metadata)

    def mark_unlisted(self, name: str) -> None:
        """Keep *name* activatable but out of the prompt manifest.

        The third visibility state, between available and unavailable: the
        skill stays registered — ``activate_skill``, ``get_skill()`` and
        ``skill_names`` still see it — but ``to_prompt_xml()`` and
        ``listed_names`` do not. For catalogues where advertising every
        entry would drown the ones that matter: a host can keep quiet about
        a skill while any path that names it (a recommender nudge, a user
        asking for it) still activates it, which is what lets it earn its
        listing back.

        Unknown names are ignored — there is nothing to hide.
        """
        if name in self._metadata:
            self._unlisted.add(name)

    @property
    def listed_names(self) -> list[str]:
        """Names of registered skills that the prompt manifest shows."""
        return [name for name in self._metadata if name not in self._unlisted]

    def get_unavailable_reason(self, name: str) -> str | None:
        """Reason a skill is unavailable in this context, or None."""
        return self._unavailable.get(name)

    @property
    def unavailable_skills(self) -> dict[str, str]:
        """Mapping of unavailable skill name -> reason."""
        return dict(self._unavailable)

    def all_metadata(self) -> list[SkillMetadata]:
        """Return metadata for all registered skills."""
        return list(self._metadata.values())

    @property
    def skill_names(self) -> list[str]:
        """Names of all registered skills."""
        return list(self._metadata.keys())

    @property
    def skill_count(self) -> int:
        """Number of registered skills."""
        return len(self._metadata)

    @property
    def has_entries(self) -> bool:
        """Whether the model has anything to be told: a skill it can activate,
        or one marked unavailable, whose reason it gives instead of guessing."""
        return bool(self._metadata or self._unavailable)

    def to_prompt_xml(self) -> str:
        """Generate spec-compliant <available_skills> XML block.

        Skills marked unavailable are listed in a separate
        ``<unavailable_skills>`` block with their reason, so the model
        can explain the gap instead of guessing at a name that will
        answer "not found". Skills marked unlisted are simply absent —
        activatable, but not advertised. Content is HTML-escaped to
        prevent injection.
        """
        listed = [meta for meta in self._metadata.values() if meta.name not in self._unlisted]
        if not listed and not self._unavailable:
            return ""

        lines: list[str] = []
        if listed:
            lines.append("<available_skills>")
            for meta in listed:
                lines.append(f'  <skill name="{escape(meta.name)}">')
                lines.append(f"    <description>{escape(meta.description)}</description>")
                if meta.license:
                    lines.append(f"    <license>{escape(meta.license)}</license>")
                if meta.compatibility:
                    lines.append(
                        f"    <compatibility>{escape(meta.compatibility)}</compatibility>"
                    )
                lines.append("  </skill>")
            lines.append("</available_skills>")
        if self._unavailable:
            lines.append("<unavailable_skills>")
            for name, reason in self._unavailable.items():
                lines.append(f'  <skill name="{escape(name)}">')
                lines.append(f"    <reason>{escape(reason)}</reason>")
                lines.append("  </skill>")
            lines.append("</unavailable_skills>")
        return "\n".join(lines)
