"""Agent Skills integration for RoomKit."""

from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillError,
    SkillParseError,
    SkillPathError,
    SkillValidationError,
    SkillVerificationError,
)
from roomkit.skills.executor import ScriptExecutor
from roomkit.skills.mcp import MCPSkillServer, SkillEntry, SkillResource
from roomkit.skills.models import (
    RequiresMatch,
    ScriptResult,
    Skill,
    SkillMetadata,
    missing_required_tools,
    serves_exactly,
)
from roomkit.skills.paths import safe_join_filename
from roomkit.skills.registry import SkillRegistry

__all__ = [
    "MCPSkillServer",
    "RequiresMatch",
    "ScriptExecutor",
    "ScriptResult",
    "Skill",
    "SkillDiscoveryError",
    "SkillEntry",
    "SkillError",
    "SkillMetadata",
    "SkillParseError",
    "SkillPathError",
    "SkillRegistry",
    "SkillResource",
    "SkillValidationError",
    "SkillVerificationError",
    "missing_required_tools",
    "safe_join_filename",
    "serves_exactly",
]
