"""Skills an MCP server serves under the MCP Skills extension (RFC §24.7)."""

from roomkit.skills.mcp.approval import ApproveExecution, SkillExecutionRequest
from roomkit.skills.mcp.catalog import ServedOrigin, SkillChangedError
from roomkit.skills.mcp.entry import (
    MAX_RESOURCES_PER_SKILL,
    MAX_TOTAL_SIZE,
    SKILLS_EXTENSION,
    SkillEntry,
    SkillResource,
    parse_entry,
)
from roomkit.skills.mcp.server import MCPSkillServer
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.skills.mcp.verify import verify_frontmatter, verify_skill_resource

__all__ = [
    "MAX_RESOURCES_PER_SKILL",
    "MAX_TOTAL_SIZE",
    "SKILLS_EXTENSION",
    "ApproveExecution",
    "MCPSkill",
    "MCPSkillServer",
    "ServedOrigin",
    "SkillChangedError",
    "SkillEntry",
    "SkillExecutionRequest",
    "SkillResource",
    "parse_entry",
    "verify_frontmatter",
    "verify_skill_resource",
]
