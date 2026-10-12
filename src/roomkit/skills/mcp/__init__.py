"""Skills an MCP server serves under the MCP Skills extension (RFC §24.7)."""

from roomkit.skills.mcp.entry import (
    MAX_RESOURCES_PER_SKILL,
    MAX_TOTAL_SIZE,
    SKILLS_EXTENSION,
    SkillEntry,
    SkillResource,
    parse_entry,
)
from roomkit.skills.mcp.server import MCPSkillServer
from roomkit.skills.mcp.verify import verify_frontmatter, verify_skill_resource

__all__ = [
    "MAX_RESOURCES_PER_SKILL",
    "MAX_TOTAL_SIZE",
    "MCPSkillServer",
    "SKILLS_EXTENSION",
    "SkillEntry",
    "SkillResource",
    "parse_entry",
    "verify_frontmatter",
    "verify_skill_resource",
]
