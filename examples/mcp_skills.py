"""RoomKit -- the skills an MCP server publishes (MCP Skills extension).

An MCP server may serve skills beside its tools under the MCP Skills
extension (``io.modelcontextprotocol/skills``, SEP-2640). ``add_mcp_server``
lists them and names each ``<label>:<path>``: nothing of a skill is read
before it is activated. Loading one reads its ``SKILL.md`` from that server
and checks it against the listing (size, SHA-256, frontmatter field by field)
before anything of it reaches the model, as that server's data (RFC §24.7).

    MCPToolProvider ── https://huggingface.co/mcp
        └── SkillRegistry.add_mcp_server(mcp, "hf")
              ├── load_skill("hf:<path>")           SKILL.md read and verified
              ├── read_file("hf:<path>", "<file>")  one listed file, verified
              └── AIChannel(skills=registry)        activate_skill on demand

With ``--chat``, a CLI conversation follows: ask how to run a model on a GPU
Space and watch the agent activate a skill the server serves.

Requirements:
    pip install roomkit[mcp]

Run:
    uv run python examples/mcp_skills.py
    uv run python examples/mcp_skills.py --skill hf:hf-cli
    ANTHROPIC_API_KEY=sk-... uv run python examples/mcp_skills.py --chat
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import log_tool_call, require_env, setup_logging

from roomkit import AIChannel, ChannelCategory, CLIChannel, HookTrigger, RoomKit
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.skills import SkillRegistry
from roomkit.tools import MCPToolProvider

logger = setup_logging("mcp_skills")

HF_MCP = "https://huggingface.co/mcp"
DEFAULT_SKILL = "hf:huggingface-zerogpu"


async def inspect(registry: SkillRegistry, name: str) -> None:
    """Load one served skill and read its first listed file, both verified."""
    skill = await registry.load_skill(name)
    if skill is None:
        logger.info("%s is unavailable: %s", name, registry.get_unavailable_reason(name))
        return
    logger.info("Loaded %s: %d characters, matching its entry", name, len(skill.instructions))
    files = skill.list_references()
    logger.info("Files its entry lists (%d): %s", len(files), ", ".join(files[:5]))
    if files:
        text = await registry.read_file(name, files[0])
        logger.info("Read %s: %d characters, matching its digest", files[0], len(text))


async def chat(registry: SkillRegistry) -> None:
    """A CLI conversation with an agent that activates served skills."""
    env = require_env("ANTHROPIC_API_KEY")
    kit = RoomKit()
    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(
            AnthropicConfig(
                api_key=env["ANTHROPIC_API_KEY"], model="claude-opus-5", max_tokens=4096
            )
        ),
        system_prompt=(
            "You help people build with Hugging Face. When a request matches a "
            "skill, activate it first and follow it."
        ),
        skills=registry,
    )
    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="skill")

    await kit.create_room(room_id="mcp-skills")
    await kit.attach_channel("mcp-skills", "cli")
    await kit.attach_channel("mcp-skills", "ai-assistant", category=ChannelCategory.INTELLIGENCE)
    await cli.run(
        kit,
        room_id="mcp-skills",
        welcome=f"\n{len(registry.skill_names)} skills served by {HF_MCP}. Ask away.\n",
    )
    await kit.close()


async def main(args: argparse.Namespace) -> None:
    # discover=False: the skills are wanted, not the server's tool catalogue.
    async with MCPToolProvider.from_url(HF_MCP, discover=False) as hf:
        registry = SkillRegistry()
        available = await registry.add_mcp_server(hf, "hf")
        logger.info(
            "%d skills listed, %d unavailable; no SKILL.md read yet",
            available,
            len(registry.unavailable_skills),
        )
        for name in registry.skill_names[:5]:
            logger.info("  %s: %s", name, registry.get_metadata(name).description[:70])

        name = args.skill
        if name == DEFAULT_SKILL and name not in registry.skill_names and registry.skill_names:
            name = registry.skill_names[0]
        await inspect(registry, name)

        if args.chat:
            await chat(registry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skill", default=DEFAULT_SKILL, help="the served skill to load")
    parser.add_argument("--chat", action="store_true", help="then chat with an agent")
    asyncio.run(main(parser.parse_args()))
