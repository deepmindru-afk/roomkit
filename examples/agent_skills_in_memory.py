"""Skills from a store, a narrowed copy per agent, and the turn's measured footprint.

A host keeps its skills in a database rather than on disk, gives each agent
the subset it is allowed, and lets RoomKit size the history to what the window
leaves. Shows:
- SkillRegistry.add(): a skill built in memory, registered as a directory is
- SkillRegistry.copy(): an agent's subset, each skill's path and marks kept
- current_turn_footprint(): what the turn takes besides its history, its input
  and its reply budget, measured by the channel before it reads its memory;
  BudgetAwareMemory reserves it
- RunSkillScriptTool: run_skill_script as a Tool, for a realtime channel
  serving skills another agent holds

Run with:
    uv run python examples/agent_skills_in_memory.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    RunSkillScriptTool,
    TextContent,
    WebSocketChannel,
)
from roomkit.memory import (
    BudgetAwareMemory,
    MemoryProvider,
    MemoryResult,
    SlidingWindowMemory,
    current_turn_footprint,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import ScriptExecutor, ScriptResult, Skill, SkillMetadata, SkillRegistry

logger = setup_logging("agent_skills_in_memory")


class FootprintLog(MemoryProvider):
    """Logs the footprint the channel measured, then reads the room as usual."""

    def __init__(self, inner: MemoryProvider) -> None:
        self._inner = inner

    async def retrieve(
        self, room_id: str, current_event: RoomEvent, context: RoomContext, **kwargs: Any
    ) -> MemoryResult:
        footprint = current_turn_footprint()
        if footprint is not None:
            logger.info(
                "turn footprint besides the history: %d input tokens, a %d-token reply",
                footprint.input_tokens,
                footprint.reply_tokens,
            )
        return await self._inner.retrieve(room_id, current_event, context, **kwargs)


class EchoExecutor(ScriptExecutor):
    async def execute(
        self, skill: Skill, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        return ScriptResult(exit_code=0, stdout=f"{skill.name}/{script_name} ran")


def skills_from_store(root: Path) -> SkillRegistry:
    """Skills as a database would hand them back, written where their scripts live."""
    registry = SkillRegistry()
    for name, body in [("refunds", "Refund within 30 days."), ("invoices", "Quote the number.")]:
        (root / name / "scripts").mkdir(parents=True)
        (root / name / "scripts" / "lookup.py").write_text("print('ok')\n")
        metadata = SkillMetadata(name=name, description=f"How we handle {name}")
        registry.add(Skill(metadata=metadata, instructions=body, path=root / name))
    return registry


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="skills-"))
    catalogue = skills_from_store(root)
    allowed = catalogue.copy(["refunds"])
    logger.info("catalogue %s, this agent %s", catalogue.skill_names, allowed.skill_names)

    memory = BudgetAwareMemory(
        FootprintLog(SlidingWindowMemory(max_events=50)), max_context_tokens=8_000
    )
    agent = Agent(
        "support",
        provider=MockAIProvider(responses=["Refunds are accepted within 30 days."]),
        role="Billing support",
        system_prompt="Answer billing questions.",
        max_tokens=400,
        skills=allowed,
        memory=memory,
    )
    kit = RoomKit()
    kit.register_channel(agent)
    kit.register_channel(WebSocketChannel("member"))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "member")
    await kit.attach_channel("room", "support", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="member", sender_id="u1", content=TextContent(body="Refund?"))
    )

    # A realtime channel serving these skills' scripts would take this tool.
    tool = RunSkillScriptTool(allowed, EchoExecutor())
    result = await tool.handler(tool.name, {"skill_name": "refunds", "script_name": "lookup.py"})
    logger.info("run_skill_script: %s", result)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
