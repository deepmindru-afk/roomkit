"""A conversation pipeline driving a speech-to-speech session (RFC §19.5).

On a realtime channel the active agent is the session's configuration: its
prompt (with its identity block), its voice and its tools, the handoff tool
among them. A call to the handoff tool hands the room off; a call to one of the
active agent's own tools is served by the handler the agent was given; a
handoff reconfigures the room's sessions to the next agent's.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.orchestration.handoff import HandoffHandler, build_handoff_tool
from roomkit.orchestration.state import get_conversation_state
from roomkit.tools.context import current_tool_room_id

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.channels.ai import ToolResult
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.core.framework import RoomKit
    from roomkit.orchestration.pipeline import PipelineStage
    from roomkit.providers.ai.base import AITool

_DEFAULT_GREETING = (
    "Handoff complete. You are now the active agent. "
    "Please introduce yourself briefly to the caller."
)


class RealtimePipeline:
    """A pipeline's agents on one realtime channel: each agent's session
    configuration, the calls its sessions make, and its handoffs."""

    def __init__(
        self,
        kit: RoomKit,
        rtv: RealtimeVoiceChannel,
        agents: list[Agent],
        stages: list[PipelineStage],
        handler: HandoffHandler,
        default_agent_id: str,
        *,
        greet_on_handoff: bool,
        greeting_prompt: str | None,
    ) -> None:
        self._kit = kit
        self._rtv = rtv
        self._handler = handler
        self._agent_map: dict[str, Agent] = {a.channel_id: a for a in agents}
        self._default_agent_id = default_agent_id
        self._greet_on_handoff = greet_on_handoff
        self._greeting_prompt = greeting_prompt
        self.agent_configs: dict[str, dict[str, Any]] = {
            agent.channel_id: self._agent_config(agent, stages) for agent in agents
        }

    def _agent_config(self, agent: Agent, stages: list[PipelineStage]) -> dict[str, Any]:
        """The prompt, voice and tools *agent*'s sessions run with."""
        prompt = agent.system_prompt or ""
        identity = agent.build_identity_block()
        if identity:
            prompt = prompt + identity
        return {
            "system_prompt": prompt or None,
            "voice": agent.voice,
            "tools": _agent_session_tools(self._rtv, agent, self._handoff_tool(agent, stages)),
        }

    def _handoff_tool(self, agent: Agent, stages: list[PipelineStage]) -> AITool:
        """The handoff tool *agent* declares, its targets the stages it reaches."""
        stage = next((s for s in stages if s.agent_id == agent.channel_id), None)
        if stage is None:
            return build_handoff_tool([])
        reachable: set[str] = set()
        if stage.next:
            reachable.add(stage.next)
        reachable.update(stage.can_return_to)
        targets: list[tuple[str, str | None]] = []
        for s in stages:
            if s.phase in reachable and s.agent_id != agent.channel_id:
                ta = self._agent_map.get(s.agent_id)
                desc = ta.description if ta else None
                if desc is None:
                    desc = s.description
                targets.append((s.agent_id, desc))
        return build_handoff_tool(targets)

    def greeting(self, agent_id: str, language: str | None = None) -> str:
        """What *agent_id* is told to say when a handoff makes it active."""
        if self._greeting_prompt:
            msg = self._greeting_prompt
        else:
            target = self._agent_map.get(agent_id)
            role = target.role if target else None
            if role:
                msg = (
                    f"Handoff complete. You are now the {role}. "
                    f"Your previous identity in this conversation no longer "
                    f"applies — introduce yourself in your new role."
                )
            else:
                msg = _DEFAULT_GREETING
        lang = language
        if not lang and agent_id in self._agent_map:
            lang = getattr(self._agent_map[agent_id], "language", None)
        if lang:
            msg = f"[Respond in {lang}] {msg}"
        return msg

    async def serve_handoff(self, arguments: dict[str, Any]) -> ToolResult:
        """Hand the room of the call off to the agent *arguments* name."""
        # Lazy import to avoid circular dependency
        from roomkit.channels.realtime_voice import get_current_voice_session

        kit = self._kit
        session = get_current_voice_session()
        session_id = session.id if session else None
        room_id = self._rtv.session_rooms.get(session_id) if session_id else None
        if not room_id:
            return json.dumps({"error": "No room context for this session"})

        room = await kit.get_room(room_id)
        state = get_conversation_state(room)
        calling_agent = state.active_agent_id or self._default_agent_id

        result = await self._handler.handle(
            room_id=room_id,
            calling_agent_id=calling_agent,
            arguments=arguments,
        )

        output = result.model_dump()
        if result.accepted and self._greet_on_handoff:
            target = arguments.get("target", "")
            # Re-read room for current language
            room = await kit.get_room(room_id)
            lang = self._handler.get_room_language(room, target)
            output["message"] = self.greeting(target, language=lang)
        return json.dumps(output)

    async def serve_agent_tool(
        self, channel_handler: Any, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Serve a call other than the handoff (RFC §19.5): the active agent's
        own tool by the handler the agent was given, any other tool, and an
        agent tool the agent has no handler for, by the channel's."""
        agent = await self._active_agent()
        agent_handler = agent._user_tool_handler if agent is not None else None
        agent_tools = agent._user_tools if agent is not None else []
        if agent_handler is not None and any(t.name == name for t in agent_tools):
            return await agent_handler(name, arguments)
        if channel_handler is not None:
            return await channel_handler(name, arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _active_agent(self) -> Agent | None:
        """The agent the call's room is talking to, by its conversation state."""
        room_id = current_tool_room_id()
        if room_id is None:
            return None
        state = get_conversation_state(await self._kit.get_room(room_id))
        return self._agent_map.get(state.active_agent_id or self._default_agent_id)

    async def on_handoff_complete(self, room_id: str, result: Any) -> None:
        """Reconfigure *room_id*'s sessions to the agent a handoff made active."""
        new_id = result.new_agent_id
        if not new_id or new_id not in self.agent_configs:
            return
        config = self.agent_configs[new_id]
        rtv = self._rtv

        # Check for per-room language override
        room = await self._kit.get_room(room_id)
        lang = self._handler.get_room_language(room, new_id)

        # Rebuild prompt with language if needed
        prompt = config["system_prompt"]
        if lang:
            agent = self._agent_map.get(new_id)
            if agent is not None:
                base = getattr(agent, "system_prompt", None) or ""
                identity = agent.build_identity_block(language=lang)
                prompt = (base + identity) if identity else prompt

        for session in rtv.get_room_sessions(room_id):
            await rtv.reconfigure_session(
                session,
                system_prompt=prompt,
                voice=config["voice"],
                tools=config["tools"],
            )

            if self._greet_on_handoff:
                # Session resumption doesn't preserve pending function-
                # call state, so the tool result alone won't trigger a
                # response.  Inject a language-aware instruction to give
                # the new agent a turn to speak in its new role. It
                # directs the model, so it carries the system intent: a
                # full-duplex provider voices a user injection instead
                # of following it (RFC §12.4).
                msg = self.greeting(new_id, language=lang)
                await rtv.provider.inject_text(
                    session,
                    msg,
                    role="system",
                )


def _agent_session_tools(
    rtv: RealtimeVoiceChannel, agent: Agent, handoff: AITool
) -> list[dict[str, Any]]:
    """The tools an agent's realtime session declares (RFC §19.5).

    The channel's own tools, which stay declared under every agent, the
    agent's, then the handoff tool; a later tool replaces an earlier one of
    the same name, so an agent may specialise one.
    """
    declared = [
        *(dict(t) for t in rtv._tools or []),
        *(t.model_dump() for t in agent._user_tools),
        handoff.model_dump(),
    ]
    return list({tool["name"]: tool for tool in declared}.values())
