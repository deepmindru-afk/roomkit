"""Pipeline orchestration strategy.

Composes ``ConversationPipeline``, ``ConversationRouter``, and
``HandoffHandler`` into a declarative strategy that can be passed to
``RoomKit(orchestration=...)`` or ``create_room(orchestration=...)``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from roomkit.channels.agent import Agent
from roomkit.channels.declared import id_of
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration._agents import HeldAgent, held
from roomkit.orchestration._handoff_greeting import HandoffGreeting
from roomkit.orchestration._realtime_pipeline import (
    RealtimePipeline,
    refuse_agents_with_unserved,
)
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.handoff import (
    HandoffHandler,
    build_handoff_tool,
    handoff_entry,
    require_live_agents,
    set_up_handoffs,
)
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage, wire_voice_map
from roomkit.orchestration.state import (
    ConversationState,
    save_conversation_state,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.channels.base import Channel
    from roomkit.core.framework import RoomKit
    from roomkit.orchestration.router import ConversationRouter

logger = logging.getLogger("roomkit.orchestration.strategies.pipeline")


class Pipeline(Orchestration):
    """Linear pipeline orchestration strategy.

    Agents are chained in order: the first agent is the entry point,
    each subsequent agent is reachable via handoff from its predecessor.

    Example::

        kit = RoomKit(
            orchestration=Pipeline(
                agents=[triage, support, billing],
            ),
        )
        room = await kit.create_room()
    """

    def __init__(
        self,
        agents: Sequence[Agent | str],
        routing: dict[str, list[str]] | None = None,
        *,
        supervisor_id: str | None = None,
        voice_channel_id: str | None = None,
        greet_on_handoff: bool = False,
        greeting_prompt: str | None = None,
        farewell_prompt: str | None = None,
    ) -> None:
        """Initialise the pipeline strategy.

        Args:
            agents: Ordered list of agents, each live or named by the id of
                one the kit holds, by its declaration alone included (RFC
                §10.6.2). First agent is the entry point. Stages chain
                linearly (each agent hands off to the next).
            routing: Optional keyword hints injected into agent system
                prompts (not runtime rules). Keys are agent channel IDs,
                values are keyword lists describing when to route there.
            supervisor_id: Optional supervisor agent ID that receives
                all events for monitoring.
            voice_channel_id: The voice channel the agents speak on. A
                realtime voice channel runs each room's active agent in its
                sessions (live agents only) and reconfigures them on a
                handoff; another voice channel takes the agents' voices.
            greet_on_handoff: The agent a handoff made active introduces
                itself; requires *voice_channel_id*.
            greeting_prompt: What that agent is cued with to introduce itself.
            farewell_prompt: Spoken before the greeting, on a voice channel
                that is not realtime, the outgoing agent's speech held
                meanwhile.

        Raises:
            ValueError: *greet_on_handoff* without *voice_channel_id*.
        """
        if greet_on_handoff and not voice_channel_id:
            raise ValueError("greet_on_handoff=True requires voice_channel_id")
        self._agents = list(agents)
        self._routing = routing or {}
        self._supervisor_id = supervisor_id
        self._voice_channel_id = voice_channel_id
        self._greet_on_handoff = greet_on_handoff
        self._greeting_prompt = greeting_prompt
        self._farewell_prompt = farewell_prompt
        # The realtime channel's wiring, built at the first install: the
        # channel serves the handoff and reads its sessions' configuration
        # from one wiring, which runs the rooms the strategy is installed in
        # only, each with its install's handler.
        self._realtime: RealtimePipeline | None = None
        self._realtime_rooms: dict[str, HandoffHandler] = {}

    def agents(self) -> list[Agent | str]:
        """Return all pipeline agents."""
        return list(self._agents)

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire pipeline routing and handoff into the room, on its voice
        channel when one is named (RFC §19.5)."""
        agents = [held(kit, a) for a in self._agents]
        rtv = self._realtime_channel(kit)
        if rtv is not None:
            # Refused before anything is installed.
            require_live_agents(agents)
            live = [a for a in agents if isinstance(a, Agent)]
            refuse_agents_with_unserved(live, rtv.channel_id)
        kit.claim_room_strategy(room_id, self)
        cp = ConversationPipeline(stages=_build_stages(agents), supervisor_id=self._supervisor_id)
        router = cp.to_router()
        handler = _handler(kit, agents, cp, router, rtv)

        # Install router as room-scoped BEFORE_BROADCAST hook
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=router.as_hook(),
                priority=-100,
                name=f"pipeline_router_{room_id}",
            ),
        )

        if rtv is not None:
            self._wire_realtime(kit, agents, handler, cp, rtv, room_id)
        else:
            self._wire_handoff(kit, agents, handler, cp, room_id)
            self._wire_voice(kit, agents, room_id)

        # Set initial conversation state
        initial_state = ConversationState(
            phase=cp.stages[0].phase,
            active_agent_id=id_of(self._agents[0]),
        )
        await save_conversation_state(kit.store, room_id, initial_state)

    async def uninstall(self, kit: RoomKit, room_id: str) -> None:
        """The realtime channel's wiring stops running *room_id*: its next
        session keeps the channel's own configuration."""
        self._realtime_rooms.pop(room_id, None)

    def _realtime_channel(self, kit: RoomKit) -> RealtimeVoiceChannel | None:
        """The realtime voice channel *voice_channel_id* names, if it names one."""
        channel = kit.channels.get(self._voice_channel_id) if self._voice_channel_id else None
        return channel if isinstance(channel, RealtimeVoiceChannel) else None

    def _wire_realtime(
        self,
        kit: RoomKit,
        agents: Sequence[Channel],
        handler: HandoffHandler,
        cp: ConversationPipeline,
        rtv: RealtimeVoiceChannel,
        room_id: str,
    ) -> None:
        """Run *room_id*'s active agent in *rtv*'s sessions, through the one
        wiring of the strategy, built at its first install, which runs the
        strategy's rooms alone; a handoff in the room reconfigures the room's
        sessions (RFC §19.5)."""
        self._realtime_rooms[room_id] = handler
        if self._realtime is None:
            live = [a for a in agents if isinstance(a, Agent)]
            self._realtime = RealtimePipeline(
                kit,
                rtv,
                live,
                cp.stages,
                handler,
                cp.stages[0].agent_id,
                greet_on_handoff=self._greet_on_handoff,
                greeting_prompt=self._greeting_prompt,
                handler_for=self._realtime_rooms.get,
            )
            self._realtime.install()
        handler.on_handoff_complete = self._realtime.on_handoff_complete

    def _wire_voice(self, kit: RoomKit, agents: Sequence[HeldAgent], room_id: str) -> None:
        """On a voice channel, the agents' voices, and a greeting on each
        handoff in *room_id* when asked: room hooks, never the kit's."""
        if self._voice_channel_id is None:
            return
        wire_voice_map(kit, agents, self._voice_channel_id)
        if not self._greet_on_handoff:
            return
        greeting = HandoffGreeting(
            kit, self._voice_channel_id, self._greeting_prompt, self._farewell_prompt
        )
        for hook in greeting.hooks(priority=-100):
            named = dataclasses.replace(hook, name=f"pipeline{hook.name}_{room_id}")
            kit.hook_engine.add_room_hook(room_id, named)

    def _wire_handoff(
        self,
        kit: RoomKit,
        agents: Sequence[Channel],
        handler: HandoffHandler,
        cp: ConversationPipeline,
        room_id: str,
    ) -> None:
        """Set up per-agent handoff tools with constrained targets, in
        *room_id*, with this install's handler (RFC §19.7)."""
        agent_map: dict[str, Channel] = {a.channel_id: a for a in agents}
        stage_by_agent: dict[str, PipelineStage] = {s.agent_id: s for s in cp.stages}

        handoffs = []
        for agent in agents:
            stage = stage_by_agent.get(agent.channel_id)
            if stage is None:
                handoffs.append((agent, handoff_entry(agent, handler, None)))
                continue

            reachable_phases: set[str] = set()
            if stage.next:
                reachable_phases.add(stage.next)
            reachable_phases.update(stage.can_return_to)

            targets: list[tuple[str, str | None]] = []
            for s in cp.stages:
                if s.phase in reachable_phases and s.agent_id != agent.channel_id:
                    target_agent = agent_map.get(s.agent_id)
                    desc = getattr(target_agent, "description", None)
                    if desc is None:
                        desc = s.description
                    targets.append((s.agent_id, desc))

            tool = build_handoff_tool(targets)
            handoffs.append((agent, handoff_entry(agent, handler, tool)))
        set_up_handoffs(kit, handoffs, room_id=room_id, owner=self)


def _build_stages(agents: Sequence[Channel | str]) -> list[PipelineStage]:
    """Build pipeline stages from the agent list: each hands off to the next."""
    stages: list[PipelineStage] = []
    for i, agent in enumerate(agents):
        phase = id_of(agent)
        next_phase = id_of(agents[i + 1]) if i + 1 < len(agents) else None
        stages.append(
            PipelineStage(
                phase=phase,
                agent_id=id_of(agent),
                next=next_phase,
                description=getattr(agent, "description", None),
            )
        )
    return stages


def _handler(
    kit: RoomKit,
    agents: Sequence[Channel],
    cp: ConversationPipeline,
    router: ConversationRouter,
    rtv: RealtimeVoiceChannel | None,
) -> HandoffHandler:
    """The handoff handler of one room's install, routing with *router*; on
    a realtime channel *rtv*, the agents its sessions run (RFC §19.5)."""
    handler = HandoffHandler(
        kit=kit,
        router=router,
        phase_map=cp.get_phase_map(),
        allowed_transitions=cp.get_allowed_transitions(),
        known_agents=({a.channel_id for a in agents} if rtv is not None else None),
        event_channel_id=(rtv.channel_id if rtv is not None else None),
    )
    handler.greeting_map = {
        a.channel_id: g for a in agents if (g := getattr(a, "greeting", None)) is not None
    }
    handler.agents = {a.channel_id: a for a in agents}
    return handler
