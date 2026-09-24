"""VoiceChannel mixin — turn detection and text routing."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.turn.base import TurnEntry

logger = logging.getLogger("roomkit.voice")

# Cap pending audio buffer at ~1MB (≈32s at 16kHz mono 16-bit).
# SmartTurnDetector uses the last 8s (≈256KB) so this is generous.
_MAX_PENDING_AUDIO_BYTES = 1_048_576


@runtime_checkable
class TurnHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceTurnMixin.

    Every attribute listed here is initialized by ``VoiceChannel.__init__``.

    Attributes:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the owning RoomKit framework (None before registration).
        _backend: The voice backend transport (None before configuration).
        _pipeline_config: Audio pipeline configuration (None when no pipeline).
        _session_bindings: Map of session ID to (room_id, binding) pairs.
        _pending_turns: Accumulated turn entries per session, awaiting turn completion.
        _pending_audio: Accumulated raw audio per session for audio-native turn detectors.
        _turn_wait_generation: Per session, bumped by every new speech onset, so a
            wait for an incomplete turn knows on waking whether it still stands.
        _turn_wait_tasks: The pending wait per session, held so it is not collected.
        _state_lock: Guards state touched from the audio thread.
    """

    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _turn_wait_generation: dict[str, int]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _state_lock: threading.Lock


class VoiceTurnMixin:
    """Turn detection and text routing for VoiceChannel.

    Host contract: :class:`TurnHost`.
    """

    # -- attributes provided by VoiceChannel.__init__ (see TurnHost) --
    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _turn_wait_generation: dict[str, int]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _state_lock: threading.Lock

    # -- cross-mixin methods (annotated as Any to avoid MRO shadowing) --
    _pipeline_audio_rate: Any  # VoicePipelineMixin

    async def _evaluate_turn(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        context: RoomContext,
        *,
        audio_bytes: bytes | None = None,
    ) -> None:
        """Evaluate turn completion using the configured TurnDetector."""
        if not self._framework or not self._pipeline_config:
            return
        turn_detector = self._pipeline_config.turn_detector
        if turn_detector is None:
            await self._route_text(session, text, room_id)
            return

        from roomkit.voice.pipeline.turn.base import TurnContext, TurnEntry

        # Accumulate entry
        entries = self._pending_turns.setdefault(session.id, [])
        entries.append(TurnEntry(text=text, role="user"))

        # Accumulate audio for audio-native turn detectors
        if audio_bytes:
            buf = self._pending_audio.setdefault(session.id, bytearray())
            buf.extend(audio_bytes)
            if len(buf) > _MAX_PENDING_AUDIO_BYTES:
                trim = len(buf) - _MAX_PENDING_AUDIO_BYTES
                trim = trim + (trim % 2)  # Align to 2-byte (16-bit PCM) boundary
                del buf[:trim]

        accumulated_audio = bytes(self._pending_audio.get(session.id, b"")) or None
        sample_rate = self._pipeline_audio_rate(session)

        turn_ctx = TurnContext(
            conversation_history=list(entries),
            silence_duration_ms=0.0,
            transcript=text,
            is_final=True,
            session_id=session.id,
            audio_bytes=accumulated_audio,
            audio_sample_rate=sample_rate,
        )
        decision = await asyncio.to_thread(turn_detector.evaluate, turn_ctx)

        if decision.is_complete:
            await self._complete_turn(session, room_id, context, decision.confidence)
        else:
            # Fire ON_TURN_INCOMPLETE hook
            combined_so_far = " ".join(e.text for e in entries)
            try:
                from roomkit.voice.events import TurnIncompleteEvent

                incomplete_event = TurnIncompleteEvent(
                    session=session,
                    text=combined_so_far,
                    confidence=decision.confidence,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_TURN_INCOMPLETE,
                    incomplete_event,
                    context,
                    skip_event_filter=True,
                )
            except Exception:
                logger.exception("Error firing ON_TURN_INCOMPLETE hook")
            self._arm_turn_wait(session, room_id, context, decision.suggested_wait_ms)

    async def _complete_turn(
        self, session: VoiceSession, room_id: str, context: RoomContext, confidence: float
    ) -> None:
        """Route the session's accumulated turn, after its ON_TURN_COMPLETE hook."""
        if not self._framework:
            return
        entries = self._pending_turns.pop(session.id, [])
        self._pending_audio.pop(session.id, None)
        if not entries:
            return
        combined = " ".join(e.text for e in entries)
        try:
            from roomkit.voice.events import TurnCompleteEvent

            event = TurnCompleteEvent(session=session, text=combined, confidence=confidence)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_TURN_COMPLETE,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_TURN_COMPLETE hook")

        await self._route_text(session, combined, room_id)

    def _arm_turn_wait(
        self,
        session: VoiceSession,
        room_id: str,
        context: RoomContext,
        suggested_wait_ms: float | None,
    ) -> None:
        """Wait for more speech on an incomplete turn, then route it (RFC §12)."""
        default_ms = self._pipeline_config.turn_incomplete_wait_ms if self._pipeline_config else 0
        wait_ms = suggested_wait_ms if suggested_wait_ms is not None else default_ms
        with self._state_lock:
            generation = self._turn_wait_generation.get(session.id, 0) + 1
            self._turn_wait_generation[session.id] = generation
        self._turn_wait_tasks[session.id] = asyncio.create_task(
            self._route_turn_after_wait(session, room_id, context, generation, wait_ms),
            name=f"turn_wait:{session.id}",
        )

    async def _route_turn_after_wait(
        self,
        session: VoiceSession,
        room_id: str,
        context: RoomContext,
        generation: int,
        wait_ms: float,
    ) -> None:
        await asyncio.sleep(max(wait_ms, 0) / 1000)
        with self._state_lock:
            if self._turn_wait_generation.get(session.id) != generation:
                return  # speech started meanwhile: the turn goes on
            bound = session.id in self._session_bindings
        if not bound or session.id not in self._pending_turns:
            return
        logger.info(
            "Turn judged incomplete, then %.0f ms of silence: routing it (long_pause)", wait_ms
        )
        await self._complete_turn(session, room_id, context, confidence=0.0)

    def _cancel_turn_wait(self, session_id: str) -> None:
        """Speech started: a pending wait no longer stands. Safe from the audio thread."""
        with self._state_lock:
            if session_id in self._turn_wait_generation:
                self._turn_wait_generation[session_id] += 1

    async def _route_text(self, session: VoiceSession, text: str, room_id: str) -> None:
        """Route transcribed text through the inbound pipeline."""
        if not self._framework:
            return
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent
        from roomkit.telemetry.context import reset_span, set_current_span

        inbound = InboundMessage(
            channel_id=self.channel_id,
            sender_id=session.participant_id,
            content=TextContent(body=text),
            metadata={"voice_session_id": session.id, "source": "voice"},
        )
        # Set voice session span as parent so INBOUND_PIPELINE is a child
        session_span = getattr(self, "_voice_session_spans", {}).get(session.id)
        token = set_current_span(session_span) if session_span else None
        try:
            await self._framework.process_inbound(inbound, room_id=room_id)
        finally:
            if token is not None:
                reset_span(token)
