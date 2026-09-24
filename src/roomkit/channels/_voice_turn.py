"""VoiceChannel mixin — turn detection and text routing."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
        _turn_speech_state: Per session, whether the user is speaking and the
            ``time.monotonic()`` of the last speech onset or end, written from VAD
            events, so a wait for an incomplete turn measures silence, not time.
        _turn_wait_tasks: The pending wait per session, cancelled when replaced.
        _scheduled_tasks: Background tasks that ``close()`` cancels.
        _state_lock: Guards state touched from the audio thread.
    """

    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _turn_speech_state: dict[str, tuple[bool, float]]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _scheduled_tasks: set[asyncio.Task[Any]]
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
    _turn_speech_state: dict[str, tuple[bool, float]]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _scheduled_tasks: set[asyncio.Task[Any]]
    _state_lock: threading.Lock

    # -- cross-mixin methods (annotated as Any to avoid MRO shadowing) --
    _task_done: Any  # VoiceChannel._task_done
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
        logger.debug(
            "Turn %s by %s (confidence %.2f, %s)",
            "complete" if decision.is_complete else "incomplete",
            turn_detector.name,
            decision.confidence,
            decision.reason,
        )

        if decision.is_complete and self._user_speaking_again(session.id):
            # The user resumed while the turn was judged: it goes on, and the new
            # speech joins it (RFC §12). The wait still routes it after silence.
            logger.debug("Turn judged complete, but the user is speaking again: holding it")
            self._arm_turn_wait(session, room_id, context, None)
        elif decision.is_complete:
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
        """Wait for more speech on an incomplete turn, then route it (RFC §12).

        A later evaluation of the same turn replaces the wait: it either completes
        the turn or arms a new one.
        """
        default_ms = self._pipeline_config.turn_incomplete_wait_ms if self._pipeline_config else 0
        wait_ms = suggested_wait_ms if suggested_wait_ms is not None else default_ms
        previous = self._turn_wait_tasks.pop(session.id, None)
        if previous is not None:
            previous.cancel()
        task = asyncio.get_running_loop().create_task(
            self._route_turn_after_wait(session, room_id, context, wait_ms),
            name=f"turn_wait:{session.id}",
        )
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)
        self._turn_wait_tasks[session.id] = task

    async def _route_turn_after_wait(
        self, session: VoiceSession, room_id: str, context: RoomContext, wait_ms: float
    ) -> None:
        wait_s = max(wait_ms, 0) / 1000
        silence_since = time.monotonic()
        while True:
            with self._state_lock:
                speaking, changed_at = self._turn_speech_state.get(session.id, (False, 0.0))
            if speaking:
                # The user is talking: the turn goes on, and silence restarts when they stop.
                await asyncio.sleep(wait_s)
                continue
            silence_since = max(silence_since, changed_at)
            remaining = silence_since + wait_s - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)
        if self._turn_wait_tasks.get(session.id) is asyncio.current_task():
            self._turn_wait_tasks.pop(session.id, None)
        if session.id not in self._session_bindings or session.id not in self._pending_turns:
            return
        logger.info(
            "Turn judged incomplete, then %.0f ms of silence: routing it (long_pause)", wait_ms
        )
        try:
            await self._complete_turn(session, room_id, context, confidence=0.0)
        except Exception:
            logger.exception("Error routing the turn after its wait")

    def _user_speaking_again(self, session_id: str) -> bool:
        """Whether the VAD hears speech that started after the segment being judged."""
        with self._state_lock:
            speaking, _ = self._turn_speech_state.get(session_id, (False, 0.0))
        return speaking

    def _note_turn_speech(self, session_id: str, speaking: bool) -> None:
        """Record a VAD speech onset or end for the turn wait. Safe from the audio thread."""
        with self._state_lock:
            self._turn_speech_state[session_id] = (speaking, time.monotonic())

    def _cancel_turn_wait(self, session_id: str) -> None:
        """Drop the session's wait and speech state (session unbound)."""
        task = self._turn_wait_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        with self._state_lock:
            self._turn_speech_state.pop(session_id, None)

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
