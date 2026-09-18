"""Transcription handling for the Gemini Live provider.

The join of streamed transcription chunks into finals and the duplicate guard
on the finals Gemini re-sends, for both the caller's and the model's speech.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.providers.gemini.realtime_state import _GeminiSessionState, _TranscriptionBuffer
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveTranscriptionMixin(RealtimeVoiceProvider):
    """Transcription chunks in, one final per utterance out.

    Mixed into GeminiLiveProvider. Gemini streams both transcripts in
    pieces and re-sends a finished utterance after its buffer flushed; this
    is where the pieces are joined per (session, role) and the re-send is
    recognised, so the channel sees each final once.
    """

    # Owned by GeminiLiveProvider; declared for typing.
    _sessions: dict[str, _GeminiSessionState]
    _transcription_buffer: _TranscriptionBuffer
    _log_event: Any

    def _clear_transcription_buffers(self, session_id: str) -> None:
        """Remove all transcription buffer entries for a session."""
        self._transcription_buffer.clear_session(session_id)

    def _is_duplicate_final(self, session: VoiceSession, role: str, text: str) -> bool:
        """Whether this final re-emits what was already flushed for this turn.

        Gemini re-sends a finished utterance after the buffer already flushed
        it at a lifecycle boundary (speech end, model turn) — unfiltered,
        every re-emission renders as a duplicate final downstream. The guard
        drops consecutive identical finals per role; it is cleared when new
        speech or a new response genuinely begins, so a user repeating the
        same words in a later turn still comes through.
        """
        state = self._sessions.get(session.id)
        if state is None:
            return False
        if state.last_final_text.get(role) == text:
            logger.info(
                "[Gemini] dropping re-emitted %s final (%d chars, session %s)",
                role,
                len(text),
                session.id,
            )
            return True
        state.last_final_text[role] = text
        return False

    async def _handle_transcription_chunk(
        self, session: VoiceSession, text: str, role: str, finished: bool
    ) -> None:
        """Accumulate transcription chunks and fire callback when complete."""
        state = self._sessions.get(session.id)
        if role == "user" and state is not None and state.awaiting_new_user_utterance:
            state.last_final_text.pop("user", None)
            state.awaiting_new_user_utterance = False
        full_text = self._transcription_buffer.append(session.id, role, text, finished)
        if full_text:
            if self._is_duplicate_final(session, role, full_text):
                return
            # Log the FINAL transcription so we can see what each side
            # actually said in the same stream as tool_call events.
            # Truncate to keep log lines readable; full text still goes
            # into the room transcript via the callback.
            self._log_event(
                session.id,
                "transcription",
                role=role,
                text=full_text[:600] + ("…" if len(full_text) > 600 else ""),
                len=len(full_text),
            )
            await self._fire(
                self._transcription_callbacks,
                session,
                full_text,
                role,
                True,
                label="transcription",
            )
        elif not finished:
            # Send non-final for real-time display in the voice modal
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                role,
                False,
                label="transcription",
            )

    async def _flush_transcription_buffer(self, session: VoiceSession, role: str) -> None:
        """Flush buffered transcription at lifecycle boundaries."""
        full_text = self._transcription_buffer.flush(session.id, role)
        if full_text:
            if self._is_duplicate_final(session, role, full_text):
                return
            logger.debug(
                "Flushing %s transcription buffer (%d chars) for session %s",
                role,
                len(full_text),
                session.id,
            )
            await self._fire(
                self._transcription_callbacks,
                session,
                full_text,
                role,
                True,
                label="transcription",
            )
