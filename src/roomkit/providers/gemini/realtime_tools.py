"""Tool-call handling for the Gemini Live provider.

The model's function calls, the results the application submits, and the
state of the calls the API is waiting on: which ones block input, what got
queued behind them, and how a cancelled or orphaned call is released. Kept
apart from the server-message dispatch because it is the one piece of the
provider with a state machine of its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.providers.gemini.realtime_config import enum_value, genai_types, warn_unsupported
from roomkit.providers.gemini.realtime_models import live_model_profile
from roomkit.providers.gemini.realtime_state import _GeminiSessionState
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveToolsMixin(RealtimeVoiceProvider):
    """Tool calls and the bookkeeping of the ones the model waits on.

    Mixed into ``GeminiLiveProvider``, which owns the sessions and the model
    id. ``pending_call_ids`` names every call the current connection issued
    and has not released. From 3.8 a tool runs in the background by default
    and a call the model does wait on (a BLOCKING declaration, or any call on
    the pre-3.8 family) closes the input channel: ``blocking_call_ids`` says
    which, and the queued injections wait behind it. Three things release a
    call: the result, a server-side cancellation, or the loss of the
    connection that issued the id.
    """

    # Owned by GeminiLiveProvider / its other mixins; declared for typing.
    _model: str
    _get_active_state: Callable[[VoiceSession], _GeminiSessionState | None]
    _log_event: Callable[..., None]
    _send_text: Callable[[_GeminiSessionState, str, str, bool], Awaitable[None]]
    _send_image: Callable[[_GeminiSessionState, bytes, str, str, bool], Awaitable[None]]
    _flush_transcription_buffer: Callable[[VoiceSession, str], Awaitable[None]]

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        types = genai_types()

        if (state := self._get_active_state(session)) is None:
            raise RuntimeError("Cannot deliver tool result without an active Gemini connection")

        # Track tool result bytes for debugging
        state.tool_result_bytes += len(result)

        # Diagnostic: log every tool result we send back to Gemini so
        # the request → response → result cycle is visible end-to-end.
        # Body is truncated to 800 chars in the log; the full thing is
        # still sent to Gemini.
        self._log_event(
            session.id,
            "submit_tool_result",
            call_id=call_id,
            len=len(result),
            preview=(result[:800] + ("…" if len(result) > 800 else "")),
        )

        if call_id in state.cancelled_call_ids:
            # The server discarded this call (tool_call_cancellation) or the
            # connection it belonged to is gone: it will not read the result,
            # and a FunctionResponse for an id it does not know is an error
            # the application never asked for.
            state.cancelled_call_ids.discard(call_id)
            logger.info(
                "[Gemini] dropping the result of cancelled tool call %s (session %s)",
                call_id,
                session.id,
            )
            return

        if len(result) > 16384:
            logger.warning(
                "Large tool result (%d chars) for call %s may cause Gemini to "
                "disconnect or silently fail (session %s)",
                len(result),
                call_id,
                session.id,
            )

        try:
            parsed = json.loads(result)
            result_dict = parsed if isinstance(parsed, dict) else {"result": parsed}
        except ValueError:  # json.JSONDecodeError is one
            result_dict = {"result": result}

        # A background call returns while the model is mid-sentence, so the
        # response has to say when to use it. WHEN_IDLE waits for the end of
        # what is being said, which is what a voice agent wants by default;
        # INTERRUPT cuts in, SILENT files it into context without a word. A
        # blocking call needs none of this: the model is already waiting, and
        # leaving the field unset keeps the pre-3.8 wire byte for byte.
        response_kwargs: dict[str, Any] = {
            "id": call_id,
            "name": "",  # Gemini uses ID-based matching
            "response": result_dict,
        }
        # Only when the caller asks. A default here looked harmless and was
        # not: gemini-3.8-live-extended-thinking closes the session with
        # `1007 Function response scheduling is not supported for this model`,
        # and the models that do take it already deliver a background result
        # sensibly on their own. Nothing to gain, a session to lose.
        was_blocking = call_id in state.blocking_call_ids
        scheduling = state.provider_config.get("tool_response_scheduling")
        if scheduling and not was_blocking:
            if live_model_profile(self._model).response_scheduling:
                response_kwargs["scheduling"] = enum_value(
                    types.FunctionResponseScheduling, scheduling, "tool_response_scheduling"
                )
            else:
                warn_unsupported(self._model, "tool_response_scheduling", state.warned_unsupported)

        await state.live_session.send_tool_response(
            function_responses=[types.FunctionResponse(**response_kwargs)],
        )

        # Release the call and flush what its blocking waited on.
        state.pending_call_ids.discard(call_id)
        state.blocking_call_ids.discard(call_id)
        if not state.blocking_call_ids:
            await self._flush_queued_injections(state)

    async def _flush_queued_injections(self, state: _GeminiSessionState) -> None:
        """Send what the blocking calls held back, text first, then images.

        Called once nothing blocks any more: a result came back, the server
        cancelled the call, or the connection that owned it is gone.
        """
        if state.queued_text_injections:
            text_injections = state.queued_text_injections[:]
            state.queued_text_injections.clear()
            for text, role, silent in text_injections:
                logger.debug(
                    "Flushing queued text injection for session %s (len=%d)",
                    state.session.id,
                    len(text),
                )
                await self._send_text(state, text, role, silent)
        if state.queued_injections:
            injections = state.queued_injections[:]
            state.queued_injections.clear()
            for image_data, mime_type, prompt, silent in injections:
                logger.debug(
                    "Flushing queued image injection for session %s (mime=%s, size=%d)",
                    state.session.id,
                    mime_type,
                    len(image_data),
                )
                await self._send_image(state, image_data, mime_type, prompt, silent)

    async def _release_calls_lost_with_the_connection(self, state: _GeminiSessionState) -> None:
        """Forget every tool call the old socket issued.

        Call ids are connection-scoped: the new socket never issued them and
        will not read their results, blocking or not. Left in the books, a
        blocking one held every injection queued behind it until the
        application's handler finished work the model had already lost, and
        the result of any of them then went out for an id the server did not
        know. Counting the background calls instead of naming them let
        exactly that happen to them.
        """
        orphaned = sorted(state.pending_call_ids)
        state.pending_call_ids.clear()
        state.blocking_call_ids.clear()
        if orphaned:
            logger.info(
                "[Gemini] %d tool call(s) did not survive the reconnect (session %s)",
                len(orphaned),
                state.session.id,
            )
            state.cancelled_call_ids.update(orphaned)
            # The application is still working for the old socket. Same fact
            # as a server cancellation: the model will not read the result.
            await self._fire(
                self._tool_call_cancelled_callbacks,
                state.session,
                orphaned,
                label="tool_call_cancelled",
            )
        try:
            await self._flush_queued_injections(state)
        except Exception:
            # The reconnect stands; what could not be delivered is logged.
            logger.warning(
                "Failed to flush queued injections after reconnecting session %s",
                state.session.id,
                exc_info=True,
            )

    async def _on_tool_call(
        self, session: VoiceSession, state: _GeminiSessionState, tool_call: Any
    ) -> None:
        # A tool call is the model acting on the user's utterance — the
        # utterance is over. Flush its final before emitting the call, for the
        # same reason as the output-transcription flush: consumers must see
        # the user final ahead of everything the model does in answer to it
        # (a late final reads as new user speech downstream).
        await self._flush_transcription_buffer(session, "user")
        for fc in tool_call.function_calls:
            if fc.id:
                state.pending_call_ids.add(fc.id)
                if fc.name in state.blocking_tool_names:
                    state.blocking_call_ids.add(fc.id)
            args_dict = dict(fc.args) if fc.args else {}
            self._log_event(
                session.id,
                "function_call",
                name=fc.name,
                id=fc.id,
                args=args_dict,
            )
            await self._fire(
                self._tool_call_callbacks,
                session,
                fc.id,
                fc.name,
                args_dict,
                label="tool_call",
            )

    async def _on_tool_call_cancellation(
        self, session: VoiceSession, state: _GeminiSessionState, cancellation: Any
    ) -> None:
        """Release the calls the server discarded.

        Sent when the user interrupts while calls are outstanding: the server
        will not read their results, and a blocking one no longer holds the
        input channel. Left in the books, the id kept every injection queued
        until the application's handler finished work the model had already
        abandoned, and if the stale FunctionResponse then failed to send,
        nothing else ever cleared it. The application hears of it through
        ``on_tool_call_cancelled`` and stops the handler; a result that still
        arrives is dropped.
        """
        ids = [call_id for call_id in (getattr(cancellation, "ids", None) or []) if call_id]
        if not ids:
            return
        logger.info("[Gemini] server cancelled tool call(s) %s (session %s)", ids, session.id)
        self._log_event(session.id, "tool_call_cancellation", ids=ids)
        for call_id in ids:
            state.pending_call_ids.discard(call_id)
            state.blocking_call_ids.discard(call_id)
            state.cancelled_call_ids.add(call_id)
        await self._fire(
            self._tool_call_cancelled_callbacks, session, ids, label="tool_call_cancelled"
        )
        if not state.blocking_call_ids:
            await self._flush_queued_injections(state)
