"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from copy import deepcopy
from typing import Any

from pydantic import SecretStr

from roomkit.core.task_utils import _finish_cleanup
from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.gemini.realtime_config import (
    blocking_tool_names,
    build_live_config,
    debug_enabled,
    enum_value,
    warn_unsupported,
)
from roomkit.providers.gemini.realtime_models import (
    MODELS,
    live_model_profile,
)
from roomkit.providers.gemini.realtime_state import (  # noqa: F401 - tests read these
    _GeminiSessionState,
    _GoAwayError,
    _TranscriptionBuffer,
)
from roomkit.providers.gemini.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.gemini.realtime")

_IDLE_STATUSES = frozenset({"IDLE", "INTERACTION_STATUS_IDLE"})
"""What ``interaction_status`` reads when the request is over. A set rather
than a suffix match: a future ``NOT_IDLE`` would end every response early."""

_KNOWN_INTERACTION_STATUSES = _IDLE_STATUSES | {"IN_PROGRESS", "INTERACTION_STATUS_IN_PROGRESS"}
"""Statuses that prove the server reports its interaction state. ``UNSPECIFIED``
does not: latching on it would retire ``turn_complete`` as an end-of-response
signal while nothing ever reads IDLE, and the session would never hand back."""


def _interaction_status_is_known(status: Any) -> bool:
    """Whether *status* is a state this build recognises."""
    return str(getattr(status, "value", status)).upper() in _KNOWN_INTERACTION_STATUSES


def _interaction_is_idle(status: Any) -> bool:
    """True when the server says the whole interaction is over.

    Accepts the SDK enum or the bare wire string: older SDKs hand back the
    latter, and the provider must not care which it got.
    """
    if status is None:
        return False
    return str(getattr(status, "value", status)).upper() in _IDLE_STATUSES


_MAX_INJECT_TEXT_LENGTH = 32_000
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_gemini_text(text: str) -> str:
    """Sanitize text for safe injection into the Gemini Live API.

    Strips null bytes, control characters (except whitespace),
    unpaired surrogates, and truncates to max length.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    if len(text) > _MAX_INJECT_TEXT_LENGTH:
        text = text[:_MAX_INJECT_TEXT_LENGTH] + "... [truncated]"
    return text


class GeminiLiveProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the Google Gemini Live API.

    Connects to Gemini's live streaming API for bidirectional
    audio conversations with built-in AI.

    Requires the ``google-genai`` package.

    Example:
        provider = GeminiLiveProvider(api_key="...")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = "gemini-3.8-live",
    ) -> None:
        super().__init__()

        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiLiveProvider. "
                "Install with: pip install 'roomkit[realtime-gemini]'"
            ) from exc

        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key

        # Tighter WebSocket keepalive to detect dead connections faster
        # (defaults are 20s interval / 20s timeout — too slow for realtime audio)
        self._client = _genai.Client(
            api_key=self._api_key.get_secret_value(),
            http_options=_types.HttpOptions(
                async_client_args={
                    "ping_interval": 10,
                    "ping_timeout": 10,
                }
            ),
        )
        self._model = model

        # Consolidated per-session state: session_id -> _GeminiSessionState
        self._sessions: dict[str, _GeminiSessionState] = {}

        self._transcription_buffer = _TranscriptionBuffer()

        # Hot-path caches (instance-level to avoid shared mutable class state)
        self._blob_cls: Any = None
        self._mime_cache: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "GeminiLiveProvider"

    @property
    def model_name(self) -> str:
        """The Gemini Live model this provider connects to, end to end."""
        return self._model

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of Gemini Live native-audio voices (fixed set)."""
        return list(_VOICES)

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Gemini Live models."""
        return list(MODELS)

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        # gemini-3.x live models reject send_client_content with WS 1007
        # after the first model turn and offer no documented dynamic
        # system_instruction update. Their session_resumption is also
        # fragile with non-trivial system prompts. Disable mid-session
        # reconfigure for the whole 3.x family so callers route changes
        # through session-start delivery instead. 2.5-era models keep
        # the old behavior.
        return not (self._model.startswith("gemini-3.") or self._model.startswith("gemini-3-"))

    def _get_active_state(self, session: VoiceSession) -> _GeminiSessionState | None:
        """Return session state if the session is connected, else None."""
        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            return None
        return state

    def _build_config(
        self,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
        server_vad: bool = True,
        warned: set[str] | None = None,
    ) -> Any:
        """The LiveConnectConfig for this provider's model.

        Thin over :func:`~roomkit.providers.gemini.realtime_config.build_live_config`,
        kept as a method because ``connect``, ``reconfigure`` and the tests
        address it on the provider.
        """
        return build_live_config(
            self._model,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
            server_vad=server_vad,
            warned=warned,
        )

    def _blocking_tool_names(
        self, tools: list[dict[str, Any]] | None, warned: set[str] | None = None
    ) -> set[str]:
        """Names whose calls the API waits on, for this provider's model."""
        return blocking_tool_names(self._model, tools, warned)

    def _log_event(self, session_id: str, label: str, **fields: Any) -> None:
        """Log a single server event from Gemini Live for diagnostics.

        Called from the receive loop and message handlers. ``label`` is
        a short tag (text_delta, tool_call, transcription, error, …) and
        ``fields`` are the salient attributes to log. Gated on the same
        ``ROOMKIT_GEMINI_DEBUG`` env var as the config dump.
        """
        if not debug_enabled():
            return
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        logger.info(
            "ROOMKIT_GEMINI_DEBUG: <<< %s session=%s %s",
            label,
            session_id[:8],
            rendered,
        )

    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        warned: set[str] = set()
        live_config = self._build_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
            server_vad=server_vad,
            warned=warned,
        )

        ctxmgr = self._client.aio.live.connect(
            model=self._model,
            config=live_config,
        )
        live_session = await ctxmgr.__aenter__()

        state = _GeminiSessionState(
            session=session,
            live_session=live_session,
            ctxmgr=ctxmgr,
            live_config=live_config,
            started_at=time.monotonic(),
            input_sample_rate=input_sample_rate,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            server_vad=server_vad,
            provider_config=deepcopy(provider_config or {}),
            blocking_tool_names=self._blocking_tool_names(tools, warned),
            warned_unsupported=warned,
        )
        self._sessions[session.id] = state

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start receive loop
        state.receive_task = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

        logger.info("Gemini Live session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._sessions.get(session.id)
        if state is None:
            return

        if session.state not in (VoiceSessionState.ACTIVE, VoiceSessionState.CONNECTING):
            return
        # Buffer audio while reconnecting instead of dropping it
        if state.live_session is None or session.state == VoiceSessionState.CONNECTING:
            session.state = VoiceSessionState.CONNECTING
            state.buffer_audio(audio)
            return

        try:
            await state.live_session.send_realtime_input(
                audio=self._make_audio_blob(audio, state.input_sample_rate),
            )
            # Mark that realtime input has been used — send_client_content is
            # no longer safe for this session (interleaving causes 1007 disconnects).
            state.realtime_input_sent = True
            # Successful send — reset suppression so next failure fires callback
            state.error_suppressed = False
        except Exception as exc:
            # Connection lost — the receive loop will handle reconnection.
            # Don't mark ENDED here; just suppress further sends.
            if session.state == VoiceSessionState.ACTIVE:
                session.state = VoiceSessionState.CONNECTING
            # Fire error callback only once per reconnection cycle
            if not state.error_suppressed:
                state.error_suppressed = True
                await self._fire(
                    self._error_callbacks, session, "send_audio_failed", str(exc), label="error"
                )
            return

    async def start_audio_stream(self, session: VoiceSession) -> None:
        """Open the realtime audio input path by sending 20 ms of silence.

        Gemini Live exposes two protocol paths on the same WebSocket —
        ``send_client_content`` for structured text turns and
        ``send_realtime_input`` for streaming audio.  Interleaving them
        after audio has started causes the server to close the socket
        with code 1008/1007 on some preview models.  Sending one frame
        of silence up-front commits the session to the realtime path so
        later :meth:`inject_text` calls stay interleave-safe.

        No-op if the session is not active or the stream is already open.
        """
        state = self._get_active_state(session)
        if state is None or state.realtime_input_sent:
            return
        if state.live_session is None:
            return
        silence = b"\x00" * (state.input_sample_rate // 50)  # 20 ms PCM-16
        await state.live_session.send_realtime_input(
            audio=self._make_audio_blob(silence, state.input_sample_rate)
        )
        state.realtime_input_sent = True

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> VoiceInjectionResult:
        if (state := self._get_active_state(session)) is None:
            return VoiceInjectionResult(
                status="not_sent", reason="voice_not_connected", retryable=True
            )

        # Sanitize to prevent 1007 disconnects from control chars / surrogates.
        text = _sanitize_gemini_text(text)
        if not text.strip():
            logger.debug(
                "inject_text: empty after sanitization, skipping (session %s)",
                session.id,
            )
            return VoiceInjectionResult(status="not_sent", reason="voice_empty_text")

        # Queue while a blocking tool call is outstanding: the API refuses
        # input until its function response comes back. A background call is
        # not one the API waits on, and queueing there would hold the
        # injection back for no reason.
        if state.blocking_call_ids:
            logger.debug(
                "Queuing text injection for session %s (pending tool calls: %d)",
                session.id,
                state.pending_tool_calls,
            )
            state.queued_text_injections.append((text, role, silent))
            return VoiceInjectionResult(status="unknown", reason="voice_provider_queued")

        await self._send_text(state, text, role, silent)
        return VoiceInjectionResult(status="sent")

    async def _send_text(
        self,
        state: _GeminiSessionState,
        text: str,
        role: str,
        silent: bool,
    ) -> None:
        from google.genai import types

        effective_role = role if role in ("user", "model") else "user"
        if effective_role != role:
            logger.debug(
                "inject_text session %s: role %r not supported by Gemini, coerced to %r",
                state.session.id,
                role,
                effective_role,
            )
        logger.debug(
            "inject_text session %s: role=%s (original=%s), silent=%s, "
            "realtime=%s, len=%d, preview=%.200s",
            state.session.id,
            effective_role,
            role,
            silent,
            state.realtime_input_sent,
            len(text),
            text,
        )

        if not state.realtime_input_sent:
            # No audio sent yet — send_client_content is safe and gives full
            # control over role and turn_complete semantics.
            await state.live_session.send_client_content(
                turns=types.Content(
                    role=effective_role,
                    parts=[types.Part(text=text)],
                ),
                turn_complete=not silent,
            )
            return

        # Audio is flowing — must use send_realtime_input to avoid 1007
        # disconnects from interleaving send_client_content with realtime input.
        # Limitations: no role parameter, no turn_complete control.
        if effective_role == "model":
            logger.warning(
                "inject_text session %s: role='model' not supported via "
                "send_realtime_input — sending as user context instead",
                state.session.id,
            )
            text = f"[Assistant previously said] {text}"

        if silent:
            text = f"[Context update, do not respond to this] {text}"
            logger.debug(
                "inject_text session %s: silent mode is best-effort via "
                "send_realtime_input (model may still respond)",
                state.session.id,
            )

        await state.live_session.send_realtime_input(text=text)

    async def inject_image(
        self,
        session: VoiceSession,
        image_data: bytes,
        mime_type: str = "image/png",
        *,
        prompt: str = "",
        silent: bool = False,
    ) -> None:
        if (state := self._get_active_state(session)) is None:
            return

        # Same guard as inject_text: only a blocking call makes the API refuse
        # client_content. Queue the injection and flush after
        # submit_tool_result.
        if state.blocking_call_ids:
            logger.debug(
                "Queuing image injection for session %s (pending tool calls: %d)",
                session.id,
                state.pending_tool_calls,
            )
            state.queued_injections.append((image_data, mime_type, prompt, silent))
            return

        await self._send_image(state, image_data, mime_type, prompt, silent)

    async def _send_image(
        self,
        state: _GeminiSessionState,
        image_data: bytes,
        mime_type: str,
        prompt: str,
        silent: bool,
    ) -> None:
        from google.genai import types

        # Sanitize once, before branching.
        if prompt:
            prompt = _sanitize_gemini_text(prompt)
            if not prompt.strip():
                prompt = ""

        if not state.realtime_input_sent:
            # No audio sent yet — send_client_content is safe.
            parts: list[types.Part] = []
            if prompt:
                parts.append(types.Part(text=prompt))
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data)))
            await state.live_session.send_client_content(
                turns=types.Content(role="user", parts=parts),
                turn_complete=not silent,
            )
            return

        # Audio is flowing — use send_realtime_input to avoid 1007
        # disconnects.  The SDK only accepts one argument per call,
        # so text prompt and media are sent as separate messages.
        if prompt:
            if silent:
                prompt = f"[Context update, do not respond to this] {prompt}"
            await state.live_session.send_realtime_input(text=prompt)
        elif silent:
            # No prompt but silent — send a standalone instruction so the
            # model doesn't react to the image (no turn_complete equivalent
            # on the realtime path).
            await state.live_session.send_realtime_input(
                text="[Context update, do not respond to this image]"
            )

        await state.live_session.send_realtime_input(
            media=types.Blob(mime_type=mime_type, data=image_data),
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        import json

        from google.genai import types

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
        except (json.JSONDecodeError, ValueError):
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
        state.pending_tool_calls = max(0, state.pending_tool_calls - 1)
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

    async def interrupt(self, session: VoiceSession) -> None:
        # Gemini doesn't have a direct cancel; send empty to reset
        if self._get_active_state(session) is None:
            return
        logger.debug("Interrupt requested for Gemini session %s (no-op)", session.id)

    async def send_activity_start(self, session: VoiceSession) -> None:
        """Send ActivityStart to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        from google.genai import types

        await state.live_session.send_realtime_input(
            activity_start=types.ActivityStart(),
        )
        logger.debug("Sent ActivityStart for session %s", session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Send ActivityEnd to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        from google.genai import types

        await state.live_session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )
        logger.debug("Sent ActivityEnd for session %s", session.id)

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        state = self._sessions.pop(session.id, None)
        if state is None:
            session.state = VoiceSessionState.ENDED
            return

        session.state = VoiceSessionState.ENDED
        state.audio_buffer.clear()

        # Cancel receive task
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task

        # Clean up transcription buffers
        self._clear_transcription_buffers(session.id)

        # Record session metrics before cleanup
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        if state.started_at:
            uptime_s = time.monotonic() - state.started_at
            telemetry.record_metric(
                "roomkit.realtime.uptime_s",
                uptime_s,
                unit="s",
                attributes={"provider": "gemini", "session_id": session.id},
            )
        telemetry.record_metric(
            "roomkit.realtime.turn_count",
            float(state.turn_count),
            attributes={"provider": "gemini", "session_id": session.id},
        )
        if state.tool_result_bytes:
            telemetry.record_metric(
                "roomkit.realtime.tool_result_bytes",
                float(state.tool_result_bytes),
                attributes={"provider": "gemini", "session_id": session.id},
            )

        # Close live session via context manager exit
        if state.ctxmgr is not None:
            with contextlib.suppress(Exception):
                await state.ctxmgr.__aexit__(None, None, None)
        elif state.live_session is not None:
            with contextlib.suppress(Exception):
                await state.live_session.close()

        logger.info(
            "Gemini session %s disconnected: received=%d audio chunks",
            session.id,
            state.audio_chunk_count,
        )
        session.state = VoiceSessionState.ENDED

    async def reconfigure(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Reconfigure a session by rebuilding config and reconnecting.

        Uses Gemini's session resumption to preserve conversation
        history while switching system prompt, voice, and tools.

        Reconfigure semantics: parameters left at ``None`` are
        preserved from the session's most recent config. ``_build_config``
        treats ``None`` as "absent" (it omits the field from the
        resulting LiveConnectConfig), so without this preservation a
        partial update like ``reconfigure(system_prompt=new)`` would
        wipe the existing tools and voice. Passing an empty list /
        empty string explicitly does still clear the field.
        """
        import contextlib

        state = self._sessions.get(session.id)
        if state is None:
            return

        if state.provider_config.get("preserve_context"):
            raise ValueError("Reconfiguration is unavailable while preserving Gemini context")

        # Discard stale queued injections from the old configuration
        state.queued_text_injections.clear()
        state.queued_injections.clear()

        # Preserve unspecified fields from the previous config so a
        # partial reconfigure (e.g. system_prompt-only) doesn't wipe
        # tools/voice/temperature. ``_build_config`` treats ``None``
        # as "absent" and would otherwise produce a config with no
        # tools at all.
        effective_prompt = system_prompt if system_prompt is not None else state.system_prompt
        effective_voice = voice if voice is not None else state.voice
        effective_tools = tools if tools is not None else state.tools
        effective_temperature = temperature if temperature is not None else state.temperature

        effective_provider_config = deepcopy(state.provider_config)
        if provider_config is not None:
            effective_provider_config.update(deepcopy(provider_config))

        new_config = self._build_config(
            system_prompt=effective_prompt,
            voice=effective_voice,
            tools=effective_tools,
            temperature=effective_temperature,
            provider_config=effective_provider_config,
            server_vad=state.server_vad,
            warned=state.warned_unsupported,
        )

        # Remember effective values so the next partial reconfigure
        # preserves them.
        state.system_prompt = effective_prompt
        state.voice = effective_voice
        state.tools = effective_tools
        state.temperature = effective_temperature
        # The new declarations decide what blocks from here on. Calls
        # outstanding from the old set keep their ids in blocking_call_ids
        # until their results come back, so the guard stays honest across
        # the change.
        state.blocking_tool_names = self._blocking_tool_names(
            effective_tools, state.warned_unsupported
        )
        state.live_config = new_config
        state.provider_config = effective_provider_config
        logger.info(
            "Reconfiguring Gemini session %s (voice=%s)",
            session.id,
            voice,
        )

        # Cancel the old receive task BEFORE reconnecting to prevent it
        # from detecting the disconnection and triggering a second
        # auto-reconnect (double-reconnect bug).
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task
            state.receive_task = None

        await self._reconnect(session)

        # Start a fresh receive loop for the new connection.
        state.receive_task = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            state = self._sessions.get(session_id)
            if state:
                await self.disconnect(state.session)

    # -- Hot-path helpers (avoid per-call imports and string formatting) --

    def _make_audio_blob(self, data: bytes, sample_rate: int) -> Any:
        """Create a Blob without per-call import or string formatting."""
        if self._blob_cls is None:
            from google.genai import types

            self._blob_cls = types.Blob
        mime = self._mime_cache.get(sample_rate)
        if mime is None:
            mime = f"audio/pcm;rate={sample_rate}"
            self._mime_cache[sample_rate] = mime
        return self._blob_cls(data=data, mime_type=mime)

    # -- Receive loop --

    _MAX_RECONNECTS = 5

    # WebSocket close codes that signal a *permanent* setup/policy problem,
    # not a transient drop. Reconnecting with the same config just reproduces
    # the failure (and stalls the user ~10s through 5 back-off retries), so we
    # fail fast and surface the exact code to the embedder instead.
    #   1007 — invalid argument (e.g. a tool schema Gemini Live won't accept)
    #   1008 — policy violation
    #   1011 — internal error, used by Gemini for quota/billing exhaustion
    _NON_RETRYABLE_CLOSE_CODES = frozenset({1007, 1008, 1011})

    @property
    def supports_context_preservation(self) -> bool:
        """Strict sessions disable compression and stop before reconnection."""
        return True

    async def _end_preserved_context(
        self, session: VoiceSession, state: _GeminiSessionState
    ) -> None:
        state.audio_buffer.clear()
        state.queued_text_injections.clear()
        state.queued_injections.clear()
        session.state = VoiceSessionState.ENDED
        await self._fire(
            self._error_callbacks,
            session,
            "context_preservation_ended",
            "Voice session ended because its full instruction context could not be "
            "preserved across a connection change. Start a new session; pending "
            "operations have not been replayed.",
            label="error",
        )

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from Gemini Live API.

        If the connection drops mid-session, the loop will attempt to
        reconnect up to ``_MAX_RECONNECTS`` times with exponential back-off.
        """
        reconnect_count = 0

        while True:
            state = self._sessions.get(session.id)
            if state is None:
                return

            # If the session was closed by the user, stop the loop
            if session.state == VoiceSessionState.ENDED:
                return

            # Handle reconnection if needed
            if state.live_session is None:
                if state.provider_config.get("preserve_context"):
                    await self._end_preserved_context(session, state)
                    return
                session.state = VoiceSessionState.CONNECTING
                reconnect_count += 1
                if reconnect_count > self._MAX_RECONNECTS:
                    logger.error(
                        "Gemini Live session %s: connection lost, max reconnects (%d) reached",
                        session.id,
                        self._MAX_RECONNECTS,
                    )
                    state.audio_buffer.clear()
                    session.state = VoiceSessionState.ENDED
                    await self._fire(
                        self._error_callbacks,
                        session,
                        "max_reconnects",
                        f"Connection lost after {self._MAX_RECONNECTS} reconnect attempts",
                        label="error",
                    )
                    return

                delay = min(0.5 * (2 ** (reconnect_count - 1)), 4.0)
                logger.warning(
                    "Gemini Live connection lost for session %s (attempt %d/%d), "
                    "reconnecting in %.1fs…",
                    session.id,
                    reconnect_count,
                    self._MAX_RECONNECTS,
                    delay,
                )
                await asyncio.sleep(delay)

                try:
                    await self._reconnect(session)
                except Exception:
                    logger.exception("Reconnect failed for session %s", session.id)
                    continue

            # Process messages from the current session
            try:
                # local ref to live_session
                live_session = state.live_session
                if live_session is None:
                    continue

                async for response in live_session.receive():
                    reconnect_count = 0
                    await self._handle_server_response(session, response)

            except asyncio.CancelledError:
                raise
            except _GoAwayError:
                # Server warned it's about to disconnect — proactive reconnect.
                # This does NOT count against the reconnect limit.
                logger.info("Proactive reconnect (GoAway) for session %s", session.id)
                state.live_session = None
                reconnect_count = 0
            except Exception as exc:
                if session.state == VoiceSessionState.ENDED:
                    return  # state may be mutated by close() during await

                uptime = time.monotonic() - state.started_at if state.started_at else 0.0
                close_code = getattr(exc, "code", None)
                # Extract detailed error info from Gemini APIError
                response_json = getattr(exc, "response_json", None)
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "Gemini session %s disconnected — "
                    "uptime=%.1fs, turns=%d, tool_result_bytes=%d, "
                    "audio_chunks=%d, close_code=%s, error=%s: %s, "
                    "status_code=%s, response_json=%s, pending_tools=%d",
                    session.id,
                    uptime,
                    state.turn_count,
                    state.tool_result_bytes,
                    state.audio_chunk_count,
                    close_code,
                    type(exc).__name__,
                    exc,
                    status_code,
                    response_json,
                    state.pending_tool_calls,
                )
                state.live_session = None

                # Permanent failures (bad config, quota) won't recover by
                # reconnecting — end the session now and surface the exact
                # close code so the embedder can show the user a precise reason
                # instead of a silent disconnect after 5 useless retries.
                if close_code in self._NON_RETRYABLE_CLOSE_CODES:
                    state.audio_buffer.clear()
                    session.state = VoiceSessionState.ENDED
                    await self._fire(
                        self._error_callbacks,
                        session,
                        f"ws_{close_code}",
                        str(exc),
                        label="error",
                    )
                    return

                # Suppress duplicate send_audio_failed errors during reconnect
                state.error_suppressed = True

    def _clear_transcription_buffers(self, session_id: str) -> None:
        """Remove all transcription buffer entries for a session."""
        self._transcription_buffer.clear_session(session_id)

    async def _reconnect(self, session: VoiceSession) -> None:
        """Reconnect to Gemini Live using the stored config."""
        import contextlib

        state = self._sessions.get(session.id)
        if state is None or session.state == VoiceSessionState.ENDED:
            raise RuntimeError("No session state for reconnection")
        if state.provider_config.get("preserve_context"):
            await self._end_preserved_context(session, state)
            raise RuntimeError("Cannot resume Gemini with uncertain context")

        # Suppress audio sends during reconnection
        session.state = VoiceSessionState.CONNECTING

        # Tear down old connection
        old_ctxmgr = state.ctxmgr
        state.ctxmgr = None
        state.live_session = None
        if old_ctxmgr:
            with contextlib.suppress(Exception):
                await old_ctxmgr.__aexit__(None, None, None)

        # Clear stale transcription buffers
        self._clear_transcription_buffers(session.id)

        live_config = state.live_config
        if not live_config:
            raise RuntimeError("No stored config for reconnection")

        # Use stored resumption handle to preserve conversation context
        resumption_handle = state.resumption_handle
        if resumption_handle and live_config.session_resumption is not None:
            live_config.session_resumption.handle = resumption_handle
            logger.info("Reconnecting session %s with resumption handle", session.id)
        elif live_config.session_resumption is not None:
            # No handle available — reset to None for a fresh session
            live_config.session_resumption.handle = None

        try:
            ctxmgr = self._client.aio.live.connect(
                model=self._model,
                config=live_config,
            )
            live_session = await ctxmgr.__aenter__()
        except Exception as exc:
            # Fallback: if reconnection with handle failed, try one fresh connect
            if resumption_handle and live_config.session_resumption is not None:
                logger.warning(
                    "Gemini reconnection with handle failed for %s, trying fresh: %s",
                    session.id,
                    exc,
                )
                state.resumption_handle = None
                live_config.session_resumption.handle = None
                ctxmgr = self._client.aio.live.connect(
                    model=self._model,
                    config=live_config,
                )
                live_session = await ctxmgr.__aenter__()
            else:
                raise

        # Keep ownership local until replay succeeds. A concurrent disconnect
        # cannot close this context twice or publish a late ACTIVE session.
        def check_owner() -> None:
            if (
                self._sessions.get(session.id) is not state
                or session.state == VoiceSessionState.ENDED
            ):
                raise asyncio.CancelledError("Session ended during reconnection")

        try:
            check_owner()
            while (chunk := state.pop_audio()) is not None:
                await live_session.send_realtime_input(
                    audio=self._make_audio_blob(chunk, state.input_sample_rate),
                )
                check_owner()
                state.realtime_input_sent = True
        except BaseException:
            await _finish_cleanup(ctxmgr.__aexit__(None, None, None))
            raise

        # No await between observing an empty buffer and activating: new
        # microphone frames cannot overtake buffered speech.
        state.ctxmgr = ctxmgr
        state.live_session = live_session
        state.response_started = False
        state.response_ended_by_interrupt = False
        session.state = VoiceSessionState.ACTIVE

        # Re-enable error callbacks for the next reconnection cycle
        state.error_suppressed = False

        await self._release_calls_lost_with_the_connection(state)

        logger.info("Gemini Live session %s reconnected", session.id)

    async def _release_calls_lost_with_the_connection(self, state: _GeminiSessionState) -> None:
        """Forget the tool calls the old socket was waiting on.

        Call ids are connection-scoped: the new socket never issued them and
        will not read their results. A blocking one left in the books held
        every injection queued behind it until the application's handler
        finished work the model had already lost, and its result then went
        out for an id the server did not know.
        """
        orphaned = sorted(state.blocking_call_ids)
        if orphaned:
            logger.info(
                "[Gemini] %d blocking tool call(s) did not survive the reconnect (session %s)",
                len(orphaned),
                state.session.id,
            )
            state.cancelled_call_ids.update(orphaned)
            state.blocking_call_ids.clear()
        state.pending_tool_calls = 0
        if orphaned:
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

    # Ordered dispatch table for server response handling.  Each entry is
    # (response_attribute, handler_method).  Order matters: go_away is
    # processed LAST so all data in the message is handled first.
    _RESPONSE_HANDLERS: list[tuple[str, str]] = [
        ("session_resumption_update", "_on_session_resumption"),
        ("voice_activity", "_on_voice_activity"),
        ("server_content", "_on_server_content"),
        ("data", "_on_audio_data"),
        ("tool_call", "_on_tool_call"),
        ("tool_call_cancellation", "_on_tool_call_cancellation"),
        ("usage_metadata", "_on_usage_metadata"),
        ("go_away", "_on_go_away"),
    ]

    async def _handle_server_response(self, session: VoiceSession, response: Any) -> None:
        """Map Gemini Live responses to callbacks."""
        state = self._sessions.get(session.id)
        if state is None:
            return

        self._log_server_message(session, response)

        for attr, method in self._RESPONSE_HANDLERS:
            value = getattr(response, attr, None)
            if value:
                await getattr(self, method)(session, state, value)

    def _log_server_message(self, session: VoiceSession, response: Any) -> None:
        """Build a compact debug log line summarising the server message."""
        parts: list[str] = []
        if getattr(response, "data", None):
            parts.append(f"audio={len(response.data)}B")
        sc = getattr(response, "server_content", None)
        if sc:
            if getattr(sc, "model_turn", None):
                parts.append("model_turn")
            if getattr(sc, "turn_complete", None):
                parts.append("turn_complete")
            if getattr(sc, "interaction_status", None):
                parts.append(f"interaction={sc.interaction_status}")
            if getattr(sc, "interrupted", None):
                parts.append("interrupted")
            if getattr(sc, "input_transcription", None):
                parts.append(f"input_tx={sc.input_transcription.text!r}")
            if getattr(sc, "output_transcription", None):
                parts.append(f"output_tx={sc.output_transcription.text!r}")
        if getattr(response, "tool_call", None):
            parts.append("tool_call")
        va = getattr(response, "voice_activity", None)
        if va:
            parts.append(f"vad={getattr(va, 'voice_activity_type', '?')}")
        if getattr(response, "go_away", None):
            parts.append("go_away")
        if getattr(response, "session_resumption_update", None):
            parts.append("resumption_update")
        um = getattr(response, "usage_metadata", None)
        if um:
            parts.append(
                f"usage(prompt={getattr(um, 'prompt_token_count', '?')}"
                f",response={getattr(um, 'response_token_count', '?')}"
                f",total={getattr(um, 'total_token_count', '?')})"
            )
        if not parts:
            parts.append(f"unknown_keys={[k for k in dir(response) if not k.startswith('_')]}")
        logger.debug("[Gemini] recv: %s (session %s)", ", ".join(parts), session.id)

    async def _on_session_resumption(
        self, session: VoiceSession, state: _GeminiSessionState, update: Any
    ) -> None:
        if not update.resumable or state.provider_config.get("preserve_context"):
            state.resumption_handle = None
        elif update.new_handle:
            state.resumption_handle = update.new_handle
            logger.debug(
                "Session resumption handle updated for %s (resumable=%s)",
                session.id,
                update.resumable,
            )

    async def _on_voice_activity(
        self, session: VoiceSession, state: _GeminiSessionState, va: Any
    ) -> None:
        vtype = getattr(va, "voice_activity_type", None)
        if not vtype:
            return
        if vtype == "ACTIVITY_START":
            logger.info("[VAD] speech_start (session %s)", session.id)
            state.user_speech_active = True
            # New utterance: a repeat of the previous words is now legitimate.
            state.last_final_text.pop("user", None)
            state.awaiting_new_user_utterance = False
            await self._fire(self._speech_start_callbacks, session, label="speech_start")
        elif vtype == "ACTIVITY_END":
            logger.info("[VAD] speech_end (session %s)", session.id)
            state.user_speech_active = False
            await self._flush_transcription_buffer(session, "user")
            await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_server_content(
        self, session: VoiceSession, state: _GeminiSessionState, content: Any
    ) -> None:
        # Input transcription (user speech-to-text)
        tr = getattr(content, "input_transcription", None)
        if tr and tr.text:
            await self._handle_transcription_chunk(session, tr.text, "user", bool(tr.finished))

        out_tr = getattr(content, "output_transcription", None)
        model_turn = getattr(content, "model_turn", None)

        # The user's utterance is over the moment the model starts replying —
        # flush it BEFORE any assistant transcription goes out. One server
        # message can carry both the reply's first transcript chunk and
        # model_turn; emitting that chunk ahead of the user final inverts the
        # conversation downstream, where the late final reads as *new* user
        # speech (phantom barge-in, duplicated user entry).
        if not state.response_started and ((out_tr and out_tr.text) or model_turn):
            await self._flush_transcription_buffer(session, "user")

        # A response may first appear as output_transcription, as model_turn,
        # or as both in one message. Lift the assistant duplicate guard before
        # consuming that first chunk; doing it in the model_turn block below is
        # too late for the coalesced form and would discard a valid repeated
        # reply.
        if (
            not state.response_started
            and not state.assistant_response_observed
            and ((out_tr and out_tr.text) or model_turn)
        ):
            state.last_final_text.pop("assistant", None)
            state.assistant_response_observed = True

        # Output transcription (model speech-to-text)
        if out_tr and out_tr.text:
            await self._handle_transcription_chunk(
                session, out_tr.text, "assistant", bool(out_tr.finished)
            )

        # Model started generating
        if model_turn and not state.response_started:
            state.response_started = True
            state.response_ended_by_interrupt = False
            state.audio_chunk_count = 0
            logger.info("[Gemini] response_start (session %s)", session.id)
            self._log_event(session.id, "response_start", turn=state.turn_count)
            await self._fire(self._response_start_callbacks, session, label="response_start")

        # Interrupted — user barged in while model was speaking
        if getattr(content, "interrupted", None):
            logger.info("[Gemini] INTERRUPTED — AI cut off by barge-in (session %s)", session.id)
            await self._flush_transcription_buffer(session, "assistant")
            # Fire speech_start ONLY if ACTIVITY_START wasn't already
            # received — Gemini doesn't always send voice_activity before
            # interrupted, so this may be the only trigger.
            if not state.user_speech_active:
                state.user_speech_active = True
                state.last_final_text.pop("user", None)
                state.awaiting_new_user_utterance = False
                await self._fire(self._speech_start_callbacks, session, label="speech_start")
            if state.response_started:
                state.response_started = False
                state.response_ended_by_interrupt = True
                await self._fire(self._response_end_callbacks, session, label="response_end")
            state.assistant_response_observed = False

        # Turn complete, and separately, interaction complete.
        #
        # Through 3.1 the two were the same event: one request, one spoken
        # turn, ``turn_complete`` at the end of it. From 3.8 the model
        # reasons and runs tools in the background while it keeps talking, so
        # it produces several turns per request and ``turn_complete`` no
        # longer means it has finished. ``interaction_status`` does: it reads
        # IN_PROGRESS while work remains and IDLE when the request is done.
        #
        # Firing ``response_end`` on every ``turn_complete`` would tell the
        # channel the reply is over while the model is still speaking, which
        # desynchronises the interruption handler and the bridge.
        status = getattr(content, "interaction_status", None)
        if status is not None and _interaction_status_is_known(status):
            state.reports_interaction_status = True
        interaction_done = _interaction_is_idle(status)

        turn_complete = bool(getattr(content, "turn_complete", None))
        if turn_complete:
            state.turn_count += 1
            logger.info(
                "[Gemini] turn_complete (session %s, %d audio chunks, status=%s)",
                session.id,
                state.audio_chunk_count,
                status,
            )
            self._log_event(
                session.id,
                "turn_complete",
                turn=state.turn_count,
                audio_chunks=state.audio_chunk_count,
                pending_tool_calls=state.pending_tool_calls,
                interaction_status=str(status) if status is not None else None,
            )

        if turn_complete or interaction_done:
            await self._flush_transcription_buffer(session, "user")
            await self._flush_transcription_buffer(session, "assistant")

        # Where the server reports its state, only IDLE closes the response.
        # Where it does not, ``turn_complete`` is the only signal there is and
        # keeps its old meaning, so 2.0 Flash Live and 2.5 native audio still
        # hand back control.
        if interaction_done or (turn_complete and not state.reports_interaction_status):
            state.response_started = False
            state.user_speech_active = False
            state.awaiting_new_user_utterance = True
            state.assistant_response_observed = False
            if state.response_ended_by_interrupt:
                # The barge-in above already ended this response. The server
                # still closes the interrupted request with turn_complete (and
                # IDLE from 3.8), and ending it again ran the channel's flush
                # and end-of-response signalling twice per interruption.
                state.response_ended_by_interrupt = False
            else:
                await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _on_audio_data(
        self, session: VoiceSession, state: _GeminiSessionState, data: bytes
    ) -> None:
        state.audio_chunk_count += 1
        if state.audio_chunk_count % 50 == 1:
            logger.debug(
                "[Gemini] audio chunk #%d (%d bytes) for session %s",
                state.audio_chunk_count,
                len(data),
                session.id,
            )
        await self._fire(self._audio_callbacks, session, data, label="audio")

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
            state.pending_tool_calls += 1
            if fc.name in state.blocking_tool_names and fc.id:
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

    async def _on_usage_metadata(
        self, session: VoiceSession, state: _GeminiSessionState, meta: Any
    ) -> None:
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
        response_tokens = getattr(meta, "response_token_count", 0) or 0
        total_tokens = getattr(meta, "total_token_count", 0) or 0
        logger.debug(
            "[Gemini] usage: prompt=%d response=%d total=%d (session %s)",
            prompt_tokens,
            response_tokens,
            total_tokens,
            session.id,
        )
        # Only log the "real" usage tick — when the model evaluates a
        # new prompt round (prompt_tokens > 0). Gemini emits one usage
        # event per audio chunk during a response, ALL with
        # prompt_tokens=0 — those would flood the diagnostic log
        # without telling us anything useful.
        if prompt_tokens:
            self._log_event(
                session.id,
                "usage",
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
            )
        self._record_usage(session, prompt_tokens, response_tokens)

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
            state.pending_tool_calls = max(0, state.pending_tool_calls - 1)
            state.blocking_call_ids.discard(call_id)
            state.cancelled_call_ids.add(call_id)
        await self._fire(
            self._tool_call_cancelled_callbacks, session, ids, label="tool_call_cancelled"
        )
        if not state.blocking_call_ids:
            await self._flush_queued_injections(state)

    async def _on_go_away(
        self, session: VoiceSession, state: _GeminiSessionState, go_away: Any
    ) -> None:
        time_left = getattr(go_away, "time_left", "unknown")
        logger.warning(
            "Gemini GoAway received for session %s (time_left=%s)",
            session.id,
            time_left,
        )
        raise _GoAwayError()

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
