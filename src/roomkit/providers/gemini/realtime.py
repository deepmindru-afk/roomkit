"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
)
from roomkit.providers.gemini.realtime_handlers import GeminiLiveEventHandlersMixin
from roomkit.providers.gemini.realtime_input import GeminiLiveInputMixin
from roomkit.providers.gemini.realtime_models import (
    MODELS,
)
from roomkit.providers.gemini.realtime_state import (  # noqa: F401 - tests read these
    _GeminiSessionState,
    _GoAwayError,
    _TranscriptionBuffer,
)
from roomkit.providers.gemini.realtime_tools import GeminiLiveToolsMixin
from roomkit.providers.gemini.realtime_transcription import GeminiLiveTranscriptionMixin
from roomkit.providers.gemini.voices import VOICES as _VOICES
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveProvider(
    GeminiLiveInputMixin,
    GeminiLiveToolsMixin,
    GeminiLiveEventHandlersMixin,
    GeminiLiveTranscriptionMixin,
    RealtimeVoiceProvider,
):
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

        ctxmgr, live_session = await self._open_live_session(live_config)

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

        self._start_receive_loop(state)

        logger.info("Gemini Live session connected: %s", session.id)

    async def _open_live_session(self, live_config: Any) -> tuple[Any, Any]:
        """Open one Live connection for *live_config*: the context manager and its session.

        The one place the socket is opened, for ``connect`` and for both
        attempts of ``_reconnect``; the caller owns the closing.
        """
        ctxmgr = self._client.aio.live.connect(model=self._model, config=live_config)
        live_session = await ctxmgr.__aenter__()
        return ctxmgr, live_session

    def _start_receive_loop(self, state: _GeminiSessionState) -> None:
        """Start the receive loop of *state*'s session on its own task."""
        session = state.session
        state.receive_task = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

    async def disconnect(self, session: VoiceSession) -> None:
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
        self._start_receive_loop(state)

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            state = self._sessions.get(session_id)
            if state:
                await self.disconnect(state.session)

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

    async def _reconnect(self, session: VoiceSession) -> None:
        """Reconnect to Gemini Live using the stored config."""
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
            ctxmgr, live_session = await self._open_live_session(live_config)
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
                ctxmgr, live_session = await self._open_live_session(live_config)
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
