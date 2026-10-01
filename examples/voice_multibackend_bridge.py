#!/usr/bin/env python3
"""Multi-transport audio bridge: SIP phones and browsers in one room, with a live analyst.

Bridges participants from different transports into a single conference
room with N-party mixing. SIP phones and browsers (WebRTC or WebSocket)
all hear each other, with live Deepgram transcription. While the call runs,
Claude reads the transcript every few lines and gives a live analysis
(sentiment, topics, alert); when the last participant leaves, it writes a
meeting summary.

One ``VoiceChannel`` serves both transports: it is built on the FastRTC
backend and ``voice.add_backend(sip_backend)`` adds the SIP one, so phone
audio goes through the same pipeline, bridge and STT, and everything sent
to a phone caller goes out on SIP.

Architecture::

    SIP phone  ──► SIP backend ─────┐   (voice.add_backend)
    Browser    ──► FastRTC (WebRTC) ┼──► VoiceChannel ──► AudioBridge (N-party mix)
    Browser    ──► FastRTC (WS)    ─┘                 └──► STT (Deepgram)
                                                              │
                                        live analysis ◄───────┤ every N lines
                                        (/ws/analysis)        │
                                        summary when the room empties

Web UI: open http://localhost:8000 (serves examples/voice_agent_ui.html), set:
  - Server URL: http://localhost:8000
  - WS endpoint: /voice
  - Transport: WebSocket or WebRTC
Live analysis feed: a WebSocket client on ws://localhost:8000/ws/analysis
receives each analysis as JSON.

Requirements:
    pip install roomkit[sip,fastrtc,deepgram,anthropic]

Run with:
    DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/voice_multibackend_bridge.py

Environment variables:
    DEEPGRAM_API_KEY   -- Deepgram API key (required)
    ANTHROPIC_API_KEY  -- Anthropic API key (required)
    STT_LANGUAGE       -- Language code for STT (default: multi = auto-detect)
    CLAUDE_MODEL       -- Claude model ID (default: claude-opus-5)
    ANALYSIS_INTERVAL  -- Analyse every N transcribed lines (default: 5, 0 = off)
    SIP_LISTEN_ADDR    -- SIP listen IP   (default: 0.0.0.0)
    SIP_LISTEN_PORT    -- SIP listen port (default: 5060)
    RTP_IP             -- RTP bind IP     (default: 0.0.0.0)
    HTTP_PORT          -- HTTP port for FastRTC + UI (default: 8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_multibackend_bridge")

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.bridge import AudioBridgeConfig
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "multi")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
ANALYSIS_INTERVAL = int(os.environ.get("ANALYSIS_INTERVAL", "5"))
SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8000"))

ROOM_ID = "bridge-room"

ANALYST_PROMPT = """\
You monitor a live phone/web conference from its transcript. Answer with one
JSON object and nothing else:
{"sentiment": "positive" | "neutral" | "negative",
 "sentiment_score": -1.0 to 1.0,
 "topics": ["..."],
 "alert": null or "one sentence on an urgent issue"}
Write the topics and the alert in the language of the transcript."""

SUMMARY_PROMPT = (
    "You are a meeting assistant. Given a conversation transcript, produce a "
    "concise summary with:\n- Key topics discussed\n- Decisions made\n"
    "- Action items (if any)\nKeep it brief and actionable.\n\n"
    "IMPORTANT: Write the summary in the same language as the transcript."
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

kit = RoomKit()
console_cleanup = setup_console(kit)
transcript: list[tuple[str, str]] = []
participants: dict[str, str] = {}  # session id -> display name
analysis_clients: set[WebSocket] = set()
background_tasks: set[asyncio.Task[None]] = set()


def _participant_name(session: Any) -> str:
    """Human-readable name from session metadata."""
    meta = session.metadata
    transport = meta.get("transport") or meta.get("backend", "unknown")
    # SIP: use caller display name
    name = (
        meta.get("caller_display_name")
        or meta.get("caller_user")
        or session.participant_id
        or session.id[:8]
    )
    return f"{name} ({transport})"


# ---------------------------------------------------------------------------
# Backends, STT and the one voice channel serving both transports
# ---------------------------------------------------------------------------

sip_backend = SIPVoiceBackend(
    local_sip_addr=(SIP_LISTEN_ADDR, SIP_LISTEN_PORT),  # nosec B104
    local_rtp_ip=RTP_IP,  # nosec B104
)

fastrtc_backend = FastRTCVoiceBackend(
    input_sample_rate=16000,
    output_sample_rate=16000,
)

stt = DeepgramSTTProvider(
    config=DeepgramConfig(
        api_key=DEEPGRAM_API_KEY,
        model="nova-3",
        language=STT_LANGUAGE,
        punctuate=True,
        smart_format=True,
        endpointing=300,
    )
)

voice = VoiceChannel(
    "voice",
    backend=fastrtc_backend,
    stt=stt,
    bridge=AudioBridgeConfig(mixing_strategy="mix", max_participants=10),
)
# Phone callers: their audio enters the same pipeline, bridge and STT, and
# whatever the channel sends to a SIP session goes out on SIP.
voice.add_backend(sip_backend)
kit.register_channel(voice)

# Claude, used twice: a live read of the transcript every few lines (called
# directly, nothing enters the room), and the summary once the room empties
# (an AI channel attached for that one turn).
claude = AnthropicAIProvider(
    AnthropicConfig(api_key=ANTHROPIC_API_KEY, model=CLAUDE_MODEL, max_tokens=1024)
)
summarizer = AIChannel("ai-summarizer", provider=claude, system_prompt=SUMMARY_PROMPT)
kit.register_channel(summarizer)


# ---------------------------------------------------------------------------
# Hooks and lifecycle events
# ---------------------------------------------------------------------------


@kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
async def on_session_started(event: Any, ctx: Any) -> None:
    if event.session is None:
        return
    participants[event.session.id] = _participant_name(event.session)
    logger.info("Joined: %s (total: %d)", participants[event.session.id], len(participants))


@kit.on("voice_session_ended")
async def on_session_ended(event: Any) -> None:
    name = participants.pop(event.data["session_id"], "someone")
    logger.info("%s left (remaining: %d)", name, len(participants))
    if not participants and transcript:
        await summarize()


@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def on_transcription(event: Any, ctx: Any) -> HookResult:
    name = _participant_name(event.session)
    transcript.append((name, event.text))
    logger.info("[STT %s] %s", name, event.text)
    if ANALYSIS_INTERVAL and len(transcript) % ANALYSIS_INTERVAL == 0:
        task = asyncio.create_task(analyse(transcript[-ANALYSIS_INTERVAL * 2 :]))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    return HookResult.allow()


# ---------------------------------------------------------------------------
# Call handling
# ---------------------------------------------------------------------------


@sip_backend.on_call
async def handle_sip_call(session: Any) -> None:
    """Incoming SIP call: add it to the bridge room.

    add_backend() lets the channel find the transport this session belongs
    to, so the bridge and TTS reach the caller over SIP. A hangup ends the
    session on the channel by itself (the backend's disconnect signal).
    """
    logger.info("SIP call: %s (session=%s)", _participant_name(session), session.id)
    await kit.join(ROOM_ID, "voice", session=session)


async def fastrtc_session_factory(connection_id: str) -> Any:
    """Create a voice session when a browser connects via FastRTC."""
    return await kit.join(ROOM_ID, "voice", participant_id=f"web-{connection_id[:8]}")


# ---------------------------------------------------------------------------
# Live analysis and summary
# ---------------------------------------------------------------------------


async def analyse(recent: list[tuple[str, str]]) -> None:
    """Ask Claude for a live read of the last lines and push it to listeners."""
    lines = "\n".join(f"{speaker}: {text}" for speaker, text in recent)
    try:
        response = await claude.generate(
            AIContext(
                system_prompt=ANALYST_PROMPT,
                messages=[AIMessage(role="user", content=lines)],
                max_tokens=300,
            )
        )
    except Exception:
        logger.exception("Live analysis failed")
        return
    try:
        analysis = json.loads(response.content)
    except json.JSONDecodeError:
        analysis = {"summary": response.content, "sentiment": "unknown"}
    logger.info("[ANALYSIS] %s", json.dumps(analysis, ensure_ascii=False))
    if analysis.get("alert"):
        logger.warning("[ALERT] %s", analysis["alert"])
    for client in list(analysis_clients):
        try:
            await client.send_json({"type": "analysis", **analysis})
        except Exception:
            analysis_clients.discard(client)


async def summarize() -> None:
    """Summarize the call once the last participant has left."""
    logger.info("All participants left. Generating summary...")
    transcript_text = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)
    transcript.clear()

    await kit.attach_channel(ROOM_ID, "ai-summarizer", category=ChannelCategory.INTELLIGENCE)
    try:
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="voice",
                sender_id="system",
                content=TextContent(
                    body="Summarize this meeting transcript:\n\n" + transcript_text
                ),
            ),
            room_id=ROOM_ID,
        )
    finally:
        await kit.detach_channel(ROOM_ID, "ai-summarizer")
    if result.error:
        logger.error("Summary failed: %s", result.error)
        return

    events = await kit.store.list_events(ROOM_ID)
    summaries = [
        e.content.body
        for e in events
        if isinstance(e.content, TextContent) and e.source.channel_id == "ai-summarizer"
    ]
    if summaries:
        logger.info(
            "\n========== MEETING SUMMARY ==========\n%s\n=====================================",
            summaries[-1],
        )
    else:
        logger.warning("No AI summary generated")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the room and the SIP backend on startup."""
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")
    await sip_backend.start()

    logger.info("=== Multi-Transport Audio Bridge ===")
    logger.info("SIP:      %s:%d", SIP_LISTEN_ADDR, SIP_LISTEN_PORT)
    logger.info("UI:       http://localhost:%d", HTTP_PORT)
    logger.info("WebRTC:   http://localhost:%d/voice/webrtc/offer", HTTP_PORT)
    logger.info("WS:       ws://localhost:%d/voice/websocket/offer", HTTP_PORT)
    logger.info("Analysis: ws://localhost:%d/ws/analysis", HTTP_PORT)
    logger.info(
        "STT: Deepgram nova-3 (lang=%s) | AI: Claude (%s) | analysis every %s lines",
        STT_LANGUAGE,
        CLAUDE_MODEL,
        ANALYSIS_INTERVAL or "no",
    )
    yield
    if console_cleanup:
        await console_cleanup()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount FastRTC endpoints at /voice
    mount_fastrtc_voice(
        app,
        fastrtc_backend,
        path="/voice",
        session_factory=fastrtc_session_factory,
        allow_anonymous=True,  # local demo; pass auth= in production
    )

    @app.websocket("/ws/analysis")
    async def analysis_feed(websocket: WebSocket) -> None:
        await websocket.accept()
        analysis_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            analysis_clients.discard(websocket)

    @app.get("/")
    async def serve_ui() -> FileResponse:
        ui_path = Path(__file__).resolve().parent / "voice_agent_ui.html"
        return FileResponse(ui_path, media_type="text/html")

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    require_env("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY")

    config = uvicorn.Config(
        create_app(),
        host="0.0.0.0",  # nosec B104
        port=HTTP_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    async def cleanup() -> None:
        server.should_exit = True
        await serve_task

    # kit.close() closes the voice channel, which closes both backends.
    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
