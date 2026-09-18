"""RoomKit -- Realtime voice with FastRTC (WebRTC) transport.

A speech-to-speech example using Google Gemini Live over WebRTC.
Audio flows between the user's browser and Gemini via FastRTC's
passthrough handler (no VAD -- Gemini handles speech detection).
Transcriptions are emitted as RoomEvents so other channels see
the conversation.

Demonstrates pluggable auth and per-connection TURN credential resolution.
Set TURN_CREDENTIALS_URL to an HTTP endpoint returning {"iceServers": [...]}.

Requirements:
    pip install roomkit[realtime-gemini,fastrtc] fastapi uvicorn httpx

Run with:
    GEMINI_API_KEY=... uv run uvicorn examples.realtime_voice_fastrtc:app
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from shared import setup_console, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.fastrtc_transport import (
    FastRTCRealtimeTransport,
    mount_fastrtc_realtime,
)

logger = setup_logging("realtime_voice_fastrtc")

kit = RoomKit()

# --- Console dashboard (set CONSOLE=1 to enable) ---
_console_cleanup = setup_console(kit)

# --- Gemini Live provider ---
provider = GeminiLiveProvider(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model="gemini-3.8-live",
)

# --- FastRTC transport (WebRTC, passthrough mode) ---
transport = FastRTCRealtimeTransport(
    input_sample_rate=16000,
    output_sample_rate=24000,
)

# --- Realtime voice channel ---
channel = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    transport=transport,
    system_prompt="You are a friendly voice assistant. Be concise.",
    voice="Aoede",
)
kit.register_channel(channel)


# --- Auto-create session on WebRTC connect ---
async def on_client_connected(webrtc_id: str) -> None:
    """Called when a browser connects via WebRTC."""
    room = await kit.create_room()
    await kit.attach_channel(room.id, "realtime-voice")
    session = await channel.start_session(room.id, "user-1", connection=webrtc_id)
    logger.info(
        "Session started: session=%s, room=%s, webrtc_id=%s",
        session.id,
        room.id,
        webrtc_id,
    )


transport.on_client_connected(on_client_connected)


# --- Optional: pluggable auth on the transport ---
async def authenticate_connection(ctx: object) -> dict[str, object] | None:
    """Example auth callback for WebRTC connections.

    In production, inspect headers/tokens from the connection context.
    Return a metadata dict on success, or None to reject.
    """
    # Accept all connections in this example
    return {"authenticated": True}


# --- FastAPI app ---
async def rtc_configuration() -> dict[str, Any]:
    """Fetch short-lived credentials when a peer connects, never at startup."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(os.environ["TURN_CREDENTIALS_URL"])
        response.raise_for_status()
        return response.json()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mount WebRTC endpoints with auth
    mount_fastrtc_realtime(
        app,
        transport,
        path="/rtc-realtime",
        auth=authenticate_connection,
        rtc_configuration=rtc_configuration if os.environ.get("TURN_CREDENTIALS_URL") else None,
    )
    yield
    if _console_cleanup:
        await _console_cleanup()
    await kit.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    """Serve the FastRTC browser client."""
    html_path = Path(__file__).parent / "voice_agent_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return {
        "message": "RoomKit Realtime Voice (FastRTC/WebRTC)",
        "webrtc_endpoint": "/rtc-realtime",
    }
