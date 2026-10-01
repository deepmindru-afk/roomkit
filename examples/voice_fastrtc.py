"""RoomKit -- Voice assistant over FastRTC (WebSocket transport).

A browser-to-AI voice assistant using FastRTCVoiceBackend for the
traditional STT/TTS pipeline. Audio flows from the browser via
FastRTC's WebSocket transport through the full voice pipeline:

  Browser mic → FastRTC WebSocket → energy VAD → Deepgram STT
    → Claude AI → ElevenLabs TTS → mu-law → Browser

The audio endpoints are mounted under /voice: /voice/websocket/offer
(mu-law in JSON) and /voice/webrtc/offer (WebRTC signaling). The page at
http://localhost:8000/ is ``voice_agent_ui.html``: open its settings, pick
"WebSocket (STT/TTS pipeline)" and keep the WS endpoint at /voice.

This module is an ASGI app, not a script: start it with uvicorn.

Requirements:
    pip install roomkit[fastrtc,anthropic,deepgram,elevenlabs]

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    uv run uvicorn examples.voice_fastrtc:app

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    STT_LANGUAGE        STT language code (default: en)
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)
    SYSTEM_PROMPT       Custom system prompt for Claude
    CONSOLE             Set to 1 for the live console dashboard
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from shared import require_env, setup_console, setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = setup_logging("voice_fastrtc")
env = require_env("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")

kit = RoomKit()
console_cleanup = setup_console(kit)

# --- Audio settings ---
INPUT_SAMPLE_RATE = 48000  # Browser mic (FastRTC default)
OUTPUT_SAMPLE_RATE = 24000  # ElevenLabs native rate

# --- Backend: FastRTC WebSocket transport ---
backend = FastRTCVoiceBackend(
    input_sample_rate=INPUT_SAMPLE_RATE,
    output_sample_rate=OUTPUT_SAMPLE_RATE,
)

# --- VAD ---
vad = EnergyVADProvider(
    energy_threshold=300.0,
    silence_threshold_ms=600,
    min_speech_duration_ms=200,
)

# --- Pipeline ---
pipeline = AudioPipelineConfig(vad=vad)

# --- Deepgram STT ---
stt_language = os.environ.get("STT_LANGUAGE", "en")
stt = DeepgramSTTProvider(
    config=DeepgramConfig(
        api_key=env["DEEPGRAM_API_KEY"],
        model="nova-3",
        language=stt_language,
        punctuate=True,
        smart_format=True,
        endpointing=300,
    )
)

# --- ElevenLabs TTS ---
tts = ElevenLabsTTSProvider(
    config=ElevenLabsConfig(
        api_key=env["ELEVENLABS_API_KEY"],
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        model_id="eleven_multilingual_v2",
        output_format=f"pcm_{OUTPUT_SAMPLE_RATE}",
        optimize_streaming_latency=3,
    )
)

# --- Claude AI ---
ai_provider = AnthropicAIProvider(
    AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-opus-5",
        max_tokens=256,
    )
)

system_prompt = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a friendly voice assistant. Keep your responses "
    "short and conversational — one or two sentences at most.",
)

# --- Channels ---
voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
kit.register_channel(voice)

ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
kit.register_channel(ai)


# --- Session factory: auto-create room + session on WebSocket connect ---
async def session_factory(websocket_id: str):
    """Create a room and voice session when a browser connects."""
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "ai", category=ChannelCategory.INTELLIGENCE)
    session = await backend.connect(room.id, "browser-user", "voice")
    session.metadata["websocket_id"] = websocket_id
    await kit.join(room.id, "voice", session=session)
    logger.info("Session created: session=%s, room=%s", session.id, room.id)
    return session


# --- Hooks ---
@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def on_transcription(event, ctx):
    logger.info("User: %s", event.text)
    return HookResult.allow()


@kit.hook(HookTrigger.BEFORE_TTS)
async def before_tts(text, ctx):
    logger.info("Assistant: %s", text)
    return HookResult.allow()


@kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
async def on_speech_start(session, ctx):
    logger.info("Speech started")


@kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
async def on_speech_end(session, ctx):
    logger.info("Speech ended")


# --- Optional: auth callback ---
async def authenticate(websocket) -> dict[str, object] | None:
    """Accept all connections. In production, validate tokens here."""
    return {"authenticated": True}


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    mount_fastrtc_voice(
        app,
        backend,
        path="/voice",
        session_factory=session_factory,
        auth=authenticate,
    )
    logger.info("FastRTC voice backend ready at /voice")
    logger.info("Open http://localhost:8000 for the browser client")
    yield
    if console_cleanup:
        await console_cleanup()
    await kit.close()


app = FastAPI(lifespan=lifespan)


# Shared browser client: pick WebSocket mode in its settings, endpoint /voice.
BROWSER_CLIENT_HTML = (Path(__file__).parent / "voice_agent_ui.html").read_text()


@app.get("/")
async def index():
    """Serve the browser client."""
    return HTMLResponse(BROWSER_CLIENT_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "transport": "fastrtc-websocket"}
