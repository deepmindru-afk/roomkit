#!/usr/bin/env python3
"""Send DTMF tones during a SIP call via AI tool calling.

Demonstrates how an AI agent can send DTMF digits into an active SIP
call — useful for navigating IVR menus, entering PINs, or interacting
with phone systems programmatically.

The AI is given a ``send_dtmf`` tool via binding metadata.  When it
decides to press a key (e.g. "press 1 for English"), it calls the tool,
which delegates to ``VoiceChannel.send_dtmf()`` →
``SIPVoiceBackend.send_dtmf()`` → RFC 4733 RTP telephone-event.

Everything but SIP is mocked: once the call is up and audio arrives, the
mock VAD/STT "hear" an IVR menu, the mock AI answers with a ``send_dtmf``
tool call for digit 1, then confirms in text. Swap in real STT/AI/TTS
providers to navigate a real IVR.

Requirements:
    pip install roomkit[sip]

Run with:
    uv run python examples/voice_sip_dtmf.py

Environment variables (all optional):
    SIP_PROXY_HOST  — SIP proxy IP        (default: 127.0.0.1)
    SIP_PROXY_PORT  — SIP proxy port      (default: 5060)
    SIP_FROM_URI    — caller SIP URI      (default: sip:bot@example.com)
    SIP_TO_URI      — callee SIP URI      (default: sip:ivr@example.com)
    SIP_LOCAL_PORT  — local SIP port      (default: 5070)
    SIP_RTP_PORT_START — first RTP port   (default: 10000)
    SIP_RTP_PORT_END   — last RTP port    (default: 20000)
    CONSOLE         — 1 for the live console dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_dtmf")

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.backends.sip import PT_PCMU, SIPVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.pipeline.dtmf import MockDTMFDetector
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_PROXY_HOST = os.environ.get("SIP_PROXY_HOST", "127.0.0.1")
SIP_PROXY_PORT = int(os.environ.get("SIP_PROXY_PORT", "5060"))
FROM_URI = os.environ.get("SIP_FROM_URI", "sip:bot@example.com")
TO_URI = os.environ.get("SIP_TO_URI", "sip:ivr@example.com")
LOCAL_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5070"))
RTP_PORT_START = int(os.environ.get("SIP_RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("SIP_RTP_PORT_END", "20000"))

SYSTEM_PROMPT = (
    "You are an AI agent navigating a phone IVR system. "
    "When you hear menu options (e.g. 'press 1 for sales'), "
    "use the send_dtmf tool to press the correct key. "
    "You can send digits 0-9, *, and #."
)

# Tool definition as a raw dict (passed via binding metadata)
DTMF_TOOL = {
    "name": "send_dtmf",
    "description": (
        "Send a DTMF tone (key press) into the active phone call. "
        "Use this to navigate IVR menus or enter numeric codes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "digit": {
                "type": "string",
                "description": "The DTMF digit to send: 0-9, *, or #",
                "enum": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "*",
                    "#",
                ],
            },
            "duration_ms": {
                "type": "integer",
                "description": "Tone duration in milliseconds (default 160)",
                "default": 160,
            },
        },
        "required": ["digit"],
    },
}


async def main() -> None:
    kit = RoomKit()
    console_cleanup = setup_console(kit)
    room_id = "dtmf-demo"

    # --- SIP backend ----------------------------------------------------------
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", LOCAL_PORT),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # --- Pipeline: VAD + DTMF (mock for demo) ---------------------------------
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"demo-audio",
                duration_ms=2000.0,
            ),
        ]
    )
    dtmf_detector = MockDTMFDetector()
    pipeline = AudioPipelineConfig(vad=vad, dtmf=dtmf_detector)

    # --- STT + TTS (mock for demo) --------------------------------------------
    stt = MockSTTProvider(
        transcripts=["Press 1 for sales, press 2 for support."],
    )
    tts = MockTTSProvider()

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    # --- DTMF tool handler ----------------------------------------------------
    # Capture the active session so the tool handler can send DTMF.
    active_session: dict[str, VoiceSession] = {}

    async def handle_tool(name: str, arguments: dict) -> str:
        if name == "send_dtmf":
            digit = arguments["digit"]
            duration_ms = arguments.get("duration_ms", 160)
            session = active_session.get(room_id)
            if session is None:
                return json.dumps({"error": "No active voice session"})
            # send_dtmf is synchronous (RFC 4733 packets are queued)
            voice.send_dtmf(session, digit, duration_ms)
            logger.info("DTMF sent: digit=%s duration=%sms", digit, duration_ms)
            return json.dumps(
                {
                    "status": "sent",
                    "digit": digit,
                    "duration_ms": duration_ms,
                }
            )
        return json.dumps({"error": f"Unknown tool: {name}"})

    # --- AI channel -----------------------------------------------------------
    # Scripted like a real model: first a send_dtmf tool call, then, once the
    # tool result is back, the text reply that goes to TTS.
    ai_provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(id="dtmf-1", name="send_dtmf", arguments={"digit": "1"}),
                ],
            ),
            AIResponse(content="I pressed 1 for sales.", finish_reason="stop"),
        ],
    )
    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=SYSTEM_PROMPT,
        tool_handler=handle_tool,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "voice")
    await kit.attach_channel(
        room_id,
        "ai",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [DTMF_TOOL]},
    )

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("Transcription: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
    async def on_dtmf(event, ctx):
        logger.info(
            "DTMF received: digit=%s duration=%sms",
            event.digit,
            event.duration_ms,
        )

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Dial -----------------------------------------------------------------
    await backend.start()

    logger.info(
        "Dialing %s from %s via %s:%d ...",
        TO_URI,
        FROM_URI,
        SIP_PROXY_HOST,
        SIP_PROXY_PORT,
    )
    try:
        session = await backend.dial(
            to_uri=TO_URI,
            from_uri=FROM_URI,
            proxy_addr=(SIP_PROXY_HOST, SIP_PROXY_PORT),
            codec=PT_PCMU,
            timeout=30.0,
        )
    except (TimeoutError, RuntimeError) as exc:
        logger.error("Call failed: %s", exc)
        await backend.close()
        return

    # Join session to room, then expose to the tool handler
    await kit.join(room_id, "voice", session=session)
    active_session[room_id] = session

    logger.info(
        "Call active — session=%s. Waiting for IVR prompts...",
        session.id,
    )

    # --- Keep running ---------------------------------------------------------
    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
