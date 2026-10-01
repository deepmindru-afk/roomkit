#!/usr/bin/env python3
"""SIP voice backend example.

Demonstrates how to use the SIPVoiceBackend to accept incoming SIP calls
from a PBX/SIP trunk and gate them with the same hooks as text messages.
There is no STT, AI or TTS here: the call is answered and joined to a
room, nothing is said to the caller (see voice_sip_local_agent.py or
realtime_voice_sip_gemini.py for a talking agent).

Each answered call goes through ``kit.process_inbound(parse_voice_session(...))``,
so its ``session_started`` event runs BEFORE_BROADCAST — which can block
the call exactly like it blocks a text message — and AFTER_BROADCAST. A
blocked call is hung up with a Q.850 "call rejected" cause.

Every call lands in the ``support`` room. The backend copies a PBX's
``X-Room-ID`` header (or the Call-ID when absent) into ``session.room_id``;
route on it only behind a PBX that sets the header, since a caller who
reaches the port directly chooses its value.

Requirements:
    pip install roomkit[sip]

Usage:
    python examples/voice_sip.py

    # From a SIP client or PBX, send an INVITE to the SIP port.

Environment variables (all optional):
    SIP_LOCAL_PORT      SIP listen port (default: 5060)
    SIP_RTP_PORT_START  First RTP port (default: 10000)
    SIP_RTP_PORT_END    Last RTP port (default: 20000)
    CONSOLE             Set to 1 for the live console dashboard
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_example")

from roomkit import RoomKit, VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = "0.0.0.0"  # nosec B104
SIP_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5060"))
RTP_IP = "0.0.0.0"  # nosec B104  — use your actual IP for real deployments
RTP_PORT_START = int(os.environ.get("SIP_RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("SIP_RTP_PORT_END", "20000"))

# Simple in-memory call log (demonstrates AFTER_BROADCAST observability)
call_log: list[dict] = []


async def main() -> None:
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # Create the SIP backend
    backend = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # Create voice channel with backend (no STT/TTS in this example)
    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    # -------------------------------------------------------------------
    # Protocol trace — channel-level (all SIP traces, no room needed)
    # -------------------------------------------------------------------

    voice.on_trace(
        lambda t: logger.info("[TRACE] %s %s: %s", t.direction, t.protocol, t.summary),
        protocols=["sip"],
    )

    # -------------------------------------------------------------------
    # Hooks — identical to what you'd write for text channels
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def log_and_gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log every inbound event. Block outside business hours as an example."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller", "unknown")
            hour = datetime.now(UTC).hour

            logger.info(
                "BEFORE_BROADCAST — new voice session from %s (hour=%d UTC)",
                caller,
                hour,
            )

            # Example: reject calls outside 8:00-20:00 UTC
            if not (8 <= hour < 20):
                logger.warning("Rejecting call outside business hours (hour=%d)", hour)
                return HookResult.block("outside_business_hours")

        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Room-level protocol trace — only traces for channels in the room."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def track_calls(event: RoomEvent, ctx: RoomContext) -> None:
        """Record every voice session that passes through — observability hook."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            entry = {
                "session_id": event.content.data.get("session_id"),
                "caller": event.content.data.get("caller"),
                "room_id": ctx.room.id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            call_log.append(entry)
            logger.info(
                "AFTER_BROADCAST — call logged: %s (provider=%s)",
                entry,
                event.source.provider,
            )

    # -----------------------------------------------------------------------
    # Room setup — create once at startup, not per call
    # -----------------------------------------------------------------------

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")

    # -----------------------------------------------------------------------
    # Incoming call → through the inbound pipeline (hooks) into the room
    # -----------------------------------------------------------------------

    @backend.on_call
    async def handle_call(session):
        """Called when a SIP INVITE is accepted and RTP is active."""
        caller = session.metadata.get("caller")
        logger.info("Incoming call — session=%s caller=%s", session.id, caller)
        result = await kit.process_inbound(
            parse_voice_session(session, channel_id="voice"), room_id="support"
        )
        if result.blocked:
            # The INVITE was already answered: hang up what the hook refused.
            logger.warning("Call refused (%s) — hanging up", result.reason)
            await backend.disconnect(session, cause=21, text="Call rejected")

    # -----------------------------------------------------------------------
    # Disconnect handler
    # -----------------------------------------------------------------------

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        """Called when the remote party hangs up (BYE).

        The voice channel unbinds the session by itself on a BYE; this
        handler only logs it.
        """
        logger.info("Call ended — session=%s", session.id)

    # -----------------------------------------------------------------------
    # Start
    # -----------------------------------------------------------------------

    await backend.start()
    logger.info(
        "SIP voice backend ready — listening on %s:%d, RTP ports %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
