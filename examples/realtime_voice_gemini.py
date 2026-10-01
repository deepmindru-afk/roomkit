"""RoomKit -- Realtime voice with Gemini Live over a plain WebSocket.

A minimal speech-to-speech server built on ``WebSocketRealtimeTransport``:
a client streams microphone audio over a WebSocket, RoomKit relays it to
Google Gemini Live, and the model's voice comes back on the same socket.
Gemini detects the end of each turn itself; there is no local VAD.

Open http://localhost:8000/ for the bundled page
(``realtime_voice_gemini.html``: microphone in, speaker out). Use
headphones: the page cancels no echo, so the model would hear itself.
Browsers grant the microphone to plain-HTTP pages on localhost only.

Any other client speaks this protocol on ``ws://localhost:8000/ws``:

    client -> server  binary frames of 16 kHz, 16-bit, mono PCM (little-endian),
                      e.g. 640 bytes per 20 ms; or {"type": "audio", "data": "<base64>"}
    server -> client  binary frames of 24 kHz, 16-bit, mono PCM (the model's voice)
                      {"type": "session_started"}  (Gemini is listening)
                      {"type": "transcription", "text": ..., "role": ..., "is_final": ...}
                      {"type": "speaking", "speaking": ..., "who": "user" | "assistant"}
                      {"type": "clear_audio"}  (the user barged in: drop queued audio)
                      {"type": "session_ended"}

Each connection gets its own room and Gemini session; closing the socket
ends the session.

Requirements:
    pip install roomkit[realtime-gemini,websocket]

Run with:
    GEMINI_API_KEY=... uv run python examples/realtime_voice_gemini.py

Environment variables:
    GEMINI_API_KEY  (required) Google AI API key
    HOST            Interface to bind (default: localhost)
    PORT            HTTP and WebSocket port (default: 8000)
    CONSOLE         Set to 1 for the live console dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from http import HTTPStatus
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, run_until_stopped, setup_console, setup_logging
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response
from websockets.typing import Origin

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport

logger = setup_logging("realtime_voice_gemini")
# websockets logs each page load ("connection rejected (200 OK)") and each socket
# a browser opens ahead of time and then drops ("opening handshake failed", with a
# traceback). Neither is a fault; a crash in handle_client still shows.
logging.getLogger("websockets.server").addFilter(
    lambda record: record.levelno >= logging.WARNING and record.msg != "opening handshake failed"
)

PAGE = Path(__file__).with_suffix(".html").read_text()


def serve_page(connection: ServerConnection, request: Request) -> Response | None:
    """Answer plain HTTP requests; let ``/ws`` continue to the WebSocket handshake."""
    if request.path == "/ws":
        return None
    if request.path != "/":
        return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
    response = connection.respond(HTTPStatus.OK, PAGE)
    del response.headers["Content-Type"]
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", "8000"))

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- Gemini Live behind a WebSocket transport ---
    # The transport passes audio through unchanged: clients send 16 kHz and
    # receive 24 kHz, Gemini Live's native rates.
    channel = RealtimeVoiceChannel(
        "realtime-voice",
        provider=GeminiLiveProvider(api_key=env["GEMINI_API_KEY"], model="gemini-3.8-live"),
        transport=WebSocketRealtimeTransport(),
        system_prompt="You are a friendly voice assistant. Be concise.",
        voice="Aoede",  # Gemini voice preset
    )
    kit.register_channel(channel)

    async def handle_client(ws: ServerConnection) -> None:
        """Give each WebSocket client its own room and Gemini Live session."""
        room = await kit.create_room()
        await kit.attach_channel(room.id, "realtime-voice")
        try:
            session = await channel.start_session(room.id, "caller", ws)
        except Exception:
            logger.exception("Could not start a Gemini Live session")
            return  # returning closes the socket
        logger.info("Session %s started in room %s", session.id, room.id)
        # The transport reads the socket; keep it open until the client
        # leaves, then the channel ends the session.
        await ws.wait_closed()
        logger.info("Session %s: client disconnected", session.id)

    # Only the bundled page and clients sending no Origin (scripts) may
    # connect: any other website open in the browser could otherwise talk to
    # Gemini on this key.
    origins: list[Origin | None] = [
        Origin(f"http://{name}:{port}") for name in ("localhost", "127.0.0.1", "[::1]")
    ]
    origins.append(None)
    server = await serve(handle_client, host, port, process_request=serve_page, origins=origins)
    print(f"Open http://localhost:{port}/ (headphones on), or stream to ws://localhost:{port}/ws")

    async def cleanup() -> None:
        server.close()
        await server.wait_closed()
        if console_cleanup:
            await console_cleanup()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
