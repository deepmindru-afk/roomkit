"""Full-duplex turn boundaries include audio activity and awaited continuations."""

from __future__ import annotations

import asyncio
import base64

import pytest

from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.reasoning import ReasoningOutput
from tests.test_openai_live import _connect, _provider, _Recorder
from tests.test_realtime_reasoning import _channel, _ScriptedBackend


@pytest.mark.parametrize("transcript", [False, True])
@pytest.mark.parametrize(
    ("codec", "silence"),
    [("pcm", b"\x00\x00" * 480), ("pcma", b"\xd5" * 160)],
    ids=["pcm", "pcma"],
)
async def test_audio_keeps_turn_open_but_continuous_silence_does_not(transcript, codec, silence):
    provider = _provider(turn_gap_ms=80)
    session = VoiceSession(id="s", room_id="r", participant_id="p", channel_id="rt")
    recorded = _Recorder(provider)
    rate = 24000 if codec == "pcm" else 8000
    await _connect(
        provider,
        session,
        input_sample_rate=rate,
        output_sample_rate=rate,
        provider_config={"codec": codec},
    )
    state = provider._states[session.id]

    async def audio(chunk):
        await provider._handle_server_event(
            state,
            {"type": "session.output_audio.delta", "delta": base64.b64encode(chunk).decode()},
        )

    try:
        if transcript:
            await provider._handle_server_event(
                state, {"type": "session.output_transcript.delta", "delta": "Goodbye."}
            )
        active = b"\x00\x20" * 480 if codec == "pcm" else b"\x80" * 160
        for _ in range(5):
            await audio(active)
            await asyncio.sleep(0.03)
            assert recorded.responses == ["start"]
        # Silence is still delivered, including the nonzero decoded G.711 floor.
        for _ in range(5):
            await audio(silence)
            await asyncio.sleep(0.03)
        assert recorded.responses == ["start", "end"]
        assert len(recorded.audio) == 10
        await audio(active)
        assert recorded.responses == ["start", "end", "start"]
    finally:
        await provider.disconnect(session)


@pytest.mark.parametrize("same_turn", [False, True])
@pytest.mark.parametrize("continuation", ["audio", "transcript"])
async def test_spoken_delegation_waits_for_actual_continuation(same_turn, continuation):
    backend = _ScriptedBackend([ReasoningOutput("Goodbye.", is_final=True)])
    _, channel, provider, session = await _channel(backend)
    try:
        if same_turn:
            await provider.simulate_response_start(session)
        await provider.simulate_delegation(session, "d1", "integrator")
        await asyncio.sleep(0.03)
        assert provider.delegation_outputs
        # A stale end, silence and a synthesized final transcript cannot acknowledge it.
        await provider.simulate_response_end(session)
        await provider.simulate_audio(session, b"\x00\x00" * 480)
        await provider.simulate_transcription(session, "Earlier speech", "assistant", True)
        with pytest.raises(TimeoutError):
            await channel.wait_idle("r1", timeout=0.01)
        if not same_turn:
            await provider.simulate_response_start(session)
        if continuation == "audio":
            await provider.simulate_audio(session, b"\x00\x20" * 480)
        else:
            await provider.simulate_transcription(session, "Goodbye.", "assistant", False)
        with pytest.raises(TimeoutError):
            await channel.wait_idle("r1", timeout=0.01)
        await provider.simulate_response_end(session)
        await channel.wait_idle("r1", timeout=1)
    finally:
        await channel.close()


async def test_silent_delegation_does_not_require_a_spoken_response():
    backend = _ScriptedBackend([ReasoningOutput("Internal fact.", spoken=False, is_final=True)])
    _, channel, provider, session = await _channel(backend)
    try:
        await provider.simulate_delegation(session, "d1", "integrator")
        await channel.wait_idle("r1", timeout=1)
        assert provider.delegation_outputs
    finally:
        await channel.close()
