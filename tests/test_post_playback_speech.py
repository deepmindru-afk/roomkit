"""Speech that starts right after the agent stops speaking (RMK-211).

Once ``send_audio()`` returns, a playback keeps an echo-decay window. Without
echo cancellation the window discards speech as echo; with an AEC — the
pipeline's, or the backend's own (``NATIVE_AEC``) — the echo is cancelled, so
a reply the user starts as soon as the agent is done is heard.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


async def _reply_right_after_playback(
    aec: MockAECProvider | None,
    *,
    capabilities: VoiceCapability = VoiceCapability.INTERRUPTION,
) -> tuple[list[str], VoiceChannel, str, RoomKit]:
    """The agent says a line; the user answers the moment it is done."""
    backend = MockVoiceBackend(capabilities=capabilities)
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 1600),
        ]
    )
    channel = VoiceChannel(
        "voice-1",
        stt=MockSTTProvider(transcripts=["Yes, that works"]),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad, aec=aec),
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")
    channel.bind_session(
        session,
        room.id,
        ChannelBinding(room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE),
    )

    heard: list[str] = []

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_transcription(event, context):
        heard.append(event.text)

    await channel.say(session, "Thanks, it means a lot.")
    # send_audio() has returned: the user starts speaking at once.
    frame = AudioFrame(data=b"\x01\x00" * 160)
    await backend.simulate_audio_received(session, frame)
    await backend.simulate_audio_received(session, frame)
    await asyncio.sleep(0.2)
    return heard, channel, session.id, kit


class TestSpeechRightAfterPlayback:
    async def test_with_aec_the_reply_is_transcribed(self) -> None:
        heard, _, _, kit = await _reply_right_after_playback(MockAECProvider())

        assert heard == ["Yes, that works"]
        await kit.close()

    async def test_with_the_backends_own_aec_the_reply_is_transcribed(self) -> None:
        # LocalAudioBackend(aec=...) cancels echo at the transport and says so
        # with NATIVE_AEC; the pipeline then runs no AEC of its own.
        heard, _, _, kit = await _reply_right_after_playback(
            None, capabilities=VoiceCapability.INTERRUPTION | VoiceCapability.NATIVE_AEC
        )

        assert heard == ["Yes, that works"]
        await kit.close()

    async def test_without_aec_the_echo_window_still_discards_it(self) -> None:
        # Nothing cancels the room's echo: speech in the decay window is echo.
        heard, _, _, kit = await _reply_right_after_playback(None)

        assert heard == []
        await kit.close()

    async def test_aec_stays_active_through_the_echo_tail(self) -> None:
        aec = MockAECProvider()
        _, channel, session_id, kit = await _reply_right_after_playback(aec)

        # The playback is over for the channel, but the AEC still cancels
        # the room's echo tail...
        assert session_id not in channel._playing_sessions
        assert aec.active_changes[-1] == (session_id, True)

        # ...and is bypassed once the tail has decayed.
        await asyncio.sleep(0.5)
        assert aec.active_changes[-1] == (session_id, False)
        await kit.close()
