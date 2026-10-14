"""One VoiceChannel serving sessions from several transports (RMK-353, RFC §12.7.3).

``VoiceChannel.add_backend`` registers a second transport — SIP callers beside
WebRTC participants — whose inbound audio enters the channel's pipeline and
whose sessions get everything addressed to them through their own backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability
from roomkit.voice.bridge import AudioBridgeConfig
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.mock import MockTTSProvider


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x01" * 160, sample_rate=16000)


class _StreamingTTS(TTSProvider):
    """Reads streamed text: a short chunk of silence per sentence."""

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for _sentence in text_stream:
            yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)


async def _kit_with_two_transports(
    *,
    phone_capabilities: VoiceCapability = VoiceCapability.NONE,
    **channel_kwargs: object,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVoiceBackend, str]:
    kit = RoomKit()
    primary = MockVoiceBackend()
    phone = MockVoiceBackend(capabilities=phone_capabilities)
    channel = VoiceChannel("voice", backend=primary, **channel_kwargs)  # type: ignore[arg-type]
    channel.add_backend(phone)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    return kit, channel, primary, phone, room.id


def _binding(room_id: str) -> ChannelBinding:
    return ChannelBinding(room_id=room_id, channel_id="voice", channel_type=ChannelType.VOICE)


def _context(room_id: str) -> RoomContext:
    return RoomContext(room=Room(id=room_id), bindings=[_binding(room_id)])


def _answer(room_id: str, body: str) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body=body),
    )


def _sent_to(backend: MockVoiceBackend, method: str) -> list[str]:
    return [c.args.get("session_id") for c in backend.calls if c.method == method]


async def _phone_session(
    **channel_kwargs: object,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVoiceBackend, str, object]:
    kit, channel, primary, phone, room_id = await _kit_with_two_transports(**channel_kwargs)
    session = await phone.connect(room_id, "caller", "voice")
    await kit.join(room_id, "voice", session=session)
    return kit, channel, primary, phone, room_id, session


class TestAddBackend:
    def test_a_channel_without_a_backend_refuses(self) -> None:
        channel = VoiceChannel("voice")
        with pytest.raises(RuntimeError, match="backend"):
            channel.add_backend(MockVoiceBackend())

    async def test_adding_the_same_backend_twice_wires_it_once(self) -> None:
        kit, channel, _, phone, room_id = await _kit_with_two_transports(
            stt=MockSTTProvider(transcripts=["hello"]),
            pipeline=AudioPipelineConfig(
                vad=MockVADProvider(
                    events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x00\x00" * 160)]
                )
            ),
        )
        channel.add_backend(phone)
        session = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=session)

        await phone.simulate_audio_received(session, _frame())
        await asyncio.sleep(0.05)

        assert len(channel._stt.calls) == 1  # type: ignore[union-attr]
        await kit.close()

    async def test_the_added_transports_audio_reaches_the_stt(self) -> None:
        transcripts: list[str] = []
        kit, _, _, phone, room_id = await _kit_with_two_transports(
            stt=MockSTTProvider(transcripts=["from the phone"]),
            pipeline=AudioPipelineConfig(
                vad=MockVADProvider(
                    events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x00\x00" * 160)]
                )
            ),
        )

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: object, ctx: object) -> None:
            transcripts.append(event.text)  # type: ignore[attr-defined]

        session = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=session)
        await phone.simulate_audio_received(session, _frame())
        await asyncio.sleep(0.05)

        assert transcripts == ["from the phone"]
        await kit.close()

    async def test_the_added_transports_session_ready_fires_the_hook(self) -> None:
        started: list[str] = []
        kit, _, _, phone, room_id = await _kit_with_two_transports()

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def on_ready(event: object, ctx: object) -> None:
            started.append(event.session.id)  # type: ignore[attr-defined]

        # MockVoiceBackend.connect() signals the session ready at once: the
        # channel holds it until the session is bound, then fires the hook.
        session = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=session)
        await asyncio.sleep(0.05)

        assert started == [session.id]
        await kit.close()

    async def test_say_to_a_phone_session_goes_out_on_the_phone(self) -> None:
        kit, channel, primary, phone, room_id = await _kit_with_two_transports(
            tts=MockTTSProvider()
        )
        session = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=session)

        await channel.say(session, "Hello caller")

        assert [c.args.get("session_id") for c in phone.calls if c.method == "send_audio"] == [
            session.id
        ]
        assert "send_audio" not in [c.method for c in primary.calls]
        await kit.close()

    async def test_the_bridge_sends_to_each_session_on_its_own_transport(self) -> None:
        kit, _, primary, phone, room_id = await _kit_with_two_transports(
            bridge=AudioBridgeConfig(mixing_strategy="forward")
        )
        browser = await primary.connect(room_id, "browser", "voice")
        caller = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=browser)
        await kit.join(room_id, "voice", session=caller)

        await primary.simulate_audio_received(browser, _frame())
        await asyncio.sleep(0.05)

        sent = {c.args["session_id"] for c in phone.calls if c.method == "send_audio_sync"}
        assert sent == {caller.id}
        assert not [c for c in primary.calls if c.method == "send_audio_sync"]
        await kit.close()

    async def test_closing_the_channel_closes_the_added_backend(self) -> None:
        kit, _, _, phone, _ = await _kit_with_two_transports()
        await kit.close()
        assert "close" in [c.method for c in phone.calls]

    async def test_leaving_a_phone_session_hangs_up_on_the_phone(self) -> None:
        kit, _, primary, phone, room_id = await _kit_with_two_transports()
        session = await phone.connect(room_id, "caller", "voice")
        await kit.join(room_id, "voice", session=session)

        await kit.leave(session)

        assert [c.args["session_id"] for c in phone.calls if c.method == "disconnect"] == [
            session.id
        ]
        assert not [c for c in primary.calls if c.method == "disconnect"]
        await kit.close()


class TestEachSessionOnItsOwnBackend:
    """What one session gets goes through its own backend, and that backend's
    capabilities decide for it (RFC §12.7.3)."""

    async def test_a_response_reaches_a_phone_session(self) -> None:
        kit, channel, primary, phone, room_id, session = await _phone_session(
            tts=MockTTSProvider()
        )

        await channel.deliver(
            _answer(room_id, "Hello caller"), _binding(room_id), _context(room_id)
        )

        assert _sent_to(phone, "send_audio") == [session.id]
        assert _sent_to(primary, "send_audio") == []
        await kit.close()

    async def test_a_streamed_response_reaches_a_phone_session(self) -> None:
        kit, channel, primary, phone, room_id, session = await _phone_session(tts=_StreamingTTS())

        async def sentences() -> AsyncIterator[str]:
            yield "Hello caller."

        await channel.deliver_stream(
            sentences(), _answer(room_id, ""), _binding(room_id), _context(room_id)
        )

        assert _sent_to(phone, "send_audio") == [session.id]
        assert _sent_to(phone, "send_transcription")
        assert _sent_to(primary, "send_audio") == []
        assert _sent_to(primary, "send_transcription") == []
        await kit.close()

    async def test_an_interruption_cancels_on_the_phone_that_can(self) -> None:
        kit, channel, _, phone, _, session = await _phone_session(
            phone_capabilities=VoiceCapability.INTERRUPTION
        )
        channel._playing_sessions[session.id] = TTSPlaybackState(session.id, "Hello")  # noqa: SLF001

        await channel.interrupt(session)

        assert _sent_to(phone, "cancel_audio") == [session.id]
        await kit.close()

    async def test_a_phone_that_cancels_its_echo_ends_its_playback_at_once(self) -> None:
        kit, channel, _, _, _, session = await _phone_session(
            tts=MockTTSProvider(), phone_capabilities=VoiceCapability.NATIVE_AEC
        )

        await channel.say(session, "Hello caller")

        assert session.id not in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    async def test_interrupting_the_room_reaches_a_phone_session(self) -> None:
        kit, channel, _, _, room_id, session = await _phone_session()
        channel._playing_sessions[session.id] = TTSPlaybackState(session.id, "Hello")  # noqa: SLF001

        assert await channel.interrupt_all(room_id) == 1
        await kit.close()

    async def test_speech_held_on_a_phone_session_is_handed_on(self) -> None:
        kit, channel, _, _, _, session = await _phone_session(
            stt=MockSTTProvider(transcripts=["held"])
        )
        channel._queued_speech[session.id] = [(b"\x00\x00" * 160, None)]  # noqa: SLF001

        await channel._flush_queued_speech(session.id)  # noqa: SLF001

        assert len(channel._stt.calls) == 1  # type: ignore[union-attr]
        await kit.close()

    async def test_dtmf_to_a_phone_session_goes_out_on_the_phone(self) -> None:
        kit, channel, primary, phone, _, session = await _phone_session()

        channel.send_dtmf(session, "5")

        assert phone.sent_dtmf == [(session.id, "5", 160)]
        assert primary.sent_dtmf == []
        await kit.close()

    async def test_disconnecting_a_phone_session_hangs_up_on_the_phone(self) -> None:
        kit, channel, primary, phone, room_id, session = await _phone_session()

        await channel.disconnect_session(session, room_id)

        assert _sent_to(phone, "disconnect") == [session.id]
        assert _sent_to(primary, "disconnect") == []
        await kit.close()
