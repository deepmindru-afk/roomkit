"""One VoiceChannel serving sessions from several transports (RMK-353, RFC §12.7.3).

``VoiceChannel.add_backend`` registers a second transport — SIP callers beside
WebRTC participants — whose inbound audio enters the channel's pipeline and
whose sessions get everything addressed to them through their own backend.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.bridge import AudioBridgeConfig
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x01" * 160, sample_rate=16000)


async def _kit_with_two_transports(
    **channel_kwargs: object,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVoiceBackend, str]:
    kit = RoomKit()
    primary = MockVoiceBackend()
    phone = MockVoiceBackend()
    channel = VoiceChannel("voice", backend=primary, **channel_kwargs)  # type: ignore[arg-type]
    channel.add_backend(phone)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    return kit, channel, primary, phone, room.id


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
