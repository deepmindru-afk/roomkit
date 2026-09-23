"""ElevenLabs request stitching from the TTS conversation context (RFC §12.2.2).

The provider receives its own previous turns (SELF) and sends ElevenLabs the
``request_id`` of the generations the user heard to the end, so each response
continues the voice of the previous ones.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from roomkit.voice.tts._elevenlabs_stitching import (
    MAX_REQUEST_IDS,
    REQUEST_ID_TTL_S,
    RequestIdLedger,
    request_id_of,
)
from roomkit.voice.tts.context import ConversationTurn, TTSContext, TTSContextLevel
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider


def _turn(turn_id: str, text: str = "said", *, interrupted: bool = False) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        role="assistant",
        participant_id="ai",
        text=text,
        played_ms=500,
        interrupted=interrupted,
    )


def _context(*turns: ConversationTurn, next_turn_id: str = "next") -> TTSContext:
    return TTSContext(context_id="s1", turns=turns, next_turn_id=next_turn_id)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestLedger:
    def test_no_turn_sends_nothing(self) -> None:
        assert RequestIdLedger().stitching_params(_context(), "v") == {}

    def test_heard_turns_send_their_ids_oldest_first(self) -> None:
        ledger = RequestIdLedger()
        for i in range(4):
            ledger.record("s1", f"t{i}", f"r{i}", "v")

        params = ledger.stitching_params(_context(*(_turn(f"t{i}") for i in range(4))), "v")

        assert params == {"previous_request_ids": ["r1", "r2", "r3"]}
        assert len(params["previous_request_ids"]) == MAX_REQUEST_IDS

    def test_a_cut_off_last_turn_sends_nothing(self) -> None:
        ledger = RequestIdLedger()
        ledger.record("s1", "t0", "r0", "v")

        params = ledger.stitching_params(_context(_turn("t0"), _turn("t1", interrupted=True)), "v")

        assert params == {}

    def test_ids_stop_at_an_interrupted_turn(self) -> None:
        ledger = RequestIdLedger()
        for turn_id in ("t0", "t1", "t2"):
            ledger.record("s1", turn_id, f"r-{turn_id}", "v")

        params = ledger.stitching_params(
            _context(_turn("t0"), _turn("t1", interrupted=True), _turn("t2")), "v"
        )

        assert params == {"previous_request_ids": ["r-t2"]}

    def test_a_turn_without_id_falls_back_to_its_text(self) -> None:
        params = RequestIdLedger().stitching_params(_context(_turn("t0", "Hello there.")), "v")

        assert params == {"previous_text": "Hello there."}

    def test_an_expired_id_falls_back_to_text(self) -> None:
        clock = _Clock()
        ledger = RequestIdLedger(clock=clock)
        ledger.record("s1", "t0", "r0", "v")
        clock.now += REQUEST_ID_TTL_S + 1

        params = ledger.stitching_params(_context(_turn("t0", "Old.")), "v")

        assert params == {"previous_text": "Old."}

    def test_user_turns_are_never_sent(self) -> None:
        user = ConversationTurn(turn_id="u", role="user", participant_id="p", text="secret")

        assert RequestIdLedger().stitching_params(_context(user), "v") == {}

    def test_forget_drops_the_session(self) -> None:
        ledger = RequestIdLedger()
        ledger.record("s1", "t0", "r0", "v")
        ledger.forget("s1")

        assert ledger.stitching_params(_context(_turn("t0", "x")), "v") == {"previous_text": "x"}

    def test_another_voice_ends_the_chain(self) -> None:
        ledger = RequestIdLedger()
        ledger.record("s1", "t0", "r0", "alice")
        ledger.record("s1", "t1", "r1", "bob")
        ledger.record("s1", "t2", "r2", "alice")
        context = _context(_turn("t0"), _turn("t1"), _turn("t2"))

        assert ledger.stitching_params(context, "alice") == {"previous_request_ids": ["r2"]}
        assert ledger.stitching_params(context, "bob") == {}

    def test_a_generation_that_never_became_a_turn_evicts_nothing(self) -> None:
        ledger = RequestIdLedger()
        ledger.record("s1", "t0", "r0", "v")
        for i in range(5):
            ledger.record("s1", f"unheard{i}", f"x{i}", "v")

        assert ledger.stitching_params(_context(_turn("t0")), "v") == {
            "previous_request_ids": ["r0"]
        }

    def test_request_id_header_is_case_insensitive(self) -> None:
        assert request_id_of({"Request-Id": "abc"}) == "abc"
        assert request_id_of({"content-type": "audio/mpeg"}) is None


class _RawResponse:
    def __init__(self, request_id: str, chunks: list[bytes]) -> None:
        self.headers = {"request-id": request_id}
        self._chunks = chunks

    @property
    def data(self) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            for chunk in self._chunks:
                yield chunk

        return gen()


def _provider(**config: Any) -> tuple[ElevenLabsTTSProvider, list[dict[str, Any]]]:
    provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", **config))
    calls: list[dict[str, Any]] = []
    client = MagicMock()
    counter = iter(range(100))

    @contextlib.asynccontextmanager
    async def raw_stream(**kwargs: Any) -> AsyncIterator[_RawResponse]:
        calls.append(kwargs)
        yield _RawResponse(f"req-{next(counter)}", [b"\x00\x01" * 1600] * 2)

    client.text_to_speech.with_raw_response.stream = raw_stream
    provider._client = client
    provider._make_voice_settings = MagicMock(return_value=None)  # type: ignore[method-assign]
    return provider, calls


async def _drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


class TestProvider:
    def test_level_is_self(self) -> None:
        provider, _ = _provider()
        assert provider.context_level == TTSContextLevel.SELF

    @pytest.mark.parametrize("config", [{"use_context": False}, {"expressive": True}])
    def test_level_is_none_when_off_or_v3(self, config: dict[str, Any]) -> None:
        provider, _ = _provider(**config)
        assert provider.context_level == TTSContextLevel.NONE

    async def test_the_second_call_carries_the_first_request_id(self) -> None:
        provider, calls = _provider()

        await _drain(provider.synthesize_stream("One.", context=_context(next_turn_id="t0")))
        await _drain(provider.synthesize_stream("Two.", context=_context(_turn("t0", "One."))))

        assert "previous_request_ids" not in calls[0]
        assert calls[1]["previous_request_ids"] == ["req-0"]
        assert "previous_text" not in calls[1]

    async def test_a_stream_closed_early_keeps_no_id(self) -> None:
        provider, calls = _provider()

        stream = provider.synthesize_stream("One.", context=_context(next_turn_id="t0"))
        await anext(stream)
        await stream.aclose()  # not read to the end: ElevenLabs could not continue from it
        await _drain(provider.synthesize_stream("Two.", context=_context(_turn("t0", "One."))))

        assert calls[1]["previous_text"] == "One."
        assert "previous_request_ids" not in calls[1]

    async def test_release_forgets_the_ids(self) -> None:
        provider, calls = _provider()
        await _drain(provider.synthesize_stream("One.", context=_context(next_turn_id="t0")))

        provider.release_context("s1")
        await _drain(provider.synthesize_stream("Two.", context=_context(_turn("t0", "One."))))

        assert calls[1]["previous_text"] == "One."


class TestThroughTheVoiceChannel:
    async def test_each_response_continues_the_previous_ones(self) -> None:
        from roomkit import RoomKit, VoiceChannel
        from roomkit.voice.backends.mock import MockVoiceBackend
        from roomkit.voice.base import VoiceCapability

        provider, calls = _provider(output_format="pcm_16000")
        backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
        channel = VoiceChannel("voice", tts=provider, backend=backend)
        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "voice")
        session = await kit.connect_voice("r1", "u", "voice")

        for text in ("One.", "Two.", "Three."):
            await channel.say(session, text)

        stitched = [c.get("previous_request_ids") for c in calls]
        assert stitched == [None, ["req-0"], ["req-0", "req-1"]]

        channel.unbind_session(session)
        assert (
            provider._request_ids.stitching_params(
                TTSContext(context_id=session.id, turns=(), next_turn_id="x"), "v"
            )
            == {}
        )
        await kit.close()

    async def test_after_a_barge_in_nothing_is_sent(self) -> None:
        import asyncio

        from roomkit import RoomKit, VoiceChannel
        from roomkit.voice.backends.mock import MockVoiceBackend
        from roomkit.voice.base import VoiceCapability

        provider, calls = _provider(output_format="pcm_16000")
        backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
        channel = VoiceChannel("voice", tts=provider, backend=backend)
        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "voice")
        session = await kit.connect_voice("r1", "u", "voice")

        slow = [b"\x00\x01" * 1600] * 10

        @contextlib.asynccontextmanager
        async def paced(**kwargs: Any) -> AsyncIterator[Any]:
            calls.append(kwargs)

            async def gen() -> AsyncIterator[bytes]:
                for chunk in slow:
                    await asyncio.sleep(0.03)
                    yield chunk

            response = MagicMock()
            response.headers = {"request-id": f"slow-{len(calls)}"}
            response.data = gen()
            yield response

        provider._client.text_to_speech.with_raw_response.stream = paced
        speaking = asyncio.create_task(channel.say(session, "A long answer."))
        await asyncio.sleep(0.1)
        await channel.interrupt(session, reason="barge_in")
        await speaking
        await channel.say(session, "Sorry, go ahead.")

        assert "previous_request_ids" not in calls[-1]
        assert "previous_text" not in calls[-1]
        await kit.close()
