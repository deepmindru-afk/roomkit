"""Vui's KV cache follows the TTS conversation context (RFC §12.2.2, AUDIO).

The session is exercised against a fake cache: no GPU and no ``vui-tts``. Each
test states what the cache must hold after a call, given what the user heard.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.tts._vui_session import FRAME_MS, VuiConversation
from roomkit.voice.tts.context import ConversationTurn, TTSContext

_AUDIO = AudioFrame(data=b"\x01\x00" * 1600, sample_rate=16000)


class FakeCache:
    """Records the operations; one KV position per text and per frame.

    Like Vui, a frame's codes enter the cache with the next decoding step:
    after the k-th frame is yielded, ``offset`` covers frames 0..k-1.
    """

    def __init__(
        self, frames_per_reply: int = 10, capacity: int = 100_000, audio_capacity: int = 100_000
    ) -> None:
        self.offset = 0
        self.capacity = capacity
        self.audio_capacity = audio_capacity
        self.prompt_frames = 50
        self.reply_positions = frames_per_reply
        self.frames_per_reply = frames_per_reply
        self.log: list[tuple[str, object]] = []

    def restart(self, voice: str) -> None:
        self.offset = 100  # the prompt
        self.log.append(("restart", voice))

    def reset(self) -> None:
        self.offset = 0
        self.log.append(("reset", None))

    def truncate(self, offset: int) -> None:
        self.offset = offset
        self.log.append(("truncate", offset))

    def add_user(self, text: str, audio: AudioFrame | None) -> None:
        self.offset += 5
        self.log.append(("user", (text, audio is not None)))

    def generate(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        self.offset += 3  # [spk] + text
        self.log.append(("generate", text))
        for i in range(self.frames_per_reply):
            if cancel.is_set():
                return
            if i:
                self.offset += 1  # the previous frame's codes enter the cache
            yield b"\x00\x00"


def _user(
    turn_id: str, text: str = "question", audio: AudioFrame | None = _AUDIO
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id, role="user", participant_id="u", text=text, audio=audio
    )


def _agent(turn_id: str, played_ms: int, *, interrupted: bool = False) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        role="assistant",
        participant_id="ai",
        text="reply",
        played_ms=played_ms,
        interrupted=interrupted,
    )


def _ctx(*turns: ConversationTurn, next_turn_id: str, context_id: str = "s1") -> TTSContext:
    return TTSContext(context_id=context_id, turns=turns, next_turn_id=next_turn_id)


def _speak(conv: VuiConversation, context: TTSContext | None, text: str = "hi") -> int:
    return len(list(conv.speak(context, "maeve", text, threading.Event())))


class TestFirstCall:
    def test_restarts_from_the_prompt_with_the_question_being_answered(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)

        _speak(conv, _ctx(_user("u1", "hello"), next_turn_id="a1"))

        assert cache.log[:3] == [
            ("restart", "maeve"),
            ("user", ("hello", True)),
            ("generate", "hi"),
        ]

    def test_without_context_every_call_starts_afresh(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)

        _speak(conv, None)
        _speak(conv, None)

        assert [op for op, _ in cache.log] == ["restart", "generate", "restart", "generate"]


class TestFollowingTurns:
    def test_a_reply_heard_whole_stays_and_the_next_question_is_added(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))
        after_reply = cache.offset

        _speak(
            conv,
            _ctx(_user("u1"), _agent("a1", 800), _user("u2", "and then"), next_turn_id="a2"),
        )

        assert ("truncate", after_reply) not in cache.log
        assert cache.log[3] == ("user", ("and then", True))

    def test_a_cut_off_reply_is_cut_back_to_what_was_heard(self) -> None:
        cache = FakeCache(frames_per_reply=10)
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))
        reply_start = 100 + 5  # prompt + u1

        heard = 3
        _speak(
            conv,
            _ctx(
                _user("u1"),
                _agent("a1", int(heard * FRAME_MS), interrupted=True),
                _user("u2"),
                next_turn_id="a2",
            ),
        )

        # [spk]+text (3 positions), then the 3 frames the user heard
        assert ("truncate", reply_start + 3 + heard) in cache.log

    def test_a_reply_nobody_heard_is_dropped(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        _speak(conv, _ctx(_user("u1"), _user("u2"), next_turn_id="a2"))

        assert ("truncate", 105) in cache.log

    def test_a_turn_without_audio_is_written_as_text(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        _speak(
            conv, _ctx(_user("u1"), _agent("a1", 800), _user("u2", audio=None), next_turn_id="a2")
        )

        assert ("user", ("question", False)) in cache.log


class TestSwitching:
    @pytest.mark.parametrize(("context_id", "voice"), [("s2", "maeve"), ("s1", "abraham")])
    def test_another_session_or_voice_restarts(self, context_id: str, voice: str) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        list(
            conv.speak(
                _ctx(_user("v1"), next_turn_id="b1", context_id=context_id),
                voice,
                "hi",
                threading.Event(),
            )
        )

        assert [op for op, _ in cache.log].count("restart") == 2

    def test_release_makes_the_next_call_restart(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        conv.release("s1")
        _speak(conv, _ctx(_user("u1"), _agent("a1", 800), _user("u2"), next_turn_id="a2"))

        assert [op for op, _ in cache.log].count("restart") == 2
        assert conv.context_id == "s1"

    def test_a_history_that_fell_out_of_the_window_restarts(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        _speak(conv, _ctx(_user("x"), _agent("y", 800), _user("z"), next_turn_id="a9"))

        assert [op for op, _ in cache.log].count("restart") == 2


class TestRelease:
    def test_release_empties_the_cache(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        conv.release("s1")

        assert cache.log[-1] == ("reset", None)
        assert cache.offset == 0

    def test_release_during_a_reply_empties_the_cache_when_it_ends(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        frames = conv.speak(_ctx(_user("u1"), next_turn_id="a1"), "maeve", "hi", threading.Event())
        next(frames)

        conv.release("s1")
        assert ("reset", None) not in cache.log
        list(frames)

        assert cache.log[-1] == ("reset", None)

    def test_releasing_another_context_leaves_the_cache(self) -> None:
        cache = FakeCache()
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))

        conv.release("s2")

        assert ("reset", None) not in cache.log


class TestCapacity:
    def test_a_cache_about_to_overflow_restarts_before_the_reply(self) -> None:
        cache = FakeCache(frames_per_reply=10, capacity=500)
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1"), next_turn_id="a1"))
        cache.offset = 400  # a long conversation

        _speak(conv, _ctx(_user("u1"), _agent("a1", 800), _user("u2"), next_turn_id="a2"))

        assert [op for op, _ in cache.log].count("restart") == 2
        assert cache.offset < cache.capacity

    def test_a_user_turn_longer_than_the_cache_keeps_its_words(self) -> None:
        cache = FakeCache(capacity=400)
        conv = VuiConversation(cache)
        minute = AudioFrame(data=b"\x01\x00" * 16000 * 60, sample_rate=16000)

        _speak(conv, _ctx(_user("u1", "long story", audio=minute), next_turn_id="a1"))

        assert ("user", ("long story", False)) in cache.log


class TestAudioBudget:
    def test_six_minutes_of_audio_restart_the_cache_before_the_kv_is_full(self) -> None:
        """The KV could hold more, but past the trained audio length the model is
        outside anything it learned from: the cache restarts."""
        cache = FakeCache(frames_per_reply=10, audio_capacity=100)
        conv = VuiConversation(cache)
        second = AudioFrame(data=b"\x01\x00" * 16000, sample_rate=16000)  # 12.5 frames
        _speak(conv, _ctx(_user("u1", audio=second), next_turn_id="a1"))  # 50 + 13 + 10

        _speak(
            conv,
            _ctx(
                _user("u1", audio=second),
                _agent("a1", 800),
                _user("u2", audio=second),
                _agent("a2", 800),  # not ours: ignored
                _user("u3", audio=second),
                next_turn_id="a3",
            ),
        )

        assert [op for op, _ in cache.log].count("restart") == 2
        assert cache.offset < cache.capacity

    def test_a_cut_reply_gives_back_the_frames_nobody_heard(self) -> None:
        cache = FakeCache(frames_per_reply=40, audio_capacity=130)
        conv = VuiConversation(cache)
        _speak(conv, _ctx(_user("u1", audio=None), next_turn_id="a1"))  # 50 + 40 = 90

        # Only 5 frames heard: 90 - 35 = 55, room for the next 40-frame reply.
        _speak(
            conv,
            _ctx(
                _user("u1", audio=None),
                _agent("a1", int(5 * FRAME_MS), interrupted=True),
                _user("u2", audio=None),
                next_turn_id="a2",
            ),
        )

        assert [op for op, _ in cache.log].count("restart") == 1


class TestCancel:
    def test_a_cancelled_generation_stops_and_is_settled_next_call(self) -> None:
        cache = FakeCache(frames_per_reply=50)
        conv = VuiConversation(cache)
        cancel = threading.Event()
        frames = conv.speak(_ctx(_user("u1"), next_turn_id="a1"), "maeve", "long", cancel)
        for _ in range(3):
            next(frames)
        cancel.set()
        assert list(frames) == []

        _speak(
            conv,
            _ctx(
                _user("u1"),
                _agent("a1", int(FRAME_MS), interrupted=True),
                _user("u2"),
                next_turn_id="a2",
            ),
        )

        assert ("truncate", 105 + 3 + 1) in cache.log


class TestVuiPrivateApi:
    def test_the_private_accesses_still_exist(self) -> None:
        """Fails when vui-tts drops what the provider relies on (RMK-197)."""
        import inspect

        engine_mod = pytest.importorskip("vui.engine")
        codec_mod = pytest.importorskip("vui.qwen_codec")

        assert hasattr(engine_mod.Engine, "_rewind_row")
        assert "_spk_token" in inspect.getsource(engine_mod.Row.__init__)
        assert "_codec_ctx" in inspect.getsource(engine_mod.Row.__init__)
        assert hasattr(codec_mod.CodecCtx, "prefill")
        # our seed survives only if Vui seeds when the state is closed
        assert "_stack is None" in inspect.getsource(engine_mod.Engine._stream_row)
        assert "_buf" in inspect.getsource(codec_mod.CodecCtx.__init__)
