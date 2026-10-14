"""A combined audio/video backend's calls, from INVITE to BYE (RFC §12.8.3).

The backend's own lifecycle is the call's, with or without video; its
``video`` facet reports the video of the calls that negotiated it. A channel
reports each once: ``ON_SESSION_STARTED`` for the call,
``ON_VIDEO_SESSION_STARTED`` and ``ON_VIDEO_SESSION_ENDED`` for its video.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from roomkit import AudioVideoChannel, HookExecution, HookTrigger, RoomKit
from roomkit.video.backends.av import AudioVideoBackend
from roomkit.video.backends.base import VideoBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.test_sip_video_backend import (
    _FakeCall,
    _make_sdp_offer,
    _mock_aiosipua,
    _mock_rtp_bridge,
    _mock_video_bridge,
)

ROOM = "room-1"


def _sip_backend() -> Any:
    with (
        patch("roomkit.voice.backends.sip.import_aiosipua", return_value=_mock_aiosipua()),
        patch(
            "roomkit.voice.backends.sip.import_rtp_bridge",
            return_value=_mock_rtp_bridge(None),
        ),
        patch(
            "roomkit.video.backends.sip._import_video_bridge",
            return_value=_mock_video_bridge(),
        ),
    ):
        from roomkit.video.backends.sip import SIPVideoBackend

        return SIPVideoBackend(
            local_sip_addr=("127.0.0.1", 5060),
            local_rtp_ip="10.0.0.5",
            rtp_port_start=10000,
            rtp_port_end=10100,
        )


async def _room_answering_calls() -> tuple[RoomKit, AudioVideoChannel, Any, dict[str, list]]:
    backend = _sip_backend()
    kit = RoomKit()
    av = AudioVideoChannel("av", stt=MockSTTProvider(), tts=MockTTSProvider(), backend=backend)
    kit.register_channel(av)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "av")

    fired: dict[str, list] = {"call": [], "video_started": [], "video_ended": []}
    for trigger, key in (
        (HookTrigger.ON_SESSION_STARTED, "call"),
        (HookTrigger.ON_VIDEO_SESSION_STARTED, "video_started"),
        (HookTrigger.ON_VIDEO_SESSION_ENDED, "video_ended"),
    ):

        async def record(event: Any, ctx: Any, key: str = key) -> None:
            fired[key].append(event.session.id)

        kit.hook(trigger, execution=HookExecution.ASYNC)(record)

    # Joined after the INVITE was answered: the call is reported live before
    # it is bound, as on a real SIP call.
    @backend.on_call
    async def join(session: Any) -> None:
        await kit.join(ROOM, "av", session=session)

    return kit, av, backend, fired


async def _invite(backend: Any, *, has_video: bool) -> _FakeCall:
    call = _FakeCall(sdp_offer=_make_sdp_offer(has_video=has_video))
    await backend._handle_invite(call)
    await asyncio.sleep(0.05)
    return call


async def _bye(backend: Any, call: _FakeCall) -> None:
    backend._handle_bye(call, object())
    await asyncio.sleep(0.05)


def test_an_av_backend_is_a_voice_backend_whose_video_is_a_facet() -> None:
    backend = _sip_backend()
    assert isinstance(backend, AudioVideoBackend)
    assert not isinstance(backend, VideoBackend)
    assert isinstance(backend.video, VideoBackend)


async def test_an_audio_only_call_starts_and_ends_without_video() -> None:
    kit, av, backend, fired = await _room_answering_calls()
    call = await _invite(backend, has_video=False)
    assert fired["call"] == ["session-1"]
    assert "session-1" in av._session_bindings

    await _bye(backend, call)

    # The BYE unbinds the call: binding, pipeline and STT go with it.
    assert "session-1" not in av._session_bindings
    assert fired["video_started"] == fired["video_ended"] == []
    await kit.close()


async def test_a_call_with_video_reports_the_call_and_its_video_once_each() -> None:
    kit, av, backend, fired = await _room_answering_calls()
    call = await _invite(backend, has_video=True)
    assert fired["call"] == ["session-1"]
    assert fired["video_started"] == ["session-1"]
    assert backend.video.get_session("session-1") is not None

    await _bye(backend, call)

    assert fired["video_ended"] == ["session-1"]
    assert "session-1" not in av._session_bindings
    assert backend.video.get_session("session-1") is None
    assert fired["call"] == ["session-1"] and fired["video_started"] == ["session-1"]
    await kit.close()


async def test_a_call_the_channel_ends_reports_its_video_ended_once() -> None:
    kit, av, backend, fired = await _room_answering_calls()
    await _invite(backend, has_video=True)
    session = backend.get_session("session-1")

    await av.disconnect_session(session, ROOM)
    await asyncio.sleep(0.05)

    assert fired["video_ended"] == ["session-1"]
    assert "session-1" not in av._session_bindings
    await kit.close()


@pytest.mark.parametrize("has_video", [False, True])
async def test_the_realtime_bridge_lets_a_call_go_on_its_bye(has_video: bool) -> None:
    from roomkit.voice.realtime.bridge import RealtimeAVBridge

    backend = _sip_backend()
    bridge = RealtimeAVBridge(MockRealtimeProvider(), backend)
    call = await _invite(backend, has_video=has_video)
    assert "session-1" in bridge._calls

    await _bye(backend, call)

    assert bridge._calls == {}
    await bridge.close()


async def test_the_video_api_left_on_the_backend_warns_and_reaches_the_facet() -> None:
    backend = _sip_backend()
    await _invite(backend, has_video=True)
    frames: list[Any] = []

    with pytest.warns(DeprecationWarning, match=r"\.video\.on_video_received\(\)"):
        backend.on_video_received(lambda _s, frame: frames.append(frame))
    with pytest.warns(DeprecationWarning, match=r"get_video_session\(\).*\.video\.get_session"):
        video_session = backend.get_video_session("session-1")
    with pytest.warns(DeprecationWarning, match=r"\.video\.list_sessions\(\)"):
        assert backend.list_video_sessions("room-1") == [video_session]
    with pytest.warns(DeprecationWarning, match=r"\.video\.capabilities"):
        assert backend.video_capabilities == backend.video.capabilities

    assert video_session is backend.video.get_session("session-1")
    assert backend.video._received_callback is not None
