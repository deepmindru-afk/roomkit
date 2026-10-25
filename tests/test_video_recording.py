"""Tests for video recording — VideoRecorder ABC, Mock, and channel integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    RoomKit,
    VideoChannel,
)
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.base import VideoSession
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.recorder import (
    MockVideoRecorder,
    VideoRecorder,
    VideoRecordingConfig,
)
from roomkit.video.recorder import opencv as opencv_recorder
from roomkit.video.recorder.opencv import OpenCVVideoRecorder
from roomkit.video.recorder.pyav import PyAVVideoRecorder
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.pipeline.recorder.base import RecordingEncryption


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestMockVideoRecorder:
    def test_start_stop(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        config = VideoRecordingConfig(storage="/tmp/test")

        handle = recorder.start(session, config)
        assert handle.state == "recording"
        assert handle.session_id == "s1"
        assert handle.id

        result = recorder.stop(handle)
        assert handle.state == "stopped"
        assert result.frame_count == 0
        assert result.id == handle.id

    def test_tap_frame(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        handle = recorder.start(session, VideoRecordingConfig())

        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        recorder.tap_frame(handle, frame)
        recorder.tap_frame(handle, frame)

        result = recorder.stop(handle)
        assert result.frame_count == 2

    def test_multiple_recordings(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        s1 = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        s2 = VideoSession(id="s2", room_id="r1", participant_id="u2", channel_id="v1")

        h1 = recorder.start(s1, VideoRecordingConfig())
        h2 = recorder.start(s2, VideoRecordingConfig())

        assert h1.id != h2.id
        assert len(recorder.recordings) == 2


class TestVideoChannelRecording:
    async def test_recording_starts_on_bind(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Recording should have started
        assert session.id in ch._recording_handles
        assert len(recorder.recordings) == 1

    async def test_recording_stops_on_unbind(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        await kit.disconnect_video(session)

        assert session.id not in ch._recording_handles
        # Recording was stopped
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"

    async def test_frames_tapped_to_recorder(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Send frames
        for i in range(5):
            frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=float(i * 100))
            await backend.simulate_video_received(session, frame)

        rec = list(recorder.recordings.values())[0]
        assert len(rec.frames) == 5

    async def test_no_recorder_no_error(self, kit: RoomKit) -> None:
        """Channel works fine without a recorder."""
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        await backend.simulate_video_received(session, frame)
        await kit.disconnect_video(session)

    async def test_close_stops_recordings(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")
        await ch.close()

        assert ch._recording_handles == {}
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"


# -- Encryption at rest (RFC §17.6, RMK-568) ---------------------------------

_CIPHER_HEADER = b"XOR1"


class _XorEncryption(RecordingEncryption):
    """Stands for the integrator's cipher: writes ``<path>.enc``, deletes the plaintext."""

    def __init__(self) -> None:
        self.plaintexts: list[bytes] = []

    @property
    def name(self) -> str:
        return "xor"

    def encrypt_file(self, path: str) -> str:
        source = Path(path)
        plaintext = source.read_bytes()
        self.plaintexts.append(plaintext)
        target = source.with_name(source.name + ".enc")
        target.write_bytes(_CIPHER_HEADER + bytes(b ^ 0x5A for b in plaintext))
        source.unlink()
        return str(target)


class _FileVideoWriter:
    """``cv2.VideoWriter`` as the recorder drives it: each frame's bytes go to the file."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "wb")  # noqa: SIM115 - released by release()

    def isOpened(self) -> bool:  # noqa: N802
        return not self._file.closed

    def write(self, frame: Any) -> None:
        self._file.write(frame.tobytes())

    def release(self) -> None:
        self._file.close()


class _FileWritingCV2:
    """The part of ``cv2`` the OpenCV recorder uses, writing real files."""

    COLOR_RGB2BGR = 4

    def VideoWriter_fourcc(self, *args: str) -> int:  # noqa: N802
        return 0

    def VideoWriter(  # noqa: N802
        self, path: str, fourcc: int, fps: float, size: tuple[int, int]
    ) -> _FileVideoWriter:
        return _FileVideoWriter(path)

    def cvtColor(self, src: Any, code: int) -> Any:  # noqa: N802
        return src


@pytest.fixture(params=["opencv", "pyav"])
def file_recorder(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[VideoRecorder, str]:
    """A recorder that writes files, and the codec to ask it for."""
    if request.param == "pyav":
        pytest.importorskip("av", reason="av (PyAV) not installed")
        return PyAVVideoRecorder(), "libx264"
    monkeypatch.setattr(opencv_recorder, "_import_cv2", _FileWritingCV2)
    return OpenCVVideoRecorder(), "mp4v"


def _raw_frame(value: int) -> VideoFrame:
    return VideoFrame(data=bytes([value]) * 64 * 48 * 3, codec="raw_rgb24", width=64, height=48)


def _video_session(sid: str) -> VideoSession:
    return VideoSession(id=sid, room_id="r1", participant_id="u1", channel_id="video-1")


class TestRecordingEncryptionAtRest:
    async def test_session_video_configured_for_encryption_is_stored_as_ciphertext(
        self, file_recorder: tuple[VideoRecorder, str], tmp_path: Path
    ) -> None:
        """A video channel's recording is handed to ``encryption`` when it ends,
        and only the cipher's artifact stays on the storage (RFC §17.6)."""
        recorder, codec = file_recorder
        cipher = _XorEncryption()
        config = VideoRecordingConfig(storage=str(tmp_path), codec=codec, encryption=cipher)
        kit, backend = RoomKit(), MockVideoBackend()
        kit.register_channel(
            VideoChannel(
                "video-1",
                backend=backend,
                pipeline=VideoPipelineConfig(recorder=recorder, recording_config=config),
            )
        )
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.join("r1", "video-1", participant_id="u1")
        for i in range(3):
            await backend.simulate_video_received(session, _raw_frame(i))
        await kit.leave(session)
        await kit.close()

        stored = sorted(tmp_path.iterdir())
        assert [p.suffix for p in stored] == [".enc"]
        assert len(cipher.plaintexts) == 1 and cipher.plaintexts[0]
        plaintext = cipher.plaintexts[0]
        assert stored[0].read_bytes() == _CIPHER_HEADER + bytes(b ^ 0x5A for b in plaintext)

    def test_result_names_the_ciphertext_and_close_encrypts_an_open_recording(
        self, file_recorder: tuple[VideoRecorder, str], tmp_path: Path
    ) -> None:
        """``stop()`` reports the encrypted artifact; ``close()`` encrypts a
        recording still open, which would otherwise stay in the clear."""
        recorder, codec = file_recorder
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, encryption=_XorEncryption()
        )
        stopped = recorder.start(_video_session("stopped-session"), config)
        left_open = recorder.start(_video_session("open-session"), config)
        for handle in (stopped, left_open):
            recorder.tap_frame(handle, _raw_frame(7))

        result = recorder.stop(stopped)
        recorder.close()

        assert result.url == stopped.path + ".enc"
        assert result.size_bytes == Path(result.url).stat().st_size
        assert result.metadata == {"encryption": "xor"}
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            Path(h.path).name + ".enc" for h in (stopped, left_open)
        )

    @pytest.mark.parametrize(
        ("storage_encrypted_at_rest", "metadata"),
        [(False, {}), (True, {"encryption": "storage"})],
    )
    def test_without_encryption_the_file_stays_as_written(
        self,
        file_recorder: tuple[VideoRecorder, str],
        tmp_path: Path,
        storage_encrypted_at_rest: bool,
        metadata: dict[str, object],
    ) -> None:
        """No ``encryption``: the recording is stored as it was written, and the
        result claims encryption only when the storage was declared to apply it."""
        recorder, codec = file_recorder
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, storage_encrypted_at_rest=storage_encrypted_at_rest
        )
        handle = recorder.start(_video_session("s1"), config)
        recorder.tap_frame(handle, _raw_frame(7))

        result = recorder.stop(handle)
        recorder.close()

        assert result.url == handle.path
        assert [str(p) for p in tmp_path.iterdir()] == [handle.path]
        assert result.size_bytes == Path(handle.path).stat().st_size > 0
        assert result.metadata == metadata
