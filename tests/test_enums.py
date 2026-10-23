"""Tests for all string enums."""

from __future__ import annotations

from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    ParticipantRole,
    ParticipantStatus,
    RoomStatus,
    TaskStatus,
)


class TestChannelType:
    def test_members(self) -> None:
        assert ChannelType.SMS == "sms"
        assert ChannelType.AI == "ai"
        assert ChannelType.VIDEO == "video"


class TestChannelCategory:
    def test_values(self) -> None:
        assert ChannelCategory.TRANSPORT == "transport"
        assert ChannelCategory.INTELLIGENCE == "intelligence"


class TestChannelDirection:
    def test_values(self) -> None:
        assert ChannelDirection.BIDIRECTIONAL == "bidirectional"


class TestChannelMediaType:
    def test_values(self) -> None:
        assert ChannelMediaType.TEXT == "text"
        assert ChannelMediaType.RICH == "rich"
        assert ChannelMediaType.TEMPLATE == "template"


class TestEventType:
    def test_message(self) -> None:
        assert EventType.MESSAGE == "message"


class TestEventStatus:
    def test_values(self) -> None:
        assert EventStatus.PENDING == "pending"
        assert EventStatus.BLOCKED == "blocked"


class TestAccess:
    def test_values(self) -> None:
        assert Access.READ_WRITE == "read_write"
        assert Access.NONE == "none"


class TestIdentificationStatus:
    def test_values(self) -> None:
        assert IdentificationStatus.IDENTIFIED == "identified"
        assert IdentificationStatus.AMBIGUOUS == "ambiguous"
        assert IdentificationStatus.UNKNOWN == "unknown"
        assert IdentificationStatus.CHALLENGE_SENT == "challenge_sent"
        assert IdentificationStatus.REJECTED == "rejected"


class TestParticipantRole:
    def test_values(self) -> None:
        assert ParticipantRole.OWNER == "owner"
        assert ParticipantRole.BOT == "bot"


class TestParticipantStatus:
    def test_values(self) -> None:
        assert ParticipantStatus.ACTIVE == "active"


class TestRoomStatus:
    def test_values(self) -> None:
        assert RoomStatus.ACTIVE == "active"
        assert RoomStatus.CLOSED == "closed"


class TestHookTrigger:
    def test_values(self) -> None:
        assert HookTrigger.BEFORE_BROADCAST == "before_broadcast"
        assert HookTrigger.ON_TASK_CREATED == "on_task_created"
        assert HookTrigger.ON_DELIVERY_STATUS == "on_delivery_status"
        assert HookTrigger.ON_EVENT_UPDATED == "on_event_updated"
        assert HookTrigger.ON_EVENT_DELETED == "on_event_deleted"
        assert HookTrigger.ON_VIDEO_SESSION_STARTED == "on_video_session_started"
        assert HookTrigger.ON_VIDEO_SESSION_ENDED == "on_video_session_ended"
        assert HookTrigger.ON_VIDEO_TRACK_ADDED == "on_video_track_added"
        assert HookTrigger.ON_VIDEO_TRACK_REMOVED == "on_video_track_removed"
        assert HookTrigger.ON_SCREEN_SHARE_STARTED == "on_screen_share_started"
        assert HookTrigger.ON_SCREEN_SHARE_STOPPED == "on_screen_share_stopped"
        assert HookTrigger.ON_PROTOCOL_TRACE == "on_protocol_trace"
        assert HookTrigger.ON_ROOM_PAUSED == "on_room_paused"
        assert HookTrigger.ON_REALTIME_DELEGATION == "on_realtime_delegation"
        assert HookTrigger.ON_SPEECH_START == "on_speech_start"
        assert HookTrigger.ON_SPEECH_END == "on_speech_end"
        assert HookTrigger.ON_TRANSCRIPTION == "on_transcription"
        assert HookTrigger.ON_PARTIAL_TRANSCRIPTION == "on_partial_transcription"
        assert HookTrigger.BEFORE_TTS == "before_tts"
        assert HookTrigger.AFTER_TTS == "after_tts"
        assert HookTrigger.ON_TTS_CANCELLED == "on_tts_cancelled"
        assert HookTrigger.ON_BARGE_IN == "on_barge_in"
        assert HookTrigger.ON_VAD_SILENCE == "on_vad_silence"
        assert HookTrigger.ON_VAD_AUDIO_LEVEL == "on_vad_audio_level"
        assert HookTrigger.ON_SPEAKER_CHANGE == "on_speaker_change"


class TestHookExecution:
    def test_values(self) -> None:
        assert HookExecution.SYNC == "sync"
        assert HookExecution.ASYNC == "async"


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
