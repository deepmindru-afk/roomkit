"""Tests for all string enums."""

from __future__ import annotations

import pytest

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
)


class TestChannelType:
    def test_members(self) -> None:
        assert ChannelType.SMS == "sms"
        assert ChannelType.AI == "ai"
        assert ChannelType.VIDEO == "video"

    def test_count(self) -> None:
        # Includes MMS, RCS, REALTIME_VOICE, REALTIME_AUDIO_VIDEO, WHATSAPP_PERSONAL,
        # DISCORD, BUZZ, CONFERENCE, etc.
        assert len(ChannelType) == 23

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            ChannelType("invalid")


class TestChannelCategory:
    def test_values(self) -> None:
        assert ChannelCategory.TRANSPORT == "transport"
        assert ChannelCategory.INTELLIGENCE == "intelligence"

    def test_count(self) -> None:
        assert len(ChannelCategory) == 2


class TestChannelDirection:
    def test_values(self) -> None:
        assert ChannelDirection.BIDIRECTIONAL == "bidirectional"

    def test_count(self) -> None:
        assert len(ChannelDirection) == 3


class TestChannelMediaType:
    def test_values(self) -> None:
        assert ChannelMediaType.TEXT == "text"
        assert ChannelMediaType.RICH == "rich"
        assert ChannelMediaType.TEMPLATE == "template"

    def test_count(self) -> None:
        assert len(ChannelMediaType) == 7


class TestEventType:
    def test_message(self) -> None:
        assert EventType.MESSAGE == "message"

    def test_count(self) -> None:
        assert len(EventType) == 28

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            EventType("nope")


class TestEventStatus:
    def test_values(self) -> None:
        assert EventStatus.PENDING == "pending"
        assert EventStatus.BLOCKED == "blocked"

    def test_count(self) -> None:
        assert len(EventStatus) == 5


class TestAccess:
    def test_values(self) -> None:
        assert Access.READ_WRITE == "read_write"
        assert Access.NONE == "none"

    def test_count(self) -> None:
        assert len(Access) == 4


class TestIdentificationStatus:
    def test_values(self) -> None:
        assert IdentificationStatus.IDENTIFIED == "identified"
        assert IdentificationStatus.AMBIGUOUS == "ambiguous"
        assert IdentificationStatus.UNKNOWN == "unknown"
        assert IdentificationStatus.CHALLENGE_SENT == "challenge_sent"
        assert IdentificationStatus.REJECTED == "rejected"

    def test_count(self) -> None:
        assert len(IdentificationStatus) == 6


class TestParticipantRole:
    def test_values(self) -> None:
        assert ParticipantRole.OWNER == "owner"
        assert ParticipantRole.BOT == "bot"

    def test_count(self) -> None:
        assert len(ParticipantRole) == 5


class TestParticipantStatus:
    def test_values(self) -> None:
        assert ParticipantStatus.ACTIVE == "active"

    def test_count(self) -> None:
        assert len(ParticipantStatus) == 4


class TestRoomStatus:
    def test_values(self) -> None:
        assert RoomStatus.ACTIVE == "active"
        assert RoomStatus.CLOSED == "closed"

    def test_count(self) -> None:
        assert len(RoomStatus) == 4


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

    def test_count(self) -> None:
        # 82, then the three identity verification triggers (RFC §11.7).
        assert len(HookTrigger) == 85

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            HookTrigger("bad")


class TestHookExecution:
    def test_values(self) -> None:
        assert HookExecution.SYNC == "sync"
        assert HookExecution.ASYNC == "async"

    def test_count(self) -> None:
        assert len(HookExecution) == 2
