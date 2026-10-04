"""``read_skill_reference`` on a skill the registry does not offer is
refused, on the AI channel as on a realtime session (RMK-480, RFC §9.3).

The model reads the same error; ON_TOOL_CALL's observers hear a refusal, not
a served call whose body happens to be an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_toolset_edges import _call, _calling, _session, _skills, _text_turn

_GHOST = {"skill_name": "ghost", "filename": "x.md"}
_ERROR = '{"error": "Skill \'ghost\' not found"}'


def _audit(kit: RoomKit) -> list[Any]:
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        reports.append(event)

    return reports


async def test_the_ai_channel_refuses_it(tmp_path: Path) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _calling("read_skill_reference", **_GHOST),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel("ai1", provider=provider, skills=_skills(tmp_path, "guide"))
    kit = RoomKit()
    kit.register_channel(channel)
    reports = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")

    await _text_turn(channel)
    await kit.close()

    assert [(e.refused, e.result) for e in reports] == [(True, _ERROR)]


async def test_a_realtime_session_refuses_it(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        skills=_skills(tmp_path, "guide"),
    )
    kit, session = await _session(channel)
    reports = _audit(kit)

    read = await _call(channel, provider, session, "read_skill_reference", _GHOST)
    await kit.close()

    assert read == _ERROR
    assert [(e.refused, e.result) for e in reports] == [(True, _ERROR)]
