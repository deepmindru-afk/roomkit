"""``read_skill_reference`` and ``run_skill_script`` on a skill the registry
does not offer are refused, on the AI channel as on a realtime session
(RMK-480, RFC §9.3), and so is ``activate_skill`` when its answer carries no
hint (no tool matches the name): nothing is revealed, the call named no skill.

The model reads the same error; ON_TOOL_CALL's observers hear a refusal, not
a served call whose body happens to be an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.executor import ScriptExecutor
from roomkit.skills.models import ScriptResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_toolset_edges import _call, _calling, _session, _skills, _text_turn

_CALLS = {
    "activate_skill": {"name": "ghost"},
    "read_skill_reference": {"skill_name": "ghost", "filename": "x.md"},
    "run_skill_script": {"skill_name": "ghost", "script_name": "x.sh"},
}
_ERROR = "Skill 'ghost' not found"
EVERY_SKILL_TOOL = pytest.mark.parametrize("tool", list(_CALLS))


class _Executor(ScriptExecutor):
    async def execute(self, skill: Any, script_name: str, arguments: Any = None) -> ScriptResult:
        raise AssertionError("no script runs")


def _audit(kit: RoomKit) -> list[Any]:
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        reports.append(event)

    return reports


@EVERY_SKILL_TOOL
async def test_the_ai_channel_refuses_it(tmp_path: Path, tool: str) -> None:
    provider = MockAIProvider(
        ai_responses=[_calling(tool, **_CALLS[tool]), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1", provider=provider, skills=_skills(tmp_path, "guide"), script_executor=_Executor()
    )
    kit = RoomKit()
    kit.register_channel(channel)
    reports = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")

    await _text_turn(channel)
    await kit.close()

    assert [e.refused for e in reports] == [True]
    assert _ERROR in str(reports[0].result)


@EVERY_SKILL_TOOL
async def test_a_realtime_session_refuses_it(tmp_path: Path, tool: str) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        skills=_skills(tmp_path, "guide"),
        script_executor=_Executor(),
    )
    kit, session = await _session(channel)
    reports = _audit(kit)

    read = await _call(channel, provider, session, tool, _CALLS[tool])
    await kit.close()

    assert _ERROR in read
    assert [e.refused for e in reports] == [True]
    assert _ERROR in str(reports[0].result)
