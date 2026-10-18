"""The voice frame-cost benchmark measures what it says (RMK-796)."""

from __future__ import annotations

import pytest
from benchmarks.voice import __main__ as cli
from benchmarks.voice.frame_cost import measure


async def test_each_second_of_tone_is_one_turn_per_call() -> None:
    cost = await measure(calls=2, seconds=4)  # tone, silence, tone, silence

    assert (cost.calls, cost.frames, cost.stt_requests) == (2, 400, 4)
    assert cost.cpu_seconds > 0 and cost.us_per_frame > 0


async def test_without_a_vad_no_turn_reaches_a_batch_stt() -> None:
    cost = await measure(calls=1, seconds=2, vad=False)

    assert (cost.frames, cost.stt_requests) == (100, 0)


@pytest.mark.parametrize("argv", [["--calls", "0"], ["--seconds", "-1"], ["--vad", "silero"]])
def test_cli_refuses_what_it_cannot_measure(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
