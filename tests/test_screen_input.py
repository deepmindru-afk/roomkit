"""Tests for screen_input.py: _parse_json_response, _build_press_key_tool, _get_scale_factor."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.screen_input import (
    _LOCATE_SCHEMA,
    _build_press_key_tool,
    _get_scale_factor,
    _locate,
    _parse_json_response,
)

# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_clean() -> None:
    raw = (
        '{"found": true, "cx": 100, "cy": 200,'
        ' "box": {"x1": 90, "y1": 190, "x2": 110, "y2": 210},'
        ' "label": "OK"}'
    )
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is True
    assert result["cx"] == 100
    assert result["label"] == "OK"


def test_parse_json_embedded_in_text() -> None:
    raw = (
        'Here is the result: {"found": true, "cx": 50,'
        ' "cy": 60, "box": {}, "label": "btn"} extra text'
    )
    result = _parse_json_response(raw)
    assert result is not None
    assert result["cx"] == 50


def test_parse_json_fallback_with_label() -> None:
    raw = 'garbled "found": true, "cx": 42, "cy": 99, broken json "label": "Search"'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is True
    assert result["cx"] == 42
    assert result["cy"] == 99
    assert result["label"] == "Search"


def test_parse_json_fallback_without_label() -> None:
    raw = '"found": false, "cx": 0, "cy": 0'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is False
    assert result["label"] == ""


def test_parse_json_garbage() -> None:
    result = _parse_json_response("totally random text with no json")
    assert result is None


# ---------------------------------------------------------------------------
# _build_press_key_tool
# ---------------------------------------------------------------------------


def test_build_press_key_tool_darwin() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        tool = _build_press_key_tool()
    assert tool["name"] == "press_key"
    assert "command" in tool["description"]


def test_build_press_key_tool_linux() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        tool = _build_press_key_tool()
    assert tool["name"] == "press_key"
    assert "ctrl" in tool["description"]


# ---------------------------------------------------------------------------
# _get_scale_factor DPI guard
# ---------------------------------------------------------------------------


def test_dpi_guard_called_once_on_windows() -> None:
    """SetProcessDpiAwareness should only be called once."""
    import roomkit.video.vision.screen_input as mod

    original = mod._dpi_initialized
    try:
        mod._dpi_initialized = False
        with patch(
            "roomkit.video.vision.screen_input.platform",
        ) as mock_platform:
            mock_platform.system.return_value = "Windows"
            with patch(
                "roomkit.video.vision.screen_input.ctypes",
                create=True,
            ):
                _get_scale_factor()
                assert mod._dpi_initialized is True

                # Second call — guard prevents re-calling SetProcessDpiAwareness
                _get_scale_factor()
                assert mod._dpi_initialized is True
    finally:
        mod._dpi_initialized = original


def test_scale_factor_linux_with_gdk_scale() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        with patch.dict("os.environ", {"GDK_SCALE": "2"}):
            sx, sy = _get_scale_factor()
    assert sx == 0.5
    assert sy == 0.5


# ---------------------------------------------------------------------------
# _locate: a provider that constrains its output is read as it is
# ---------------------------------------------------------------------------

_FRAME = VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)
_FOUND = json.dumps(
    {
        "found": True,
        "cx": 10,
        "cy": 12,
        "box": {"x1": 5, "y1": 6, "x2": 15, "y2": 18},
        "label": "OK",
    }
)


async def test_locate_passes_the_schema_to_a_provider_that_takes_one() -> None:
    vision = MockVisionProvider([_FOUND], response_schema=True)

    answer = await _locate(vision, _FRAME, "find OK")

    assert answer is not None and answer["label"] == "OK" and answer["cx"] == 10


async def test_locate_reports_no_answer_when_the_constrained_one_is_not_the_shape() -> None:
    vision = MockVisionProvider(["an OK button, top left"], response_schema=True)

    assert await _locate(vision, _FRAME, "find OK") is None


async def test_the_mock_spends_a_description_that_fails_the_check() -> None:
    vision = MockVisionProvider(["not a document", _FOUND], response_schema=True)

    with pytest.raises(ResponseSchemaError):
        await vision.analyze_frame(_FRAME, response_schema=_LOCATE_SCHEMA)
    answer = await vision.analyze_frame(_FRAME, response_schema=_LOCATE_SCHEMA)

    assert json.loads(answer.description)["cx"] == 10


async def test_locate_logs_a_free_answer_it_cannot_read(caplog: pytest.LogCaptureFixture) -> None:
    vision = MockVisionProvider(["I see no such button"])

    with caplog.at_level("WARNING", logger="roomkit.video.vision.screen_input"):
        assert await _locate(vision, _FRAME, "find OK") is None

    assert "I see no such button" in caplog.text


async def test_locate_repairs_a_free_answer_from_a_provider_without_schemas() -> None:
    vision = MockVisionProvider(['Here: {"found": true, "cx": 3, "cy": 4, "label": "OK"}'])

    answer = await _locate(vision, _FRAME, "find OK")

    assert answer is not None and answer["cx"] == 3
