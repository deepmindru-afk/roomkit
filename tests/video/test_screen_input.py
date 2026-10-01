"""Tests for screen input: clipboard paste and pyautogui's failsafe."""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

import pytest

from roomkit.video.vision.screen_input import ScreenInputTools, _clipboard_paste, _get_pyautogui


class TestClipboardPaste:
    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_macos_uses_pbcopy(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("hello world")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["pbcopy"]
        assert call_args[1]["input"] == b"hello world"
        mock_pag.hotkey.assert_called_once_with("command", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_linux_uses_xclip(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Linux"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("test text")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["xclip", "-selection", "clipboard"]
        mock_pag.hotkey.assert_called_once_with("ctrl", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_windows_uses_clip(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Windows"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("windows text")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["clip"]
        mock_pag.hotkey.assert_called_once_with("ctrl", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_fallback_to_typewrite(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        """Should fall back to typewrite when clipboard fails."""
        import subprocess as real_subprocess

        mock_platform.system.return_value = "Darwin"
        mock_subprocess.run.side_effect = real_subprocess.SubprocessError("fail")
        mock_subprocess.SubprocessError = real_subprocess.SubprocessError
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("fallback text")

        mock_pag.typewrite.assert_called_once_with("fallback text", interval=0.02)


class _FailSafeError(Exception):
    """Stands in for pyautogui.FailSafeException."""


def _fake_pyautogui(press_error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("pyautogui")
    module.FAILSAFE = True
    module.FailSafeException = _FailSafeError
    module.press = MagicMock(side_effect=press_error)
    return module


class TestFailsafe:
    def test_import_leaves_the_failsafe_on(self) -> None:
        fake = _fake_pyautogui()
        with patch.dict("sys.modules", {"pyautogui": fake}):
            assert _get_pyautogui() is fake
        assert fake.FAILSAFE is True

    async def test_a_tripped_failsafe_answers_an_error_the_model_reads(self) -> None:
        fake = _fake_pyautogui(press_error=_FailSafeError("mouse in corner"))
        with patch.dict("sys.modules", {"pyautogui": fake}):
            result = await ScreenInputTools().handler("press_key", {"key": "enter"})
        error = json.loads(result)["error"]
        assert "screen corner" in error
        assert "nothing was typed or clicked" in error

    async def test_other_errors_still_raise(self) -> None:
        fake = _fake_pyautogui(press_error=RuntimeError("display gone"))
        with (
            patch.dict("sys.modules", {"pyautogui": fake}),
            pytest.raises(RuntimeError, match="display gone"),
        ):
            await ScreenInputTools().handler("press_key", {"key": "enter"})

    async def test_an_untripped_call_runs(self) -> None:
        fake = _fake_pyautogui()
        with patch.dict("sys.modules", {"pyautogui": fake}):
            result = await ScreenInputTools().handler("press_key", {"key": "enter"})
        assert result == "Pressed key: enter"
        fake.press.assert_called_once_with("enter")
