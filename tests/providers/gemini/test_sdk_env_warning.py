"""The SDK's env-key warning is silenced only where it would be wrong.

google-genai calls ``get_env_api_key()`` at every client construction, before
it looks at the key it was handed, so with both variables set it announces
"Using GOOGLE_API_KEY" even when the caller passed its own. For RoomKit that
statement is false, not merely noisy.
"""

from __future__ import annotations

import logging

import pytest

from roomkit.providers.gemini.sdk import _DropEnvKeyWarning, _without_env_key_warning

SDK_LOGGER = "google_genai._api_client"
MESSAGE = "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."


def _emit() -> None:
    logging.getLogger(SDK_LOGGER).warning(MESSAGE)


def test_the_warning_is_dropped_while_we_supply_the_key(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.WARNING, logger=SDK_LOGGER),
        _without_env_key_warning(explicit_key=True),
    ):
        _emit()

    assert MESSAGE not in caplog.text


def test_the_warning_survives_when_we_supply_no_key(caplog: pytest.LogCaptureFixture) -> None:
    """Env resolution is then the real behaviour, and worth hearing about."""
    with (
        caplog.at_level(logging.WARNING, logger=SDK_LOGGER),
        _without_env_key_warning(explicit_key=False),
    ):
        _emit()

    assert MESSAGE in caplog.text


def test_the_filter_is_removed_afterwards(caplog: pytest.LogCaptureFixture) -> None:
    """An application that builds its own client still hears the SDK."""
    with _without_env_key_warning(explicit_key=True):
        pass

    with caplog.at_level(logging.WARNING, logger=SDK_LOGGER):
        _emit()

    assert MESSAGE in caplog.text
    assert not logging.getLogger(SDK_LOGGER).filters


def test_the_filter_is_removed_even_when_the_build_raises() -> None:
    with pytest.raises(RuntimeError), _without_env_key_warning(explicit_key=True):
        raise RuntimeError("client construction failed")

    assert not logging.getLogger(SDK_LOGGER).filters


def test_other_records_from_the_same_logger_pass(caplog: pytest.LogCaptureFixture) -> None:
    """Matched on its text, so nothing else the SDK says is lost."""
    with (
        caplog.at_level(logging.WARNING, logger=SDK_LOGGER),
        _without_env_key_warning(explicit_key=True),
    ):
        logging.getLogger(SDK_LOGGER).warning("quota exceeded, retrying")

    assert "quota exceeded" in caplog.text


def test_the_filter_matches_the_message_it_targets() -> None:
    record = logging.LogRecord(SDK_LOGGER, logging.WARNING, __file__, 1, MESSAGE, None, None)
    assert _DropEnvKeyWarning().filter(record) is False
