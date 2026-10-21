"""Regression tests for local realtime example audio defaults."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from examples.shared.audio import build_denoiser, build_turn_detector, build_vad

from roomkit.local_models import find_local_model


def _downloaded(monkeypatch: pytest.MonkeyPatch, root: Path, *names: str) -> Path:
    """Mark local models downloaded in *root* (the marker alone) and point to it."""
    monkeypatch.setenv("ROOMKIT_MODELS_DIR", str(root))
    for name in names:
        model = find_local_model(name)
        (root / name).mkdir(parents=True)
        digests = "\n".join(f.sha256 for f in model.files)
        (root / name / ".roomkit-complete").write_text(digests + "\n")
    return root


def test_openai_local_example_defaults_to_continuous_webrtc_ns() -> None:
    """Open-speaker OpenAI sessions must not silently fall back to no NS."""
    example = Path(__file__).parents[1] / "examples" / "realtime_voice_local_openai.py"
    tree = ast.parse(example.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_denoiser"
    ]

    assert len(calls) == 1
    defaults = [keyword.value for keyword in calls[0].keywords if keyword.arg == "default"]
    assert len(defaults) == 1
    assert isinstance(defaults[0], ast.Constant)
    assert defaults[0].value == "webrtc"


def test_openai_local_example_guards_initial_provider_vad_input() -> None:
    """Open-speaker playback gives AEC time to converge before server VAD."""
    example = Path(__file__).parents[1] / "examples" / "realtime_voice_local_openai.py"
    tree = ast.parse(example.read_text())

    env_gets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "BARGE_IN_GUARD_MS"
    ]
    guard_configs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InterruptionConfig"
    ]

    assert len(env_gets) == 1
    assert isinstance(env_gets[0].args[1], ast.Constant)
    assert env_gets[0].args[1].value == "2000"
    assert len(guard_configs) == 1
    assert any(keyword.arg == "allow_during_first_ms" for keyword in guard_configs[0].keywords)


def test_webrtc_denoiser_factory_constructs_before_logging_enabled(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("DENOISE", raising=False)
    denoiser = MagicMock()
    constructor = MagicMock(return_value=denoiser)

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.webrtc.WebRTCNoiseSuppressorProvider",
            constructor,
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(24000, default="webrtc")

    assert result is denoiser
    constructor.assert_called_once_with(sample_rate=24000)
    assert "Denoiser enabled (WebRTC NS)" in caplog.messages


def test_failed_denoiser_initialization_is_not_logged_as_enabled(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("DENOISE", "rnnoise")

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.rnnoise.RNNoiseDenoiserProvider",
            side_effect=ImportError("librnnoise missing"),
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(24000)

    assert result is None
    assert "Denoiser enabled (RNNoise)" not in caplog.messages
    assert "RNNoise not installed — denoiser disabled" in caplog.messages


def test_dpdfnet_mode_builds_a_dpdfnet_denoiser(monkeypatch, caplog, tmp_path) -> None:
    monkeypatch.setenv("DENOISE", "dpdfnet")
    monkeypatch.delenv("DENOISE_MODEL", raising=False)
    root = _downloaded(monkeypatch, tmp_path, "dpdfnet2")
    constructor = MagicMock()
    constructor.return_value.model_sample_rate = 16000

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.sherpa_onnx.SherpaOnnxDenoiserProvider",
            constructor,
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(16000)

    assert result is constructor.return_value
    config = constructor.call_args.args[0]
    assert config.model_type == "dpdfnet"
    assert config.model == str(root / "dpdfnet2" / "dpdfnet2.onnx")
    assert "Denoiser enabled (sherpa-onnx dpdfnet, model=dpdfnet2)" in caplog.messages


def test_dpdfnet_is_left_out_at_a_rate_its_model_does_not_run(
    monkeypatch, caplog, tmp_path
) -> None:
    """At 24 kHz a 16 kHz DPDFNet passed the audio through, logged as enabled."""
    monkeypatch.setenv("DENOISE", "dpdfnet")
    monkeypatch.delenv("DENOISE_MODEL", raising=False)
    _downloaded(monkeypatch, tmp_path, "dpdfnet2")
    constructor = MagicMock()
    constructor.return_value.model_sample_rate = 16000

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.sherpa_onnx.SherpaOnnxDenoiserProvider",
            constructor,
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(24000)

    assert result is None
    assert not any("Denoiser enabled" in m for m in caplog.messages)
    assert any("runs at 16000 Hz and this pipeline at 24000 Hz" in m for m in caplog.messages)


def test_dpdfnet_takes_the_48khz_model_on_a_48khz_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DENOISE", "dpdfnet")
    monkeypatch.delenv("DENOISE_MODEL", raising=False)
    root = _downloaded(monkeypatch, tmp_path, "dpdfnet2-48khz-hr")
    constructor = MagicMock()
    constructor.return_value.model_sample_rate = 48000

    with patch(
        "roomkit.voice.pipeline.denoiser.sherpa_onnx.SherpaOnnxDenoiserProvider", constructor
    ):
        result = build_denoiser(48000)

    assert result is constructor.return_value
    expected = root / "dpdfnet2-48khz-hr" / "dpdfnet2_48khz_hr.onnx"
    assert constructor.call_args.args[0].model == str(expected)


def test_a_denoiser_model_file_overrides_the_local_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DENOISE", "sherpa")
    monkeypatch.setenv("DENOISE_MODEL", "/models/my_gtcrn.onnx")
    monkeypatch.setenv("ROOMKIT_MODELS_DIR", str(tmp_path))  # empty: never consulted
    constructor = MagicMock()

    with patch(
        "roomkit.voice.pipeline.denoiser.sherpa_onnx.SherpaOnnxDenoiserProvider", constructor
    ):
        build_denoiser(16000)

    assert constructor.call_args.args[0].model == "/models/my_gtcrn.onnx"


def test_a_missing_local_model_exits_with_the_download_command(
    monkeypatch, caplog, tmp_path
) -> None:
    monkeypatch.setenv("DENOISE", "sherpa")
    monkeypatch.delenv("DENOISE_MODEL", raising=False)
    monkeypatch.setenv("ROOMKIT_MODELS_DIR", str(tmp_path))

    with pytest.raises(SystemExit) as excinfo, caplog.at_level(logging.ERROR):
        build_denoiser(16000)

    assert excinfo.value.code == 1
    assert "python -m roomkit.local_models download gtcrn-simple" in caplog.text


@pytest.mark.parametrize("mode", ["ten", "silero"])
def test_sherpa_vad_runs_its_local_model(monkeypatch, tmp_path, mode: str) -> None:
    """VAD=ten|silero once built a config with no model file at all."""
    monkeypatch.setenv("VAD", mode)
    monkeypatch.delenv("VAD_MODEL", raising=False)
    root = _downloaded(monkeypatch, tmp_path, f"{mode}-vad")
    constructor = MagicMock()

    with patch("roomkit.voice.pipeline.vad.sherpa_onnx.SherpaOnnxVADProvider", constructor):
        build_vad(16000)

    config = constructor.call_args.args[0]
    assert config.model_type == mode
    assert config.model.startswith(str(root / f"{mode}-vad"))
    assert config.sample_rate == 16000


def test_turn_detector_runs_the_local_smart_turn(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TURN_DETECTOR", "1")
    monkeypatch.setenv("TURN_THRESHOLD", "0.7")
    monkeypatch.delenv("TURN_MODEL", raising=False)
    root = _downloaded(monkeypatch, tmp_path, "smart-turn-v3.2-cpu")
    constructor = MagicMock()

    with patch("roomkit.voice.pipeline.turn.smart_turn.SmartTurnDetector", constructor):
        build_turn_detector()

    config = constructor.call_args.args[0]
    assert config.model_path == str(root / "smart-turn-v3.2-cpu" / "smart-turn-v3.2-cpu.onnx")
    assert config.threshold == 0.7
