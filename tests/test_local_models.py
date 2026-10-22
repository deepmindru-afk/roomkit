"""Local model files: pinned downloads, offline resolution, configs built by name."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest

from roomkit.local_models import (
    KOKORO_V1_VOICES,
    LOCAL_MODELS,
    SHERPA_ONNX_DENOISER,
    SHERPA_ONNX_STT,
    SHERPA_ONNX_TTS,
    SHERPA_ONNX_VAD,
    SMART_TURN,
    LocalModel,
    ModelDownloadError,
    ModelFile,
    ModelNotDownloadedError,
    _install,
    default_models_dir,
    download_local_model,
    find_local_model,
    is_downloaded,
    list_local_models,
    local_model_config,
    local_model_dir,
)
from roomkit.local_models.__main__ import main
from roomkit.voice.pipeline.denoiser.sherpa_onnx import SherpaOnnxDenoiserConfig
from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig

CONFIG_OF_KIND = {
    SHERPA_ONNX_STT: SherpaOnnxSTTConfig,
    SHERPA_ONNX_TTS: SherpaOnnxTTSConfig,
    SHERPA_ONNX_VAD: SherpaOnnxVADConfig,
    SHERPA_ONNX_DENOISER: SherpaOnnxDenoiserConfig,
    SMART_TURN: SmartTurnConfig,
}

BASE = "https://models.example"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_bz2(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeServer:
    """Stands in for urllib's urlopen: serves bytes by URL and counts fetches."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.files: dict[str, bytes] = {}
        self.fetched: list[str] = []
        monkeypatch.setattr(_install.urllib.request, "urlopen", self._urlopen)

    def serve(self, name: str, data: bytes) -> ModelFile:
        url = f"{BASE}/{name}"
        self.files[url] = data
        return ModelFile(url, _sha(data))

    def _urlopen(self, request: object, timeout: float) -> io.BytesIO:
        url = request.full_url  # type: ignore[attr-defined]
        self.fetched.append(url)
        if url not in self.files:
            raise urllib.error.URLError("404")
        return io.BytesIO(self.files[url])


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    return FakeServer(monkeypatch)


def _demo_model(server: FakeServer, *, weights: bytes = b"weights") -> LocalModel:
    return LocalModel(
        name="demo",
        kind=SHERPA_ONNX_VAD,
        description="demo",
        size_mb=1,
        files=(
            server.serve("weights.onnx", weights),
            server.serve("bundle.tar.bz2", _tar_bz2({"bundle/tokens.txt": b"a b c"})),
        ),
        paths={"model": "weights.onnx"},
    )


def _mark_downloaded(model: LocalModel, root: Path) -> Path:
    """What a completed download leaves, without its files: the marker alone."""
    target = root / model.name
    target.mkdir(parents=True)
    (target / ".roomkit-complete").write_text("\n".join(f.sha256 for f in model.files) + "\n")
    return target


class TestKnownModels:
    def test_names_are_unique(self) -> None:
        names = [model.name for model in LOCAL_MODELS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("model", LOCAL_MODELS, ids=lambda m: m.name)
    def test_every_value_is_a_field_of_its_config(self, model: LocalModel) -> None:
        fields = {f.name for f in dataclasses.fields(CONFIG_OF_KIND[model.kind])}
        keys = set(model.paths) | set(model.settings)
        if model.voices:
            keys.add("voice_names")
        assert keys <= fields

    @pytest.mark.parametrize("model", LOCAL_MODELS, ids=lambda m: m.name)
    def test_every_path_comes_from_a_pinned_file(self, model: LocalModel) -> None:
        """A path names a downloaded file, or lies inside an extracted archive."""
        plain = {file.name for file in model.files if not file.is_archive}
        for value in model.paths.values():
            for path in value.split(","):
                assert path in plain or any(file.is_archive for file in model.files)

    def test_kokoro_voices_follow_the_models_speaker_table(self) -> None:
        assert len(KOKORO_V1_VOICES) == 54
        assert KOKORO_V1_VOICES.index("af_heart") == 3
        assert KOKORO_V1_VOICES.index("ff_siwis") == 30

    def test_list_holds_every_model_or_those_of_one_config(self) -> None:
        assert list_local_models() == list(LOCAL_MODELS)
        tts = list_local_models(SHERPA_ONNX_TTS)
        assert tts and all(model.kind == SHERPA_ONNX_TTS for model in tts)
        assert {m.name for m in tts} == {m.name for m in LOCAL_MODELS if m.kind == SHERPA_ONNX_TTS}

    def test_list_refuses_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="sherpa-onnx-stt"):
            list_local_models("sherpa-onnx")

    def test_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="zipformer-en-2023-06-26"):
            find_local_model("nope")

    def test_a_model_of_another_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sherpa-onnx-vad model, not a sherpa-onnx-stt"):
            find_local_model("ten-vad", kind=SHERPA_ONNX_STT)


class TestModelFile:
    def test_refuses_a_url_that_is_not_https(self) -> None:
        with pytest.raises(ValueError, match="https"):
            ModelFile("http://models.example/a.onnx", "0" * 64)

    def test_refuses_a_malformed_sha256(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            ModelFile(f"{BASE}/a.onnx", "ABC")

    def test_an_archive_is_told_by_its_name(self) -> None:
        assert ModelFile(f"{BASE}/a.tar.bz2", "0" * 64).is_archive
        assert not ModelFile(f"{BASE}/a.onnx", "0" * 64).is_archive


class TestModelsDir:
    def test_the_env_var_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ROOMKIT_MODELS_DIR", str(tmp_path))
        assert default_models_dir() == tmp_path

    def test_follows_the_xdg_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("ROOMKIT_MODELS_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_models_dir() == tmp_path / "roomkit" / "models"


class TestDownload:
    def test_installs_files_and_extracts_archives(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        model = _demo_model(server)
        target = download_local_model(model, tmp_path)

        assert target == tmp_path / "demo"
        assert (target / "weights.onnx").read_bytes() == b"weights"
        assert (target / "bundle" / "tokens.txt").read_bytes() == b"a b c"
        assert not (target / "bundle.tar.bz2").exists()
        assert is_downloaded(model, tmp_path)

    def test_a_downloaded_model_is_not_fetched_again(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        model = _demo_model(server)
        download_local_model(model, tmp_path)
        server.fetched.clear()

        download_local_model(model, tmp_path)

        assert server.fetched == []

    def test_a_file_that_does_not_match_is_refused_and_nothing_is_left(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        model = _demo_model(server)
        server.files[model.files[0].url] = b"tampered"

        with pytest.raises(ModelDownloadError, match="does not match the pinned"):
            download_local_model(model, tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_a_network_failure_is_a_download_error(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        model = _demo_model(server)
        del server.files[model.files[1].url]

        with pytest.raises(ModelDownloadError, match="cannot download"):
            download_local_model(model, tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_a_zip_member_escaping_the_directory_is_refused(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("../escaped.txt", "x")
        model = LocalModel(
            name="evil",
            kind=SHERPA_ONNX_VAD,
            description="evil",
            size_mb=1,
            files=(server.serve("evil.zip", buffer.getvalue()),),
            paths={"model": "escaped.txt"},
        )

        with pytest.raises(ModelDownloadError, match="escapes"):
            download_local_model(model, tmp_path / "models")

        assert not (tmp_path / "escaped.txt").exists()

    def test_new_pinned_files_replace_an_older_copy(
        self, server: FakeServer, tmp_path: Path
    ) -> None:
        download_local_model(_demo_model(server, weights=b"v1"), tmp_path)
        newer = _demo_model(server, weights=b"v2")

        assert not is_downloaded(newer, tmp_path)
        download_local_model(newer, tmp_path)

        assert (tmp_path / "demo" / "weights.onnx").read_bytes() == b"v2"
        assert [p.name for p in tmp_path.iterdir()] == ["demo"]


class TestOfflineResolution:
    def test_a_missing_model_names_the_command_that_downloads_it(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotDownloadedError) as excinfo:
            local_model_dir("ten-vad", tmp_path)

        message = str(excinfo.value)
        assert "python -m roomkit.local_models download" in message
        assert f"--models-dir {tmp_path}" in message
        assert message.endswith("ten-vad")

    def test_never_touches_the_network(self, server: FakeServer, tmp_path: Path) -> None:
        with pytest.raises(ModelNotDownloadedError):
            local_model_dir("ten-vad", tmp_path)
        assert server.fetched == []

    def test_config_values_point_into_the_models_directory(self, tmp_path: Path) -> None:
        kokoro = find_local_model("kokoro-int8-multi-lang-v1_0")
        target = _mark_downloaded(kokoro, tmp_path)

        values = local_model_config(kokoro.name, kind=SHERPA_ONNX_TTS, models_dir=tmp_path)

        assert values["model"] == str(target / "kokoro-int8-multi-lang-v1_0/model.int8.onnx")
        # Several lexicons stay comma-joined, each resolved.
        assert values["lexicon"].split(",") == [
            str(target / "kokoro-int8-multi-lang-v1_0/lexicon-us-en.txt"),
            str(target / "kokoro-int8-multi-lang-v1_0/lexicon-zh.txt"),
        ]
        assert values["voice_names"] == KOKORO_V1_VOICES
        assert values["sample_rate"] == 24000


class TestConfigsFromLocalModel:
    def test_stt(self, tmp_path: Path) -> None:
        target = _mark_downloaded(find_local_model("zipformer-fr-kroko-2025-08-06"), tmp_path)

        config = SherpaOnnxSTTConfig.from_local_model(
            "zipformer-fr-kroko-2025-08-06", models_dir=tmp_path, provider="cuda"
        )

        directory = target / "sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06"
        assert config.encoder == str(directory / "encoder.onnx")
        assert config.tokens == str(directory / "tokens.txt")
        assert config.mode == "transducer"
        assert config.tail_padding_s == 1.5
        assert config.provider == "cuda"  # an override wins

    def test_tts_by_voice_name(self, tmp_path: Path) -> None:
        _mark_downloaded(find_local_model("kokoro-int8-multi-lang-v1_0"), tmp_path)

        config = SherpaOnnxTTSConfig.from_local_model(
            "kokoro-int8-multi-lang-v1_0", voice="ff_siwis", models_dir=tmp_path, lang="fr"
        )

        assert config.model_type == "kokoro"
        assert config.speaker_id == 30
        assert config.lang == "fr"
        assert config.speaker_id_for("af_heart") == 3

    def test_tts_default_voice_is_the_models(self, tmp_path: Path) -> None:
        _mark_downloaded(find_local_model("kokoro-int8-multi-lang-v1_0"), tmp_path)
        config = SherpaOnnxTTSConfig.from_local_model(
            "kokoro-int8-multi-lang-v1_0", models_dir=tmp_path
        )
        assert config.speaker_id == 3  # af_heart

    def test_tts_unknown_voice_lists_the_names(self, tmp_path: Path) -> None:
        _mark_downloaded(find_local_model("kokoro-int8-multi-lang-v1_0"), tmp_path)
        with pytest.raises(ValueError, match="af_heart"):
            SherpaOnnxTTSConfig.from_local_model(
                "kokoro-int8-multi-lang-v1_0", voice="nobody", models_dir=tmp_path
            )

    def test_vad_denoiser_and_turn(self, tmp_path: Path) -> None:
        for name in ("silero-vad", "gtcrn-simple", "smart-turn-v3.2-cpu"):
            _mark_downloaded(find_local_model(name), tmp_path)

        vad = SherpaOnnxVADConfig.from_local_model("silero-vad", models_dir=tmp_path)
        denoiser = SherpaOnnxDenoiserConfig.from_local_model("gtcrn-simple", models_dir=tmp_path)
        turn = SmartTurnConfig.from_local_model("smart-turn-v3.2-cpu", models_dir=tmp_path)

        assert vad.model == str(tmp_path / "silero-vad" / "silero_vad.onnx")
        assert vad.model_type == "silero"
        assert denoiser.model == str(tmp_path / "gtcrn-simple" / "gtcrn_simple.onnx")
        assert turn.model_path == str(
            tmp_path / "smart-turn-v3.2-cpu" / "smart-turn-v3.2-cpu.onnx"
        )

    def test_a_model_for_another_config_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a sherpa-onnx-stt"):
            SherpaOnnxSTTConfig.from_local_model("ten-vad", models_dir=tmp_path)

    def test_a_missing_model_raises_before_any_config(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotDownloadedError):
            SherpaOnnxSTTConfig.from_local_model("zipformer-en-2023-06-26", models_dir=tmp_path)


class TestSpeakerIds:
    def test_a_number_is_still_a_speaker_id(self) -> None:
        config = SherpaOnnxTTSConfig(model="m.onnx", tokens="t.txt", voice_names=("a", "b"))
        assert config.speaker_id_for("1") == 1
        assert config.speaker_id_for("b") == 1
        assert config.speaker_id_for(None) == 0

    def test_a_config_without_names_says_so(self) -> None:
        config = SherpaOnnxTTSConfig(model="m.onnx", tokens="t.txt")
        with pytest.raises(ValueError, match="names no voices"):
            config.speaker_id_for("af_heart")


class TestCommandLine:
    def test_list_marks_the_downloaded_models(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _mark_downloaded(find_local_model("ten-vad"), tmp_path)
        caplog.set_level("INFO", logger="roomkit.local_models")

        assert main(["list", "--models-dir", str(tmp_path), "--kind", SHERPA_ONNX_VAD]) == 0

        lines = [record.getMessage() for record in caplog.records]
        assert any(line.startswith("* ten-vad") for line in lines)
        assert any(line.startswith("  silero-vad") for line in lines)
        assert not any("kokoro" in line for line in lines)

    def test_download_then_path(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _demo_model(server)
        monkeypatch.setattr("roomkit.local_models.__main__.find_local_model", lambda _n: model)

        assert main(["path", "--models-dir", str(tmp_path), "demo"]) == 1
        assert main(["download", "--models-dir", str(tmp_path), "demo"]) == 0
        assert main(["path", "--models-dir", str(tmp_path), "demo"]) == 0

    def test_a_failed_download_exits_1(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _demo_model(server)
        server.files.clear()
        monkeypatch.setattr("roomkit.local_models.__main__.find_local_model", lambda _n: model)

        assert main(["download", "--models-dir", str(tmp_path), "demo"]) == 1

    def test_the_suggested_command_runs_as_printed(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command a ModelNotDownloadedError names is the one that downloads it."""
        model = _demo_model(server)
        monkeypatch.setattr("roomkit.local_models.__main__.find_local_model", lambda _n: model)
        with pytest.raises(ModelNotDownloadedError) as excinfo:
            local_model_dir(model, tmp_path)
        command = str(excinfo.value).split("python -m roomkit.local_models ", 1)[1]

        assert main(command.split()) == 0
        assert is_downloaded(model, tmp_path)

    def test_an_unknown_name_exits_2(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert main(["download", "--models-dir", str(tmp_path), "nope"]) == 2
        assert "Unknown local model 'nope'" in caplog.text
