"""Text-to-speech models for SherpaOnnxTTSConfig."""

from __future__ import annotations

from roomkit.local_models._model import SHERPA_ONNX_TTS, LocalModel, ModelFile

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# The speaker table of Kokoro v1.0, read from the model's own `speaker_names`
# metadata: the position of a name is its speaker id.
KOKORO_V1_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora",
    "em_alex", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex",
    "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "em_santa",
)  # fmt: skip


def _piper(
    voice: str, sha256: str, *, description: str, size_mb: int, sample_rate: int
) -> LocalModel:
    directory = f"vits-piper-{voice}"
    speaker = voice.split("-")[1]
    return LocalModel(
        name=directory,
        kind=SHERPA_ONNX_TTS,
        description=description,
        size_mb=size_mb,
        files=(ModelFile(f"{_RELEASES}/tts-models/{directory}.tar.bz2", sha256),),
        paths={
            "model": f"{directory}/{voice}.onnx",
            "tokens": f"{directory}/tokens.txt",
            "data_dir": f"{directory}/espeak-ng-data",
        },
        settings={"model_type": "vits", "sample_rate": sample_rate},
        voices=(speaker,),
    )


TTS_MODELS = (
    LocalModel(
        name="kokoro-int8-multi-lang-v1_0",
        kind=SHERPA_ONNX_TTS,
        description="TTS, Kokoro v1.0 (int8), 54 voices: EN, ES, FR, HI, IT, JA, PT, ZH",
        size_mb=132,
        files=(
            ModelFile(
                f"{_RELEASES}/tts-models/kokoro-int8-multi-lang-v1_0.tar.bz2",
                "4c3052abaa60943a341f193888cf6abd68787dae6ab8ae5c925a706caa247e4e",
            ),
        ),
        paths={
            "model": "kokoro-int8-multi-lang-v1_0/model.int8.onnx",
            "voices": "kokoro-int8-multi-lang-v1_0/voices.bin",
            "tokens": "kokoro-int8-multi-lang-v1_0/tokens.txt",
            "data_dir": "kokoro-int8-multi-lang-v1_0/espeak-ng-data",
            "lexicon": (
                "kokoro-int8-multi-lang-v1_0/lexicon-us-en.txt,"
                "kokoro-int8-multi-lang-v1_0/lexicon-zh.txt"
            ),
        },
        settings={"model_type": "kokoro", "sample_rate": 24000, "speaker_id": 3},  # af_heart
        voices=KOKORO_V1_VOICES,
    ),
    _piper(
        "en_US-amy-low",
        "c70f5284a09a7fd4ed203b39b2ff51cac1432b422b852eb647b481dade3cf639",
        description="TTS, EN (US), Piper 'amy' voice, 16 kHz",
        size_mb=67,
        sample_rate=16000,
    ),
    _piper(
        "fr_FR-siwis-medium",
        "375909aa30842b3a4efa10b1beb1d761af792960ae6873b4d53889f96c66195b",
        description="TTS, FR, Piper 'siwis' voice, 22 kHz",
        size_mb=67,
        sample_rate=22050,
    ),
)
