"""sherpa-onnx models: speech-to-text, text-to-speech, VAD and denoiser."""

from __future__ import annotations

from roomkit.local_models._model import (
    SHERPA_ONNX_DENOISER,
    SHERPA_ONNX_STT,
    SHERPA_ONNX_TTS,
    SHERPA_ONNX_VAD,
    LocalModel,
    ModelFile,
)

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_ZIPFORMER_EN = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    "/resolve/672fbf1b30579d6585301139bb363f42a0ad4a24"
)
_ZIPFORMER_EN_SUFFIX = "epoch-99-avg-1-chunk-16-left-128"

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


def _kroko(language: str, sha256: str, **settings: object) -> LocalModel:
    directory = f"sherpa-onnx-streaming-zipformer-{language}-kroko-2025-08-06"
    return LocalModel(
        name=f"zipformer-{language}-kroko-2025-08-06",
        kind=SHERPA_ONNX_STT,
        description=f"Streaming STT, {language.upper()}, Kroko zipformer transducer",
        size_mb=57,
        files=(ModelFile(f"{_RELEASES}/asr-models/{directory}.tar.bz2", sha256),),
        paths={
            "encoder": f"{directory}/encoder.onnx",
            "decoder": f"{directory}/decoder.onnx",
            "joiner": f"{directory}/joiner.onnx",
            "tokens": f"{directory}/tokens.txt",
        },
        settings={"mode": "transducer", "language": language, **settings},
    )


SHERPA_ONNX_MODELS = (
    LocalModel(
        name="zipformer-en-2023-06-26",
        kind=SHERPA_ONNX_STT,
        description="Streaming STT, EN, zipformer transducer (int8)",
        size_mb=73,
        files=(
            ModelFile(
                f"{_ZIPFORMER_EN}/encoder-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
                "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
            ),
            ModelFile(
                f"{_ZIPFORMER_EN}/decoder-{_ZIPFORMER_EN_SUFFIX}.onnx",
                "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
            ),
            ModelFile(
                f"{_ZIPFORMER_EN}/joiner-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
                "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297",
            ),
            ModelFile(
                f"{_ZIPFORMER_EN}/tokens.txt",
                "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
            ),
        ),
        paths={
            "encoder": f"encoder-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
            "decoder": f"decoder-{_ZIPFORMER_EN_SUFFIX}.onnx",
            "joiner": f"joiner-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
            "tokens": "tokens.txt",
        },
        settings={"mode": "transducer", "language": "en"},
    ),
    _kroko("en", "c8676e5ff9ac2a85296e53ee0fd4d5fb1db6770e7a7647166eeafe349ade6834"),
    # Kroko French drops the last word ("midi" -> "mid") below ~1.5 s of tail
    # silence; the padding costs compute, not waiting time.
    _kroko(
        "fr",
        "e6ffd3dc43725cd6c8137b05c739f15607d0df946b9b90eb141e10059efca024",
        tail_padding_s=1.5,
    ),
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
    LocalModel(
        name="vits-piper-en_US-amy-low",
        kind=SHERPA_ONNX_TTS,
        description="TTS, EN (US), Piper 'amy' voice, 16 kHz",
        size_mb=67,
        files=(
            ModelFile(
                f"{_RELEASES}/tts-models/vits-piper-en_US-amy-low.tar.bz2",
                "c70f5284a09a7fd4ed203b39b2ff51cac1432b422b852eb647b481dade3cf639",
            ),
        ),
        paths={
            "model": "vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            "tokens": "vits-piper-en_US-amy-low/tokens.txt",
            "data_dir": "vits-piper-en_US-amy-low/espeak-ng-data",
        },
        settings={"model_type": "vits", "sample_rate": 16000},
        voices=("amy",),
    ),
    LocalModel(
        name="ten-vad",
        kind=SHERPA_ONNX_VAD,
        description="Voice activity detection, TEN-VAD",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/asr-models/ten-vad.onnx",
                "718cb7eef47e3cf5ddbe7e967a7503f46b8b469c0706872f494dfa921b486206",
            ),
        ),
        paths={"model": "ten-vad.onnx"},
        settings={"model_type": "ten"},
    ),
    LocalModel(
        name="silero-vad",
        kind=SHERPA_ONNX_VAD,
        description="Voice activity detection, Silero VAD",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/asr-models/silero_vad.onnx",
                "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
            ),
        ),
        paths={"model": "silero_vad.onnx"},
        settings={"model_type": "silero"},
    ),
    LocalModel(
        name="gtcrn-simple",
        kind=SHERPA_ONNX_DENOISER,
        description="Speech enhancement (denoiser), GTCRN",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/speech-enhancement-models/gtcrn_simple.onnx",
                "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
            ),
        ),
        paths={"model": "gtcrn_simple.onnx"},
        settings={"model_type": "gtcrn"},
    ),
)
