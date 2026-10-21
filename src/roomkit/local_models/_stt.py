"""Speech-to-text models for SherpaOnnxSTTConfig."""

from __future__ import annotations

from roomkit.local_models._model import SHERPA_ONNX_STT, LocalModel, ModelFile

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_HF = "https://huggingface.co/csukuangfj"


def _hf_files(repo: str, revision: str, sha256: dict[str, str]) -> tuple[ModelFile, ...]:
    """Files of one Hugging Face repo at one commit, by name -> SHA-256."""
    base = f"{_HF}/{repo}/resolve/{revision}"
    return tuple(ModelFile(f"{base}/{name}", digest) for name, digest in sha256.items())


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


def _whisper(
    size: str,
    revision: str,
    sha256: tuple[str, str, str],
    *,
    description: str,
    size_mb: int,
    language: str,
) -> LocalModel:
    encoder, decoder, tokens = (
        f"{size}-encoder.int8.onnx",
        f"{size}-decoder.int8.onnx",
        f"{size}-tokens.txt",
    )
    return LocalModel(
        name=f"whisper-{size}-int8",
        kind=SHERPA_ONNX_STT,
        description=description,
        size_mb=size_mb,
        files=_hf_files(
            f"sherpa-onnx-whisper-{size}",
            revision,
            dict(zip((encoder, decoder, tokens), sha256, strict=True)),
        ),
        paths={"encoder": encoder, "decoder": decoder, "tokens": tokens},
        settings={"mode": "whisper", "language": language},
    )


def _moonshine(size: str, sha256: str, *, description: str, size_mb: int) -> LocalModel:
    directory = f"sherpa-onnx-moonshine-{size}-en-quantized-2026-02-27"
    return LocalModel(
        name=f"moonshine-{size}-en-2026-02-27",
        kind=SHERPA_ONNX_STT,
        description=description,
        size_mb=size_mb,
        files=(ModelFile(f"{_RELEASES}/asr-models/{directory}.tar.bz2", sha256),),
        paths={
            "encoder": f"{directory}/encoder_model.ort",
            "decoder": f"{directory}/decoder_model_merged.ort",
            "tokens": f"{directory}/tokens.txt",
        },
        settings={"mode": "moonshine", "language": "en"},
    )


_ZIPFORMER_EN_SUFFIX = "epoch-99-avg-1-chunk-16-left-128"

STT_MODELS = (
    LocalModel(
        name="zipformer-en-2023-06-26",
        kind=SHERPA_ONNX_STT,
        description="Streaming STT, EN, zipformer transducer (int8)",
        size_mb=73,
        files=_hf_files(
            "sherpa-onnx-streaming-zipformer-en-2023-06-26",
            "672fbf1b30579d6585301139bb363f42a0ad4a24",
            {
                f"encoder-{_ZIPFORMER_EN_SUFFIX}.int8.onnx": (
                    "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1"
                ),
                f"decoder-{_ZIPFORMER_EN_SUFFIX}.onnx": (
                    "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef"
                ),
                f"joiner-{_ZIPFORMER_EN_SUFFIX}.int8.onnx": (
                    "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297"
                ),
                "tokens.txt": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
            },
        ),
        paths={
            "encoder": f"encoder-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
            "decoder": f"decoder-{_ZIPFORMER_EN_SUFFIX}.onnx",
            "joiner": f"joiner-{_ZIPFORMER_EN_SUFFIX}.int8.onnx",
            "tokens": "tokens.txt",
        },
        settings={"mode": "transducer", "language": "en"},
    ),
    # Drops the first words of an utterance resampled from 48 kHz (LiveKit's
    # finding): small and fast, for a 16 kHz source.
    LocalModel(
        name="zipformer-en-20M-2023-02-17",
        kind=SHERPA_ONNX_STT,
        description="Streaming STT, EN, small zipformer transducer (int8, 20M params)",
        size_mb=44,
        files=_hf_files(
            "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
            "d42f2d9f7ca24806fb667456a18a9f1b60f70d16",
            {
                "encoder-epoch-99-avg-1.int8.onnx": (
                    "3810755ce7c3ab26b42a8bcf39d191308fa27fb0f53358823ba46141d03b7eb3"
                ),
                "decoder-epoch-99-avg-1.int8.onnx": (
                    "21e2a2acd961b3ac72f55be2f10f1a285e1b0b0ba010d7c0b6eab141411b163c"
                ),
                "joiner-epoch-99-avg-1.int8.onnx": (
                    "e085d73b593cf9b0707f370dbd656d58327d3fe36d80d849202ef81df02cb01e"
                ),
                "tokens.txt": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
            },
        ),
        paths={
            "encoder": "encoder-epoch-99-avg-1.int8.onnx",
            "decoder": "decoder-epoch-99-avg-1.int8.onnx",
            "joiner": "joiner-epoch-99-avg-1.int8.onnx",
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
        name="parakeet-tdt-0.6b-v3-int8",
        kind=SHERPA_ONNX_STT,
        description="Offline STT, 25 European languages (EN, FR, DE, ES, IT...), Parakeet TDT v3",
        size_mb=487,
        files=(
            ModelFile(
                f"{_RELEASES}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
                "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
            ),
        ),
        paths={
            "encoder": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx",
            "decoder": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx",
            "joiner": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx",
            "tokens": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt",
        },
        settings={"mode": "transducer", "model_type": "nemo_transducer"},
    ),
    _whisper(
        "tiny.en",
        "d026532c022fa99fd789d6b32446a1df7b6bfc43",
        (
            "0ce578b827c94a961aacb8fa14b02f096504b337e5c94be37c36238cbe3e8bc6",
            "06c0e6ff6348d427e51839219d1c886c18cfdf411e629e33f5e1679bff9c1527",
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        ),
        description="Offline STT, EN, Whisper tiny.en (int8)",
        size_mb=104,
        language="en",
    ),
    # Multilingual: an empty language lets Whisper detect it; language="fr"
    # (to from_local_model()) fixes it.
    _whisper(
        "base",
        "bb53ee204431c90d314c1cc08d28d23e5b7927cc",
        (
            "0b8fb1304b6109976038efff5ace81720e00386f3ff6b54ee8c75291ca0a1e11",
            "9759d217388a01b3a4c7c15533201067b48ae819c4daafc8624e64b9409dc02d",
            "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
        ),
        description="Offline STT, multilingual, Whisper base (int8), language detected",
        size_mb=161,
        language="",
    ),
    _moonshine(
        "tiny",
        "9ec31b342d8fa3240c3b81b8f82e1cf7e3ac467c93ca5a999b741d5887164f8d",
        description="Offline STT, EN, Moonshine v2 tiny (quantized)",
        size_mb=30,
    ),
    _moonshine(
        "base",
        "43232c1d13013d37317163baec3135bd771a186a4356f28c889bab453bb0e891",
        description="Offline STT, EN, Moonshine v2 base (quantized)",
        size_mb=111,
    ),
)
