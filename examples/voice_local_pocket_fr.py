"""RoomKit -- Local French voice assistant with Pocket TTS, on CPU or GPU.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and French speech-to-text (CPU)
  - a local LLM served by Ollama
  - Kyutai's Pocket TTS, French model, on the CPU (default) or a CUDA GPU

    Mic → [AEC] → VAD → sherpa-onnx STT (fr) → local LLM → [StripEmoji] → Pocket TTS (fr) → Speaker

Pocket TTS is a 100M-parameter model that streams faster than real time on
two CPU cores: no GPU is needed for the voice. Measured on a desktop CPU, the
``french`` model starts speaking ~80 ms after it gets a sentence.

Requirements:
    Ollama running locally, with a model that answers without reasoning first:
        ollama pull qwen3:4b-instruct
    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

Models (download once, into examples/models/):
    mkdir -p examples/models && cd examples/models
    # VAD: TEN-VAD
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
    # STT: Kroko, a Zipformer transducer, French, streaming
    # (model license: huggingface.co/Banafo/Kroko-ASR)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06.tar.bz2
    cd ../..

    Pocket TTS weights and its pre-made voices download from Hugging Face on
    first run (licences per voice: huggingface.co/kyutai/tts-voices).

Run (from the repository root):
    uv run --extra local-audio --extra webrtc-aec --extra ollama \\
        --extra sherpa-onnx --extra pocket-tts \\
        python examples/voice_local_pocket_fr.py

    On Linux this installs the CUDA build of PyTorch (~3 GB). For CPU only,
    install torch from https://download.pytorch.org/whl/cpu first (see the
    Pocket TTS guide).

Environment variables:
    --- LLM (Ollama) ---
    LLM_MODEL           Ollama model (default: qwen3:4b-instruct)
    OLLAMA_HOST         Ollama server (default: http://localhost:11434)
    LLM_MAX_TOKENS      Max response tokens (default: 200)
    SYSTEM_PROMPT       Custom system prompt

    --- STT and VAD (sherpa-onnx, CPU) ---
    MODELS_DIR          Where the models were downloaded (default: examples/models)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)
    VAD_MODEL, STT_ENCODER, STT_DECODER, STT_JOINER, STT_TOKENS
                        Override one model file (default: found in MODELS_DIR)

    --- Pocket TTS ---
    POCKET_DEVICE       cpu | cuda (default: cpu)
    POCKET_LANGUAGE     french | french_24l (default: french). french_24l
                        sounds better but is ~3x slower, and often stops a
                        long sentence at its first comma.
    POCKET_VOICE        A pre-made voice (default: estelle, the French one) or
                        the path of a clean speech clip to clone
    POCKET_QUANTIZE     1 for int8 quantization, CPU only (default: 0)
    POCKET_DISABLE_CUDNN
                        1 if the GPU run fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                        (a system cuDNN shadowing PyTorch's)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    MUTE_MIC            Mute the mic while the assistant speaks: 1 | 0
                        (default: 0 with AEC). Muting disables barge-in.

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_logging

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.filters import StripEmoji
from roomkit.voice.tts.pocket import SAMPLE_RATE, PocketTTSConfig, PocketTTSProvider

logger = setup_logging("voice_local_pocket_fr")

MIC_RATE = 16000
BLOCK_MS = 20

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
STT_DIR = "sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06"
MODEL_FILES = {
    "VAD_MODEL": "ten-vad.onnx",
    "STT_ENCODER": f"{STT_DIR}/encoder.onnx",
    "STT_DECODER": f"{STT_DIR}/decoder.onnx",
    "STT_JOINER": f"{STT_DIR}/joiner.onnx",
    "STT_TOKENS": f"{STT_DIR}/tokens.txt",
}

SYSTEM_PROMPT = (
    "You are a warm voice assistant having a spoken conversation. "
    "Always answer in French, in one or two short, natural sentences, the way people talk. "
    "Never use lists, markdown or emojis."
)


def model_paths() -> dict[str, str]:
    """Each model file: its env var when set, else its place in MODELS_DIR."""
    models_dir = Path(os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR))
    paths = {
        var: os.environ.get(var) or str(models_dir / name) for var, name in MODEL_FILES.items()
    }
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        logger.error("Model files not found, see 'Models' at the top of this example:")
        for path in missing:
            logger.error("  %s", path)
        sys.exit(1)
    return paths


def build_aec() -> object | None:
    mode = os.environ.get("AEC", "webrtc").lower()
    if mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        return WebRTCAECProvider(sample_rate=MIC_RATE)
    if mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        frame = MIC_RATE * BLOCK_MS // 1000
        return SpeexAECProvider(frame_size=frame, filter_length=frame * 10, sample_rate=MIC_RATE)
    return None


async def main() -> None:
    env = model_paths()
    if os.environ.get("POCKET_DISABLE_CUDNN") == "1":
        import torch

        torch.backends.cudnn.enabled = False

    kit = RoomKit()

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while the assistant speaks, so
    # you can interrupt it; without, it is muted during playback.
    aec = build_aec()
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=SAMPLE_RATE,  # Pocket TTS speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )
    logger.info("Audio: AEC=%s, mic muted during playback=%s", type(aec).__name__, mute_mic)

    # --- VAD and STT (sherpa-onnx, CPU) ----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=env["VAD_MODEL"],
            model_type="ten",
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=MIC_RATE,
            provider="cpu",
        )
    )
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode="transducer",
            encoder=env["STT_ENCODER"],
            decoder=env["STT_DECODER"],
            joiner=env["STT_JOINER"],
            tokens=env["STT_TOKENS"],
            sample_rate=MIC_RATE,
            provider="cpu",
            # Kroko French drops the last word ("midi" -> "mid") below ~1.5 s
            # of tail silence; the padding costs compute, not waiting time.
            tail_padding_s=1.5,
        )
    )

    # --- Pocket TTS: French, on the CPU or the GPU -----------------------------
    voice_name = os.environ.get("POCKET_VOICE", "estelle")
    tts = PocketTTSProvider(
        PocketTTSConfig(
            language=os.environ.get("POCKET_LANGUAGE", "french"),
            voices={"assistant": voice_name},
            device=os.environ.get("POCKET_DEVICE", "cpu"),
            quantize=os.environ.get("POCKET_QUANTIZE") == "1",
        )
    )

    # --- LLM (Ollama's native API: thinking off, a spoken reply cannot wait) ------
    llm_model = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ai_provider = OllamaAIProvider(
        OllamaConfig(
            host=ollama_host,
            model=llm_model,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "200")),
            think=False,
        )
    )
    logger.info("LLM: %s (%s), Pocket TTS voice: %s", llm_model, ollama_host, voice_name)

    # --- Channels and room -------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad, aec=aec),
        # The LLM adds emoji despite the prompt; spoken, they sound wrong.
        tts_filter=StripEmoji(),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT),
        )
    )
    await kit.create_room(room_id="local-pocket-fr")
    await kit.attach_channel("local-pocket-fr", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(event, ctx):
        logger.info("Barge-in: the assistant stops speaking")

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading Pocket TTS, STT and VAD models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    await kit.attach_channel("local-pocket-fr", "voice")  # opens the mic
    logger.info("Ready: speak French into the microphone. Ctrl+C to stop.")

    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
