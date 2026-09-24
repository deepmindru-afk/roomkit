"""RoomKit -- Local voice assistant with Vui Nano: it hears the conversation.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and speech-to-text (CPU)
  - a local LLM served by Ollama
  - Vui Nano text-to-speech (CUDA GPU), which generates each reply inside the
    dialogue: the TTS conversation context (RFC §12.2.2) hands it every turn,
    the audio of what you said included, and a barge-in cuts its reply back to
    what you actually heard.

    Mic → [AEC] → VAD → sherpa-onnx STT → local LLM → Vui → Speaker

Vui speaks English only: the STT model, the LLM prompt and the voice are
English here.

Requirements:
    Python 3.12 (a vui-tts requirement), a CUDA GPU (Vui takes ~3.5 GB of VRAM)
    Ollama running locally, with a model that answers without reasoning first:
        ollama pull qwen3:4b-instruct   # ~3 GB of VRAM, fits beside Vui on 12 GB
    (Ollama's qwen3:4b is the Thinking-2507 build: it reasons whatever `think`
    says, and Vui would read its reasoning aloud.)

    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

Models (download once, into examples/models/):
    mkdir -p examples/models && cd examples/models
    # VAD: TEN-VAD
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
    # STT: Zipformer transducer, English, streaming
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    cd ../..

    Vui's weights and voice presets download from Hugging Face on first run.

Run (from the repository root; .venv-vui keeps the Python 3.12 environment
apart from the project's .venv):
    UV_PROJECT_ENVIRONMENT=.venv-vui uv run --python 3.12 \\
        --extra local-audio --extra webrtc-aec --extra ollama \\
        --extra sherpa-onnx --extra vui \\
        python examples/voice_local_vui.py

Environment variables:
    --- LLM (Ollama) ---
    LLM_MODEL           Ollama model (default: qwen3:4b-instruct; thinking is
                        turned off for models that can switch it)
    OLLAMA_HOST         Ollama server (default: http://localhost:11434)
    LLM_MAX_TOKENS      Max response tokens (default: 200)
    SYSTEM_PROMPT       Custom system prompt

    --- STT and VAD (sherpa-onnx, CPU) ---
    MODELS_DIR          Where the models were downloaded (default: examples/models)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)
    VAD_MODEL, STT_ENCODER, STT_DECODER, STT_JOINER, STT_TOKENS
                        Override one model file (default: found in MODELS_DIR)

    --- Vui ---
    VUI_VOICE           Preset voice: maeve | abraham | rhian | harry (default: maeve)
    VUI_INCLUDE_AUDIO   Let Vui hear your voice, not only your words: 1 | 0 (default: 1)
    VUI_DISABLE_CUDNN   1 if the codec fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                        (a system cuDNN shadowing PyTorch's)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    MUTE_MIC            Mute the mic while Vui speaks: 1 | 0 (default: 0 with AEC).
                        Muting disables barge-in.

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
    TTSContextConfig,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.vui import SAMPLE_RATE, VuiTTSConfig, VuiTTSProvider, VuiVoice

logger = setup_logging("voice_local_vui")

MIC_RATE = 16000
BLOCK_MS = 20

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
STT_DIR = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
MODEL_FILES = {
    "VAD_MODEL": "ten-vad.onnx",
    "STT_ENCODER": f"{STT_DIR}/encoder-epoch-99-avg-1.onnx",
    "STT_DECODER": f"{STT_DIR}/decoder-epoch-99-avg-1.onnx",
    "STT_JOINER": f"{STT_DIR}/joiner-epoch-99-avg-1.onnx",
    "STT_TOKENS": f"{STT_DIR}/tokens.txt",
}

SYSTEM_PROMPT = (
    "You are a friendly voice assistant having a spoken conversation in English. "
    "Keep every reply short and natural, one or two sentences, the way people talk. "
    "You may use [breath], [laugh] or [hesitate] where a person would. "
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
    if os.environ.get("VUI_DISABLE_CUDNN") == "1":
        import torch

        torch.backends.cudnn.enabled = False

    kit = RoomKit()

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while Vui speaks, so you can
    # interrupt it; without, it is muted during playback to avoid feedback.
    aec = build_aec()
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=SAMPLE_RATE,  # Vui speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )
    logger.info("Audio: AEC=%s, mic muted during playback=%s", type(aec).__name__, mute_mic)

    # --- VAD and STT (sherpa-onnx, CPU: the GPU is Vui's and the LLM's) --------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=env["VAD_MODEL"],
            model_type="ten",
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            speech_pad_ms=300,
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
        )
    )

    # --- Vui: the reply is generated inside the dialogue ------------------------
    voice_name = os.environ.get("VUI_VOICE", "maeve")
    tts = VuiTTSProvider(VuiTTSConfig(voices={voice_name: VuiVoice(voice_name)}))
    include_audio = os.environ.get("VUI_INCLUDE_AUDIO", "1") == "1"

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
    logger.info("LLM: %s (%s), Vui voice: %s", llm_model, ollama_host, voice_name)

    # --- Channels and room -------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad, aec=aec),
        # Vui hears the dialogue: your words, and your voice when include_audio.
        tts_context=TTSContextConfig(include_audio=include_audio),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT),
        )
    )
    await kit.create_room(room_id="local-vui")
    await kit.attach_channel("local-vui", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Vui: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(event, ctx):
        logger.info("Barge-in: Vui stops and keeps only what you heard")

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading Vui, STT and VAD models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    await kit.attach_channel("local-vui", "voice")  # opens the mic
    logger.info("Ready: speak English into the microphone. Ctrl+C to stop.")

    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
