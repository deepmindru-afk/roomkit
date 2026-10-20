"""RoomKit -- Local French voice assistant with Gradium's Phonon TTS, on the CPU.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and French speech-to-text (CPU)
  - a local LLM that RoomKit runs itself through llama.cpp (or Ollama)
  - Gradium's Phonon, a 40M-parameter French TTS, on the CPU through its Rust runtime

    Mic → [AEC] → VAD → sherpa-onnx STT (fr) → local LLM → [StripEmoji] → Phonon (fr) → Speaker

Phonon needs neither torch nor a GPU. Measured on a desktop CPU with two
threads, ``phonon-40m-fr`` runs about ten times faster than real time and
starts speaking ~50 ms after it gets a sentence.

Requirements:
    Nothing to run beside it: the LLM, Qwen3-4B-Instruct (Q4_K_M), is downloaded
    on first run with the llama.cpp build for this machine, and RoomKit starts
    and stops llama-server itself (~4 GB of VRAM on a GPU, or the CPU, slower).
    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

    The Phonon 40m models are gated: accept their terms on
    https://huggingface.co/gradium/phonon-40m-fr (its licence is stated
    there), then run ``hf auth login`` or set HF_TOKEN. The weights (75 MB)
    and the voice you pick download on first run.

Models (download once, into examples/models/):
    mkdir -p examples/models && cd examples/models
    # VAD: TEN-VAD
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
    # STT: Kroko, a Zipformer transducer, French, streaming
    # (model license: huggingface.co/Banafo/Kroko-ASR)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06.tar.bz2
    cd ../..

Run (from the repository root):
    uv run --extra local-audio --extra webrtc-aec --extra llamacpp \\
        --extra sherpa-onnx --extra phonon \\
        python examples/voice_local_phonon.py

    With the console dashboard (add --extra console):
    CONSOLE=1 uv run ... --extra console python examples/voice_local_phonon.py

Environment variables:
    --- Phonon ---
    PHONON_MODEL        Hugging Face repo or local checkpoint
                        (default: hf://gradium/phonon-40m-fr; also
                        hf://gradium/phonon-40m-multilingual)
    PHONON_LANGUAGE     How numbers, dates and symbols are spelled out:
                        en | fr | de | es | pt (default: fr)
    PHONON_VOICE        A voice from huggingface.co/gradium/phonon-voices, by
                        name (Noémie, Lucien...) or id (default: the model's own)
    PHONON_SPEED        Speaking rate: negative is faster, positive slower,
                        e.g. -0.2 (default: the model's)
    PHONON_THREADS      CPU threads of the runtime (default: 2)

    --- LLM ---
    LLM_BACKEND         llamacpp | ollama (default: llamacpp)
    LLM_MODEL           llamacpp: GGUF "repo:quant" or .gguf path
                        (default: unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M);
                        ollama: model name (default: qwen3:4b-instruct; add
                        --extra ollama to the run command)
    OLLAMA_HOST         Ollama server (default: http://localhost:11434)
    LLM_MAX_TOKENS      Max response tokens (default: 200)
    SYSTEM_PROMPT       Custom system prompt

    --- Barge-in and turns ---
    INTERRUPTION        semantic | confirmed | immediate | disabled (default: semantic:
                        an acknowledgement like "d'accord" does not stop the voice)
    TURN_DETECTOR       smart-turn | 0 (default: 0; add --extra smart-turn).
                        TURN_MODEL names the .onnx file.
    VOICE_DEBUG         1 to log turn-taking decisions

    --- STT and VAD (sherpa-onnx, CPU) ---
    MODELS_DIR          Where the models were downloaded (default: examples/models)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)

    --- Other ---
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import ptts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_aec,
    build_interruption,
    build_turn_detector,
    enable_voice_debug,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIProvider
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.filters import StripEmoji
from roomkit.voice.tts.phonon import SAMPLE_RATE, PhononTTSConfig, PhononTTSProvider

logger = setup_logging("voice_local_phonon")
logging.getLogger("httpx").setLevel(logging.WARNING)

MIC_RATE = 16000
BLOCK_MS = 20

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
STT_DIR = "sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06"

SYSTEM_PROMPT = (
    "You are a warm voice assistant having a spoken conversation. "
    "Always answer in French, in one or two short, natural sentences, the way people talk. "
    "Never use lists, markdown or emojis."
)


def build_llm() -> AIProvider:
    """The local LLM: llama.cpp run by RoomKit, or an Ollama server."""
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "200"))
    if os.environ.get("LLM_BACKEND", "llamacpp") == "ollama":
        return OllamaAIProvider(
            OllamaConfig(
                host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                model=os.environ.get("LLM_MODEL", "qwen3:4b-instruct"),
                max_tokens=max_tokens,
                think=False,
            )
        )
    return LlamaCppAIProvider(
        LlamaCppConfig(
            model=os.environ.get("LLM_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"),
            max_tokens=max_tokens,
            enable_thinking=False,
        )
    )


async def start_llm(provider: AIProvider) -> None:
    """Load a llama.cpp model before the first word; an Ollama server is already up."""
    if isinstance(provider, LlamaCppAIProvider):
        await provider.start()


def model_paths() -> dict[str, str]:
    """The VAD and STT model files in MODELS_DIR; exits naming the missing ones."""
    models_dir = Path(os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR))
    paths = {
        "vad": models_dir / "ten-vad.onnx",
        "encoder": models_dir / STT_DIR / "encoder.onnx",
        "decoder": models_dir / STT_DIR / "decoder.onnx",
        "joiner": models_dir / STT_DIR / "joiner.onnx",
        "tokens": models_dir / STT_DIR / "tokens.txt",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        logger.error("Model files not found, see 'Models' at the top of this example:")
        for path in missing:
            logger.error("  %s", path)
        sys.exit(1)
    return {name: str(path) for name, path in paths.items()}


def build_tts() -> PhononTTSProvider:
    """Phonon in French; the runtime's thread pool is sized before the model loads."""
    ptts.set_num_threads(int(os.environ.get("PHONON_THREADS", "2")))
    speed = os.environ.get("PHONON_SPEED")
    config = PhononTTSConfig(
        model=os.environ.get("PHONON_MODEL", "hf://gradium/phonon-40m-fr"),
        language=os.environ.get("PHONON_LANGUAGE", "fr"),
        voice=os.environ.get("PHONON_VOICE") or None,
        duration_delta=float(speed) if speed else None,
    )
    logger.info("Phonon: %s, voice %s", config.model, config.voice or "of the model")
    return PhononTTSProvider(config)


async def main() -> None:
    models = model_paths()
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while the assistant speaks, so
    # you can interrupt it; without, it is muted during playback.
    aec = build_aec(MIC_RATE, BLOCK_MS)
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=SAMPLE_RATE,  # Phonon speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=aec is None,
        aec=aec,
    )

    # --- VAD and STT (sherpa-onnx, CPU) ----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=models["vad"],
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
            encoder=models["encoder"],
            decoder=models["decoder"],
            joiner=models["joiner"],
            tokens=models["tokens"],
            sample_rate=MIC_RATE,
            provider="cpu",
            # Kroko French drops the last word ("midi" -> "mid") below ~1.5 s
            # of tail silence; the padding costs compute, not waiting time.
            tail_padding_s=1.5,
        )
    )

    tts = build_tts()
    ai_provider = build_llm()
    logger.info("LLM: %s (%s)", ai_provider.model_name, ai_provider.name)

    # --- Channels and room -------------------------------------------------------
    turn_detector = build_turn_detector()
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        # No aec= here: the backend feeds the echo reference itself and
        # reports NATIVE_AEC, so a pipeline copy would never run.
        pipeline=AudioPipelineConfig(vad=vad, turn_detector=turn_detector),
        # The LLM adds emoji despite the prompt; spoken, they sound wrong.
        tts_filter=StripEmoji(),
        interruption=build_interruption(),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT),
        )
    )
    await kit.create_room(room_id="local-phonon")
    await kit.attach_channel("local-phonon", "ai", category=ChannelCategory.INTELLIGENCE)

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

    if os.environ.get("VOICE_DEBUG") == "1":
        enable_voice_debug(kit)

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading the LLM, Phonon, STT and VAD models...")
    try:
        await asyncio.gather(stt.warmup(), tts.warmup(), start_llm(ai_provider))
        await kit.attach_channel("local-phonon", "voice")  # opens the mic
        logger.info("Ready: speak French into the microphone. Ctrl+C to stop.")
        await run_until_stopped(kit, cleanup=console_cleanup)
    finally:
        await ai_provider.close()  # stops llama-server


if __name__ == "__main__":
    asyncio.run(main())
