"""Discover the voices each realtime (speech-to-speech) provider supports.

Every ``RealtimeVoiceProvider`` exposes two voice-discovery entry points,
mirroring the model catalog:

- ``available_voices()`` — a curated, *offline* catalog (a classmethod, so it
  needs no API key, network, or provider SDK). Call it to learn which voice ids
  you can pass to ``connect(voice=...)``.
- ``list_voices()`` — a *live* query against the provider's API for the voices
  the account exposes right now. Fixed-voice providers (OpenAI Realtime and
  GPT-Live, Gemini Live, xAI, Deepgram, PersonaPlex) fall back to the curated
  catalog; ElevenLabs queries its live voices endpoint.

Run with:
    uv run python examples/list_voices.py

Environment variables (both optional, for the live ElevenLabs call):
    ELEVENLABS_API_KEY   ElevenLabs API key
    ELEVENLABS_AGENT_ID  ElevenLabs Conversational AI agent id. The realtime
                         provider is built for one agent, so the live
                         ``list_voices()`` call runs only when both are set.
"""

from __future__ import annotations

import asyncio
import os

from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.providers.openai.live import OpenAILiveProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.realtime.provider import VoiceInfo

CURATED_PROVIDERS = {
    "OpenAI Realtime": OpenAIRealtimeProvider,
    "OpenAI GPT-Live": OpenAILiveProvider,
    "Gemini Live": GeminiLiveProvider,
    "xAI Grok": XAIRealtimeProvider,
    "Deepgram Voice Agent": DeepgramAgentProvider,
    "ElevenLabs Agents": ElevenLabsRealtimeProvider,
    "PersonaPlex": PersonaPlexRealtimeProvider,
}


def _format(voice: VoiceInfo) -> str:
    bits = [voice.id]
    if voice.name and voice.name != voice.id:
        bits.append(f"({voice.name})")
    if voice.gender:
        bits.append(f"[{voice.gender}]")
    if voice.description:
        bits.append(f"— {voice.description}")
    return "  " + " ".join(bits)


def show_curated_catalogs() -> None:
    """Print the offline voice catalog for every provider — no key required."""
    for label, provider_cls in CURATED_PROVIDERS.items():
        voices = provider_cls.available_voices()
        print(f"\n{label} — {len(voices)} voices")
        for voice in voices:
            print(_format(voice))


async def show_live_elevenlabs() -> None:
    """Query ElevenLabs' live voices endpoint when an API key and agent id are set."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not (api_key and agent_id):
        print(
            "\n(set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID to see a live "
            "ElevenLabs list_voices() call)"
        )
        return

    provider = ElevenLabsRealtimeProvider(
        ElevenLabsRealtimeConfig(api_key=api_key, agent_id=agent_id)
    )
    try:
        voices = await provider.list_voices()
    except Exception as exc:  # the SDK's ApiError carries the reason in .body
        print(f"\nElevenLabs list_voices() failed: {getattr(exc, 'body', None) or exc}")
        raise SystemExit(1) from None
    print(f"\nElevenLabs live — {len(voices)} voices reported by the account")
    for voice in voices:
        print(_format(voice))


async def main() -> None:
    show_curated_catalogs()
    await show_live_elevenlabs()


if __name__ == "__main__":
    asyncio.run(main())
