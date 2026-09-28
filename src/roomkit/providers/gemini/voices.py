"""Curated catalog of Google Gemini prebuilt voices.

Hand-maintained, offline list returned by both
``GeminiLiveProvider.available_voices`` and
``GeminiTTSProvider.available_voices`` — one table, because Live native-audio
output and the standalone TTS models draw from the same 30 prebuilt voices.
Sourced from the Gemini API speech-generation docs; refresh against the docs
when it changes. Gender is not documented there, so only the single-word
characterization is captured.

The live catalog is far larger — 2,089 voices on 2026-09-27, 68 of them
``fr-CA`` — and :func:`voice_info_from_catalog` maps its entries;
``GeminiTTSProvider.list_voices`` reads it.
"""

from __future__ import annotations

from typing import Any

from roomkit.voice.realtime.provider import VoiceInfo

_CATALOG_ATTRIBUTES = ("persona", "context", "region_code", "model", "key")
"""Catalog fields kept under ``VoiceInfo.attributes``, under Google's names."""


def voice_info_from_catalog(voice: Any) -> VoiceInfo:
    """A ``GET /v1beta/voices`` entry as a :class:`VoiceInfo`.

    The fields every catalog shares become fields; what only Google reports
    stays under ``attributes``, including ``type`` (``prebuilt``, ``prompted``,
    ``replicated``) and, for a custom voice, when it expires.
    """
    attributes = {
        name: str(value)
        for name in _CATALOG_ATTRIBUTES
        if (value := getattr(voice, name, None)) not in (None, "")
    }
    voice_type = getattr(voice, "type", None)
    if voice_type is not None:
        attributes["type"] = str(getattr(voice_type, "value", voice_type))
    pitch = getattr(voice, "pitch", None)
    if pitch is not None:
        attributes["pitch"] = str(getattr(pitch, "value", pitch))
    expires = getattr(voice, "expire_time", None)
    if expires is not None:
        attributes["expire_time"] = expires.isoformat()
    return VoiceInfo(
        # An unstored custom voice has no id, only its key, which the TTS takes.
        id=getattr(voice, "id", None) or voice.key,
        name=getattr(voice, "display_name", None) or None,
        language=getattr(voice, "language_code", None) or None,
        gender=getattr(voice, "gender", None) or None,
        accent=getattr(voice, "accent", None) or None,
        description=getattr(voice, "description", None) or None,
        attributes=attributes,
    )


VOICES: list[VoiceInfo] = [
    VoiceInfo(id="Zephyr", name="Zephyr", description="Bright"),
    VoiceInfo(id="Puck", name="Puck", description="Upbeat"),
    VoiceInfo(id="Charon", name="Charon", description="Informative"),
    VoiceInfo(id="Kore", name="Kore", description="Firm"),
    VoiceInfo(id="Fenrir", name="Fenrir", description="Excitable"),
    VoiceInfo(id="Leda", name="Leda", description="Youthful"),
    VoiceInfo(id="Orus", name="Orus", description="Firm"),
    VoiceInfo(id="Aoede", name="Aoede", description="Breezy"),
    VoiceInfo(id="Callirrhoe", name="Callirrhoe", description="Easy-going"),
    VoiceInfo(id="Autonoe", name="Autonoe", description="Bright"),
    VoiceInfo(id="Enceladus", name="Enceladus", description="Breathy"),
    VoiceInfo(id="Iapetus", name="Iapetus", description="Clear"),
    VoiceInfo(id="Umbriel", name="Umbriel", description="Easy-going"),
    VoiceInfo(id="Algieba", name="Algieba", description="Smooth"),
    VoiceInfo(id="Despina", name="Despina", description="Smooth"),
    VoiceInfo(id="Erinome", name="Erinome", description="Clear"),
    VoiceInfo(id="Algenib", name="Algenib", description="Gravelly"),
    VoiceInfo(id="Rasalgethi", name="Rasalgethi", description="Informative"),
    VoiceInfo(id="Laomedeia", name="Laomedeia", description="Upbeat"),
    VoiceInfo(id="Achernar", name="Achernar", description="Soft"),
    VoiceInfo(id="Alnilam", name="Alnilam", description="Firm"),
    VoiceInfo(id="Schedar", name="Schedar", description="Even"),
    VoiceInfo(id="Gacrux", name="Gacrux", description="Mature"),
    VoiceInfo(id="Pulcherrima", name="Pulcherrima", description="Forward"),
    VoiceInfo(id="Achird", name="Achird", description="Friendly"),
    VoiceInfo(id="Zubenelgenubi", name="Zubenelgenubi", description="Casual"),
    VoiceInfo(id="Vindemiatrix", name="Vindemiatrix", description="Gentle"),
    VoiceInfo(id="Sadachbia", name="Sadachbia", description="Lively"),
    VoiceInfo(id="Sadaltager", name="Sadaltager", description="Knowledgeable"),
    VoiceInfo(id="Sulafat", name="Sulafat", description="Warm"),
]
