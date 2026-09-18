"""LiveConnectConfig construction for the Gemini Live provider.

Pure functions over the model id, the caller's parameters and the model's
:class:`~roomkit.providers.gemini.realtime_models.LiveModelProfile`: nothing
here holds a socket or a session. ``connect`` and ``reconfigure`` call
:func:`build_live_config`; the runtime guard on blocking tool calls calls
:func:`blocking_tool_names` through the same :func:`tool_behavior` the
declarations went out with, so the two can never disagree.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any, cast

from roomkit.providers.gemini.realtime_models import (
    THINKING_LEVELS,
    TOOL_BEHAVIORS,
    LiveModelProfile,
    live_model_profile,
)
from roomkit.providers.gemini.schema import clean_gemini_schema

logger = logging.getLogger("roomkit.providers.gemini.realtime")

__all__ = [
    "blocking_tool_names",
    "build_live_config",
    "debug_enabled",
    "enum_value",
    "genai_types",
    "warn_unsupported",
]


def genai_types() -> Any:
    """The SDK's ``types`` module, resolved at call time.

    google-genai is an optional dependency. Binding ``types`` when this module
    is imported would make it unimportable without the SDK, and would freeze
    whichever module object ``sys.modules`` held at that moment, which is not
    what a test that swaps the SDK in and out expects. A missing SDK is named
    with the extra that installs it, as ``GeminiLiveProvider.__init__`` does.
    """
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for the Gemini Live provider. "
            "Install with: pip install 'roomkit[realtime-gemini]'"
        ) from exc
    return types


def debug_enabled() -> bool:
    """Whether ``ROOMKIT_GEMINI_DEBUG`` asks for the config and event dumps."""
    return os.environ.get("ROOMKIT_GEMINI_DEBUG", "").lower() in {"1", "true", "yes"}


def enum_value(enum_cls: Any, value: Any, field: str) -> str:
    """Normalise *value* to a member of *enum_cls*, or refuse it here.

    google-genai answers an unrecognised enum string with a ``UserWarning``
    and forwards it unchanged, so a typo in ``provider_config`` reaches the
    server and takes the whole setup down with it. Failing at the boundary
    names the field and says what it takes, which is the point of validating
    a developer-supplied value at all.
    """
    candidate = str(value).upper()
    allowed = {str(member.value) for member in enum_cls}
    if candidate not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
    return candidate


def warn_unsupported(
    model: str, field: str, warned: set[str], *, replacement: str | None = None
) -> None:
    """Report a setup field the target model does not take, once per session.

    Dropping it silently would leave a deployment believing a setting is
    in force when it is not; raising would break a config that was valid
    against the model it was written for. ``warned`` is the session's own
    record: a provider-wide one reported the first call of the process
    and stayed silent for every session after it. ``replacement`` names
    what went out instead when the value was replaced rather than dropped:
    "ignored" would be false there, and the one log line an operator reads
    should say what actually went out.
    """
    if field in warned:
        return
    warned.add(field)
    if replacement is None:
        logger.warning(
            "[Gemini] %s is not supported by %s - ignored for this session",
            field,
            model,
        )
    else:
        logger.warning(
            "[Gemini] %s is not supported by %s - sent as %s",
            field,
            model,
            replacement,
        )


def tool_behavior(
    model: str, profile: LiveModelProfile, requested: str | None, warned: set[str]
) -> str:
    """Pick the execution mode a declaration is sent with.

    Left to the server the answer would differ by generation, which is
    how the same tool set works on one model and stalls on another. The
    profile's default is stated instead, and a tool may ask for the other
    mode by carrying ``behavior`` in its dict. Asking for BLOCKING where
    the model answers a hard error to it is downgraded rather than sent:
    a refused setup takes the whole session down with it.
    """
    if requested is None:
        return profile.default_tool_behavior
    behavior = str(requested).upper()
    if behavior not in TOOL_BEHAVIORS:
        raise ValueError(
            f"tool behavior must be one of {sorted(TOOL_BEHAVIORS)}, got {requested!r}"
        )
    if behavior == "BLOCKING" and not profile.blocking_tools:
        warn_unsupported(
            model,
            "BLOCKING tool behavior",
            warned,
            replacement="NON_BLOCKING, which the model does accept",
        )
        return "NON_BLOCKING"
    return behavior


def blocking_tool_names(
    model: str, tools: list[dict[str, Any]] | None, warned: set[str] | None = None
) -> set[str]:
    """Names whose calls the API waits on, so an injection must queue.

    Resolved through the same :func:`tool_behavior` the declarations go
    out with, so the runtime guard and the wire can never disagree about
    which mode a tool is in.
    """
    profile = live_model_profile(model)
    if warned is None:
        warned = set()
    return {
        name
        for tool in tools or []
        if (name := tool.get("name", ""))
        and tool_behavior(model, profile, tool.get("behavior"), warned) == "BLOCKING"
    }


# Setup fields a model generation may refuse, and the profile flag that says
# so. Present in ``provider_config`` and allowed: applied. Present and
# refused: reported once for the session and dropped, rather than reaching an
# API that would take the whole setup down with it.
_PROFILE_GATED: dict[str, str] = {
    "enable_affective_dialog": "affective_dialog",
    "thinking_budget": "thinking_budget",
    "thinking_level": "thinking_level",
    "proactive_audio": "proactivity",
}


def _gated(
    model: str, pc: dict[str, Any], profile: LiveModelProfile, key: str, warned: set[str]
) -> Any:
    """The value of a profile-gated setup field, or None when absent or refused."""
    value = pc.get(key)
    if value is None:
        return None
    if getattr(profile, _PROFILE_GATED[key]):
        return value
    warn_unsupported(model, key, warned)
    return None


# Knobs copied from ``provider_config`` as they are, each through the type the
# API takes. Generation parameters go on the config itself, the timing ones
# on the server VAD.
_GENERATION_FIELDS: dict[str, Callable[[Any], Any]] = {
    "top_p": float,
    "top_k": float,
    "max_output_tokens": int,
    "seed": int,
}
_VAD_TIMING_FIELDS: dict[str, Callable[[Any], Any]] = {
    "silence_duration_ms": int,
    "prefix_padding_ms": int,
}


def _apply_coerced(
    source: dict[str, Any], target: dict[str, Any], fields: dict[str, Callable[[Any], Any]]
) -> None:
    """Copy each key of *fields* present in *source* into *target*, coerced."""
    for key, coerce in fields.items():
        value = source.get(key)
        if value is not None:
            target[key] = coerce(value)


def _sensitivity_enum(value: str, prefix: str) -> str:
    """The full enum name of a VAD sensitivity, from its short form or as given.

    ``LOW`` and ``HIGH`` expand to ``<prefix>_SENSITIVITY_<value>``; a full
    name passes through upper-cased.
    """
    val = str(value).upper()
    if val in ("LOW", "HIGH"):
        return f"{prefix}_SENSITIVITY_{val}"
    return val


def transcription_config(types: Any, options: dict[str, Any] | None) -> Any:
    """Build the inbound transcription config from ``provider_config``.

    Only the inbound side is configurable: language biasing, a custom
    vocabulary and diarization describe the caller's speech, and applying
    them to the model's own transcript would bias it towards words the
    model did not say.
    """
    if not options:
        return types.AudioTranscriptionConfig()

    kwargs: dict[str, Any] = {}

    # A marker object rather than a boolean upstream: present means the
    # server may switch language mid-conversation, absent means it may not.
    if options.get("language_auto"):
        kwargs["language_auto"] = types.LanguageAuto()

    # Hints are a wrapper around the same list of codes that
    # ``language_codes`` takes flat, so the caller writes a plain list
    # either way and the shape is applied here.
    hints = options.get("language_hints")
    if hints:
        kwargs["language_hints"] = types.LanguageHints(language_codes=list(hints))

    for key in ("language_codes", "custom_vocabulary", "adaptation_phrases"):
        value = options.get(key)
        if value:
            kwargs[key] = list(value)
    for key in ("diarization", "word_timestamp"):
        value = options.get(key)
        if value is not None:
            kwargs[key] = bool(value)
    mode = options.get("mode")
    if mode:
        kwargs["mode"] = enum_value(types.AudioTranscriptionConfigMode, mode, "transcription.mode")
    return types.AudioTranscriptionConfig(**kwargs)


def build_live_config(
    model: str,
    *,
    system_prompt: str | None = None,
    voice: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    provider_config: dict[str, Any] | None = None,
    server_vad: bool = True,
    warned: set[str] | None = None,
) -> Any:
    """Build the LiveConnectConfig for *model* from the caller's parameters.

    Shared by ``connect`` and ``reconfigure``. ``warned`` is the
    session's record of the fields already reported as dropped, so a
    reconfigure does not repeat what the connect said; a config built on
    its own reports everything.
    """
    types = genai_types()

    pc = provider_config or {}
    if warned is None:
        warned = set()
    profile = live_model_profile(model)

    # Response modalities: ["AUDIO"], ["TEXT"], or ["AUDIO", "TEXT"]
    # Future: ["VIDEO"] when supported by the API.
    response_modalities = pc.get("response_modalities", ["AUDIO"])

    config: dict[str, Any] = {
        "response_modalities": response_modalities,
        "input_audio_transcription": transcription_config(types, pc.get("transcription")),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
    }

    # --- Voice / language ---
    speech_kwargs: dict[str, Any] = {}
    if voice:
        speech_kwargs["voice_config"] = types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
        )
    language = pc.get("language")
    if language:
        speech_kwargs["language_code"] = language
    if speech_kwargs:
        config["speech_config"] = types.SpeechConfig(**speech_kwargs)

    if system_prompt:
        config["system_instruction"] = system_prompt

    # --- Generation parameters ---
    if temperature is not None:
        config["temperature"] = temperature
    _apply_coerced(pc, config, _GENERATION_FIELDS)

    # --- Affective dialog (expressive/emotional responses) ---
    # Removed from the API with the 3.8 family. A deployment that carried
    # it over from 3.1 keeps working: the field is dropped here rather
    # than refused by the server.
    affective = _gated(model, pc, profile, "enable_affective_dialog", warned)
    if affective is not None:
        config["enable_affective_dialog"] = bool(affective)

    # --- Thinking ---
    # 3.1 took a token budget, extended-thinking takes a discrete level,
    # and plain 3.8 takes neither. Both keys are read so a config can name
    # the one its model understands without branching on the model.
    thinking_kwargs: dict[str, Any] = {}

    budget = _gated(model, pc, profile, "thinking_budget", warned)
    if budget is not None:
        thinking_kwargs["thinking_budget"] = int(budget)

    requested_level = _gated(model, pc, profile, "thinking_level", warned)
    if requested_level is not None:
        level = str(requested_level).upper()
        if level not in THINKING_LEVELS:
            raise ValueError(
                f"thinking_level must be one of {sorted(THINKING_LEVELS)}, got {requested_level!r}"
            )
        thinking_kwargs["thinking_level"] = level

    # Required, not merely accepted: the model refuses the session outright
    # when the level is missing, so a caller who named none still gets one.
    if "thinking_level" not in thinking_kwargs and profile.default_thinking_level:
        thinking_kwargs["thinking_level"] = profile.default_thinking_level

    if thinking_kwargs:
        config["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

    # --- Proactivity (AI can speak without being prompted) ---
    # Permanently on from 3.8: stating it either way is an error there.
    proactive = _gated(model, pc, profile, "proactive_audio", warned)
    if proactive is not None:
        config["proactivity"] = types.ProactivityConfig(proactive_audio=bool(proactive))

    # --- VAD / realtime input config ---
    vad_kwargs: dict[str, Any] = {}
    for key, prefix in (
        ("start_of_speech_sensitivity", "START"),
        ("end_of_speech_sensitivity", "END"),
    ):
        sensitivity = pc.get(key)
        if sensitivity:
            vad_kwargs[key] = _sensitivity_enum(sensitivity, prefix)
    _apply_coerced(pc, vad_kwargs, _VAD_TIMING_FIELDS)

    # When server_vad=True (default), enable automatic activity detection
    # so the provider's server-side VAD handles speech boundaries.
    # When server_vad=False (manual mode), disable it — the channel sends
    # activityStart/activityEnd from local VAD instead.
    if server_vad:
        aad = types.AutomaticActivityDetection(**vad_kwargs)
    else:
        aad = types.AutomaticActivityDetection(disabled=True)
        logger.info("Server-side VAD disabled — using manual mode (local VAD)")

    realtime_input_kwargs: dict[str, Any] = {
        "automatic_activity_detection": aad,
    }
    no_interruption = pc.get("no_interruption")
    if no_interruption:
        realtime_input_kwargs["activity_handling"] = "NO_INTERRUPTION"
    # Which input the server folds into a turn. The default moved to
    # TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO with 3.8, which bills
    # every video frame; a video deployment that does not want that needs
    # the knob.
    turn_coverage = pc.get("turn_coverage")
    if turn_coverage:
        realtime_input_kwargs["turn_coverage"] = enum_value(
            types.TurnCoverage, turn_coverage, "turn_coverage"
        )
    config["realtime_input_config"] = types.RealtimeInputConfig(**realtime_input_kwargs)

    # --- Tools ---
    if tools:
        genai_tools = []
        for tool in tools:
            genai_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            parameters=cast(Any, clean_gemini_schema(tool.get("parameters"))),
                            behavior=tool_behavior(model, profile, tool.get("behavior"), warned),
                        )
                    ]
                )
            )
        config["tools"] = genai_tools

    # --- Session resilience ---
    if not pc.get("preserve_context"):
        config["session_resumption"] = types.SessionResumptionConfig(handle=None)
        config["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        )

    # Debug dump of what we're handing to Gemini Live. Gated on
    # ``ROOMKIT_GEMINI_DEBUG=1`` so prod logs stay clean. Useful
    # for diagnosing why one session invokes tools and another
    # doesn't — compares cleanly across sessions when copy/pasted.
    if debug_enabled():
        log_config_dump(config, system_prompt, tools)

    return types.LiveConnectConfig(**config)


def log_config_dump(
    config: dict[str, Any],
    system_prompt: str | None,
    tools: list[dict[str, Any]] | None,
) -> None:
    """Dump the LiveConnectConfig parameters for diagnostics.

    Called from :func:`build_live_config` when ``ROOMKIT_GEMINI_DEBUG`` is on.
    Logs at INFO so it's visible without raising the root level.
    Sister ``GeminiLiveProvider._log_event`` dumps every server event coming the
    other way (text deltas, tool calls, transcription, errors) so
    you can see the full request/response cycle in the same log.
    """
    # ── System prompt ────────────────────────────────────────────
    # Full body, line-prefixed, with a length header so it's easy
    # to see at a glance. Capped at ~12 KB (~3 K tokens) — beyond
    # that we lose the per-line layout in container logs.
    prompt_len = len(system_prompt) if system_prompt else 0
    logger.info("ROOMKIT_GEMINI_DEBUG: ===== system_prompt (len=%d) =====", prompt_len)
    if system_prompt:
        shown = system_prompt[:12000]
        for line in shown.splitlines():
            logger.info("ROOMKIT_GEMINI_DEBUG: | %s", line)
        if prompt_len > 12000:
            logger.info(
                "ROOMKIT_GEMINI_DEBUG: | … [truncated %d chars] …",
                prompt_len - 12000,
            )
    logger.info("ROOMKIT_GEMINI_DEBUG: ===== /system_prompt =====")

    # ── Tools ────────────────────────────────────────────────────
    # All tool names + one-line descriptions. With 30+ tools this
    # is the single most useful piece of context when diagnosing
    # "model didn't pick the right tool" — you can see at a glance
    # what the model was actually shown.
    tool_count = len(tools or [])
    logger.info("ROOMKIT_GEMINI_DEBUG: ===== tools (count=%d) =====", tool_count)
    properties_without_type: list[str] = []
    for tool in tools or []:
        name = tool.get("name", "?")
        desc = (tool.get("description") or "").splitlines()[0][:140]
        cleaned = clean_gemini_schema(tool.get("parameters")) or {}
        param_count = len(cleaned.get("properties") or {})
        required = cleaned.get("required") or []
        logger.info(
            "ROOMKIT_GEMINI_DEBUG: | %-44s params=%d required=%d desc=%s",
            name,
            param_count,
            len(required),
            desc,
        )
        for prop_name, prop_schema in (cleaned.get("properties") or {}).items():
            if isinstance(prop_schema, dict) and "type" not in prop_schema:
                properties_without_type.append(f"{name}.{prop_name}")
    logger.info("ROOMKIT_GEMINI_DEBUG: ===== /tools =====")

    if properties_without_type:
        logger.warning(
            "ROOMKIT_GEMINI_DEBUG: %d tool properties have NO type after cleaning "
            "(Gemini will silently reject these tools): %s",
            len(properties_without_type),
            properties_without_type[:20],
        )

    # ── Other config ─────────────────────────────────────────────
    speech = config.get("speech_config")
    voice_name = ""
    if speech is not None:
        vc = getattr(speech, "voice_config", None)
        if vc is not None:
            pre = getattr(vc, "prebuilt_voice_config", None)
            if pre is not None:
                voice_name = getattr(pre, "voice_name", "") or ""
    logger.info(
        "ROOMKIT_GEMINI_DEBUG: voice=%r temperature=%s response_modalities=%s "
        "session_resumption=%s context_window_compression=%s",
        voice_name,
        config.get("temperature"),
        config.get("response_modalities"),
        bool(config.get("session_resumption")),
        bool(config.get("context_window_compression")),
    )

    # ── First tool's full cleaned schema (for paranoid review) ──
    if tools:
        first = tools[0]
        cleaned_first = {
            "name": first.get("name"),
            "description": (first.get("description") or "")[:200],
            "parameters": clean_gemini_schema(first.get("parameters")) or {},
        }
        logger.info(
            "ROOMKIT_GEMINI_DEBUG: first_tool_full_schema=%s",
            json.dumps(cleaned_first)[:2000],
        )
