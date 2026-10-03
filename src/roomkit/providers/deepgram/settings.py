"""Translation of RoomKit's session arguments into Deepgram's ``Settings`` payload.

Pure functions over a :class:`DeepgramAgentConfig` and the per-session
``provider_config`` dict: no sockets, no session state. The provider calls
:func:`build_settings` once at connect time, then :func:`patch_think` and
:func:`patch_speak` when a live session is reconfigured.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from roomkit.providers.ai.tool_declaration import ToolNameRule, declared_parameters
from roomkit.providers.anthropic.request import ANTHROPIC_TOOL_NAMES
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.gemini.schema import GEMINI_TOOL_NAMES
from roomkit.providers.openai.ai import OPENAI_TOOL_NAMES


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Deepgram's provider objects are type-discriminated unions, so an override
    that names a different ``type`` replaces the whole dict instead of merging:
    blending fields from two vendors (an Aura ``model`` next to an ElevenLabs
    ``model_id``) produces a payload Deepgram rejects.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if (
            isinstance(current, dict)
            and isinstance(value, dict)
            and ("type" not in value or value.get("type") == current.get("type"))
        ):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


_THINK_TOOL_NAMES: dict[str, ToolNameRule] = {
    "open_ai": OPENAI_TOOL_NAMES,
    "anthropic": ANTHROPIC_TOOL_NAMES,
    "google": GEMINI_TOOL_NAMES,
}
"""The tool-name rule of each think provider whose vendor RoomKit knows."""


def think_tool_names(think: dict[str, Any]) -> ToolNameRule | None:
    """The tool names a Think block's LLM accepts: its vendor's rule, none
    for a vendor RoomKit does not know or a custom endpoint (RFC §6.7).

    Deepgram applies the settings whatever the names and passes the functions
    to the think provider, which refuses a name its vendor refuses on the
    first turn (``THINK_REQUEST_FAILED``, measured with ``open_ai``).
    """
    if think.get("endpoint"):
        return None
    return _THINK_TOOL_NAMES.get((think.get("provider") or {}).get("type", ""))


def check_think_functions(think: dict[str, Any]) -> None:
    """Raise, before the block is sent, for a function name the Think block's
    LLM refuses (RFC §6.7): the block as it goes out, so a think provider or
    an endpoint set through ``settings`` or switched by a reconfigure is the
    one whose rule applies."""
    rule = think_tool_names(think)
    if rule is not None:
        rule.check(function.get("name", "") for function in think.get("functions") or [])


def format_functions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Project RoomKit tool dicts to Deepgram's ``think.functions`` shape.

    Tool dicts reaching the provider carry extra keys the caller uses elsewhere
    (notably ``tags``, added for cross-lingual Tool Search); Deepgram rejects
    unknown fields, so only name/description/parameters survive. A function
    carrying an ``endpoint`` is executed by Deepgram server-side; without one it
    comes back as a ``client_side`` call for RoomKit to run.
    """
    functions: list[dict[str, Any]] = []
    for tool in tools or []:
        name = tool.get("name")
        if not name:
            continue
        function: dict[str, Any] = {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": declared_parameters(tool.get("parameters")),
        }
        if tool.get("endpoint"):
            function["endpoint"] = tool["endpoint"]
        functions.append(function)
    return functions


def build_think(
    cfg: DeepgramAgentConfig,
    *,
    system_prompt: str | None,
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``agent.think`` block — the LLM stage."""
    provider: dict[str, Any] = {
        "type": pc.get("think_provider", cfg.think_provider),
        "model": pc.get("think_model", cfg.think_model),
    }
    if temperature is not None:
        provider["temperature"] = temperature

    think: dict[str, Any] = {"provider": provider}
    if pc.get("think_endpoint"):
        think["endpoint"] = pc["think_endpoint"]
    if pc.get("context_length") is not None:
        think["context_length"] = pc["context_length"]
    if system_prompt:
        think["prompt"] = system_prompt
    functions = format_functions(tools)
    if functions:
        think["functions"] = functions
    return think


def build_speak(
    cfg: DeepgramAgentConfig, *, voice: str | None, pc: dict[str, Any]
) -> dict[str, Any]:
    """Build the ``agent.speak`` block — the TTS stage.

    A ``speak_provider`` dict (per-session or from config) is passed through
    verbatim, so any TTS vendor Deepgram supports can be named with that
    vendor's own field shape. It takes precedence over ``voice`` and
    ``speak_model``, which describe Aura voices. ``speak_endpoint`` rides
    alongside — the URL + auth headers that BYO-key vendors require.
    """
    custom = pc.get("speak_provider", cfg.speak_provider)
    if custom is not None:
        provider: dict[str, Any] = deepcopy(custom)
    else:
        provider = {
            "type": "deepgram",
            "model": voice or pc.get("speak_model", cfg.speak_model),
        }
        language = pc.get("speak_language", cfg.speak_language)
        if language:
            provider["language"] = language
    speak: dict[str, Any] = {"provider": provider}
    endpoint = pc.get("speak_endpoint", cfg.speak_endpoint)
    if endpoint:
        speak["endpoint"] = deepcopy(endpoint)
    return speak


def patch_think(
    current: dict[str, Any],
    *,
    system_prompt: str | None,
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Patch a live Think block while preserving every omitted field.

    Deepgram's ``UpdateThink`` replaces the whole block.  Starting from the
    session's current value is therefore required: rebuilding from provider
    defaults would silently discard per-session models, endpoints and context
    settings whenever a skill only changes the prompt or tools.
    """
    think = deepcopy(current)
    provider = dict(think.get("provider") or {})

    if "think_provider" in pc:
        provider["type"] = pc["think_provider"]
    if "think_model" in pc:
        provider["model"] = pc["think_model"]
    if temperature is not None:
        provider["temperature"] = temperature
    think["provider"] = provider

    if "think_endpoint" in pc:
        if pc["think_endpoint"]:
            think["endpoint"] = pc["think_endpoint"]
        else:
            think.pop("endpoint", None)
    if "context_length" in pc:
        if pc["context_length"] is not None:
            think["context_length"] = pc["context_length"]
        else:
            think.pop("context_length", None)

    if system_prompt is not None:
        if system_prompt:
            think["prompt"] = system_prompt
        else:
            think.pop("prompt", None)
    if tools is not None:
        functions = format_functions(tools)
        if functions:
            think["functions"] = functions
        else:
            think.pop("functions", None)
    check_think_functions(think)
    return think


def patch_speak(
    current: dict[str, Any],
    *,
    voice: str | None,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Patch a live Speak block while preserving omitted provider settings.

    A ``speak_provider`` in ``pc`` replaces the provider block wholesale —
    provider types are discriminated unions, and merging fields across vendors
    would send Deepgram a hybrid it rejects. ``voice`` and ``speak_model`` name
    Aura voices, so they only apply when the resulting provider is Deepgram's.
    """
    speak = deepcopy(current)
    if pc.get("speak_provider") is not None:
        provider: dict[str, Any] = deepcopy(pc["speak_provider"])
    else:
        provider = dict(speak.get("provider") or {})
    if provider.get("type", "deepgram") == "deepgram":
        if voice is not None:
            provider["model"] = voice
        elif "speak_model" in pc:
            provider["model"] = pc["speak_model"]
        if "speak_language" in pc:
            if pc["speak_language"]:
                provider["language"] = pc["speak_language"]
            else:
                provider.pop("language", None)
    speak["provider"] = provider
    if "speak_endpoint" in pc:
        if pc["speak_endpoint"]:
            speak["endpoint"] = deepcopy(pc["speak_endpoint"])
        else:
            speak.pop("endpoint", None)
    return speak


def build_listen(cfg: DeepgramAgentConfig, *, pc: dict[str, Any]) -> dict[str, Any]:
    """Build the ``agent.listen`` block — the speech-to-text stage."""
    provider: dict[str, Any] = {
        "type": "deepgram",
        "model": pc.get("listen_model", cfg.listen_model),
    }
    version = pc.get("listen_version", cfg.listen_version)
    if version:
        provider["version"] = version
    language = pc.get("listen_language", cfg.listen_language)
    if language:
        provider["language"] = language
    if pc.get("keyterms"):
        provider["keyterms"] = list(pc["keyterms"])
    if pc.get("smart_format") is not None:
        provider["smart_format"] = bool(pc["smart_format"])
    return {"provider": provider}


def build_settings(
    cfg: DeepgramAgentConfig,
    *,
    system_prompt: str | None,
    voice: str | None,
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    input_sample_rate: int,
    output_sample_rate: int,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Build the full ``Settings`` message sent once, right after connecting.

    ``pc["settings"]`` is deep-merged last, so an integrator can reach a field
    this module does not model without losing the rest of the payload.
    """
    output: dict[str, Any] = {
        "encoding": pc.get("output_encoding", "linear16"),
        "sample_rate": output_sample_rate,
        "container": pc.get("output_container", "none"),
    }
    if pc.get("output_bitrate") is not None:
        output["bitrate"] = int(pc["output_bitrate"])

    agent: dict[str, Any] = {
        "listen": build_listen(cfg, pc=pc),
        "think": build_think(
            cfg, system_prompt=system_prompt, tools=tools, temperature=temperature, pc=pc
        ),
        "speak": build_speak(cfg, voice=voice, pc=pc),
    }
    greeting = pc.get("greeting", cfg.greeting)
    if greeting:
        agent["greeting"] = greeting

    settings: dict[str, Any] = {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": pc.get("input_encoding", "linear16"),
                "sample_rate": input_sample_rate,
            },
            "output": output,
        },
        "agent": agent,
    }
    if pc.get("tags"):
        settings["tags"] = list(pc["tags"])
    if pc.get("settings"):
        settings = deep_merge(settings, pc["settings"])
    # The merged block: an override may have changed the think stage, or
    # broken it, which the provider reports before the socket opens.
    merged_agent = settings.get("agent")
    think = merged_agent.get("think") if isinstance(merged_agent, dict) else None
    if isinstance(think, dict):
        check_think_functions(think)
    return settings
