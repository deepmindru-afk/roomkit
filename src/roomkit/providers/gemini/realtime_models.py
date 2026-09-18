"""Offline catalog of Google Gemini Live speech-to-speech models.

Hand-maintained list returned by ``GeminiLiveProvider.available_models`` — a
counterpart to ``gemini/models.py`` kept apart from it for the reason the
image catalog gives (RFC §25.6): the sets are disjoint. No id here answers a
generate-content call, and no chat id opens a Live session.

Sourced from Google's Live API docs (ai.google.dev), verified 2026-09-17. No
public aggregator mirrors the Live lineup, so ``scripts/check_models.py``
names this catalog in ``UNMIRRORED_CATALOGS`` rather than comparing it
against a slice that cannot contain it.

Context windows are omitted deliberately: the realtime channel never trims
history against a window, and an unknown number is safer than one nobody
reconciles. Pricing is omitted for a unit reason: Live sessions bill audio
tokens at rates :class:`~roomkit.providers.ai.base.ModelPricing` does not
model, and restating text rates alone would price the wrong unit.

Every Live model accepts image frames (``inject_image`` rides the same
``send_realtime_input`` the video path uses), hence ``supports_vision=True``
throughout — documented behaviour, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.8-live",
        display_name="Gemini 3.8 Live",
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3.8-live-extended-thinking",
        display_name="Gemini 3.8 Live Extended Thinking",
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3.1-flash-live-preview",
        display_name="Gemini 3.1 Flash Live (preview)",
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-12-2025",
        display_name="Gemini 2.5 Flash Native Audio (preview)",
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.0-flash-live-001",
        display_name="Gemini 2.0 Flash Live",
        supports_vision=True,
    ),
]


@dataclass(frozen=True)
class LiveModelProfile:
    """What one Live model's session setup accepts.

    Google changed the Live contract with the 3.8 family: affective dialog was
    removed from the API, proactive audio became permanent (sending it
    explicitly is refused), and ``thinking_config`` moved from a token budget
    to a discrete level that only the extended-thinking model takes. A field
    the target model no longer accepts must not reach the wire, so the setup
    is filtered against this profile before it is built.

    The tool-calling fields live here too rather than in a second table: they
    are decided by the same thing, the model's generation, and splitting them
    would mean two tables to keep in step for one fact.
    """

    affective_dialog: bool
    """``enable_affective_dialog`` may reach the API."""

    proactivity: bool
    """``proactivity`` may reach the API. False where proactive audio is
    permanent and stating it is an error."""

    thinking_budget: bool
    """``thinking_config.thinking_budget`` may reach the API."""

    thinking_level: bool
    """``thinking_config.thinking_level`` may reach the API."""

    blocking_tools: bool
    """``BLOCKING`` is an accepted function-call behaviour. False where the
    model answers a hard error to it."""

    default_tool_behavior: str
    """``Behavior`` value put on a declaration the caller left unqualified."""


LEGACY_PROFILE = LiveModelProfile(
    affective_dialog=True,
    proactivity=True,
    thinking_budget=True,
    thinking_level=False,
    blocking_tools=True,
    default_tool_behavior="BLOCKING",
)
"""Everything before the 3.8 family, and any id this module does not know."""

LIVE_38_PROFILE = LiveModelProfile(
    affective_dialog=False,
    proactivity=False,
    thinking_budget=False,
    thinking_level=False,
    blocking_tools=True,
    default_tool_behavior="NON_BLOCKING",
)
"""``gemini-3.8-live``: no thinking config at all, blocking still tolerated."""

LIVE_38_THINKING_PROFILE = LiveModelProfile(
    affective_dialog=False,
    proactivity=False,
    thinking_budget=False,
    thinking_level=True,
    blocking_tools=False,
    default_tool_behavior="NON_BLOCKING",
)
"""``gemini-3.8-live-extended-thinking``: levels instead of a budget, and a
hard error on a blocking tool."""

THINKING_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
"""Levels the extended-thinking model takes, as the SDK's ``ThinkingLevel``
spells them. The enum also carries ``MINIMAL``, which that model refuses: it
is rejected here rather than spent on a round trip."""


def live_model_profile(model: str) -> LiveModelProfile:
    """Return the session-setup profile for *model*.

    Resolution is by family prefix, not by exact id: Google ships new preview
    ids inside a family faster than this catalog is updated, and an id this
    module has never seen must keep the behaviour it has today rather than
    silently lose its configuration. The longest prefix wins, so
    ``gemini-3.8-live-extended-thinking`` resolves to its own profile and not
    to the plain 3.8 one it also starts with.
    """
    if model.startswith("gemini-3.8-live-extended-thinking"):
        return LIVE_38_THINKING_PROFILE
    if model.startswith("gemini-3.8-live"):
        return LIVE_38_PROFILE
    return LEGACY_PROFILE
