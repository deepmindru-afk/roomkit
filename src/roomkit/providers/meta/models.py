"""Offline metadata for Meta's Muse Spark chat models.

Hand-maintained list returned by ``MetaAIProvider.available_models`` — the
context windows and rates roomkit needs before it can make a network call.
:meth:`~roomkit.providers.meta.ai.MetaAIProvider.list_models` reads the
account's ``/v1/models`` for what is live.

Sourced from the Meta Model API docs (dev.meta.ai/docs/models and
/docs/pricing-rate-limits), verified 2026-09-27, and the ids against a live
``GET /v1/models`` the same day.

Every Muse Spark model reads images (and video, PDF and audio, which roomkit's
AI parts do not carry), calls tools and always reasons — ``"thinking"`` is on
every entry, and ``reasoning_effort="none"`` is refused by the service. The
reasoning itself is not returned by Chat Completions, only counted.

The ``-contributor`` ids are the same models at a fraction of the price,
because Meta trains its models on their traffic. Meta charges no long-context
premium.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 9, 27)
_WINDOW = 1_048_576
_CAPS = ["tools", "thinking"]

_STANDARD = ModelPricing(
    input_per_million=1.25,
    output_per_million=4.25,
    cache_read_per_million=0.15,
    verified=_VERIFIED,
)
_CONTRIBUTOR = ModelPricing(
    input_per_million=0.10,
    output_per_million=0.20,
    cache_read_per_million=0.002,
    verified=_VERIFIED,
)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="muse-spark-1.3",
        display_name="Muse Spark 1.3",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.2",
        display_name="Muse Spark 1.2",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.1",
        display_name="Muse Spark 1.1",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.3-contributor",
        display_name="Muse Spark 1.3 (Contributor: Meta trains on the traffic)",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_CONTRIBUTOR,
    ),
    ModelInfo(
        id="muse-spark-1.2-contributor",
        display_name="Muse Spark 1.2 (Contributor: Meta trains on the traffic)",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_CONTRIBUTOR,
    ),
]
