"""A decision model RoomKit runs itself through llama.cpp, as a classifier (RFC §6.8).

The first question (or :meth:`LlamaCppClassifier.start`) downloads the pinned
llama.cpp build for this machine and the model, starts ``llama-server`` on a
free local port, and asks its System One API (:class:`SystemOneClassifier`);
:meth:`LlamaCppClassifier.close` stops it. It is what
:class:`~roomkit.providers.llamacpp.LlamaCppAIProvider` does for a chat model.

Requires the ``typesafe`` and ``llamacpp`` extras::

    pip install roomkit[typesafe,llamacpp]
"""

from __future__ import annotations

from collections.abc import Mapping

from roomkit.classifiers.base import Answers, Classifier, ClassifierError, Question, State
from roomkit.classifiers.system_one import SystemOneClassifier
from roomkit.providers.ai.base import ProviderError
from roomkit.providers.llamacpp.config import LlamaCppConfig
from roomkit.providers.llamacpp.server import LlamaServer


def _require_httpx() -> None:
    """llama-server's health check needs httpx, which the typesafe extra lacks:
    say so before anything is downloaded or started."""
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "httpx is required for LlamaCppClassifier. "
            "Install it with: pip install roomkit[typesafe,llamacpp]"
        ) from exc


class LlamaCppClassifier(Classifier):
    """Run a decision model locally through llama.cpp, with nothing to install or start.

    The model is a GGUF decision model llama.cpp serves on ``/v1/systemone``,
    LiquidAI's d1 among them. Some decision models (vllm-d3, laya, clef,
    d1-omni) read their whole prompt in one batch: give them a
    ``--ubatch-size`` and ``--batch-size`` that hold it, through
    ``extra_args``.

    Example::

        from roomkit import LlamaCppClassifier
        from roomkit.providers.llamacpp import LlamaCppConfig

        classifier = LlamaCppClassifier(LlamaCppConfig(model="LiquidAI/d1-2B-GGUF:Q8_0"))

    Args:
        config: The model and how to run it, as for ``LlamaCppAIProvider``;
            ``max_tokens`` and ``enable_thinking`` do not apply.
        timeout: Seconds a call may take once the server runs, before
            ``ClassifierError``. The start has its own, ``startup_timeout``.
    """

    def __init__(self, config: LlamaCppConfig, *, timeout: float = 30.0) -> None:
        _require_httpx()
        self._server = LlamaServer(config)
        self._classifier = SystemOneClassifier(
            self._server.base_url.removesuffix("/v1"), model=config.model, timeout=timeout
        )

    async def start(self) -> None:
        """Download what is missing and start the server now, not on the first question.

        Safe to call more than once.
        """
        try:
            await self._server.start()
        except (ProviderError, OSError) as exc:
            raise ClassifierError(str(exc)) from exc

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        await self.start()
        return await self._classifier.classify(state, questions)

    async def close(self) -> None:
        """Close the HTTP client and stop ``llama-server``."""
        try:
            await self._classifier.close()
        finally:
            await self._server.stop()
