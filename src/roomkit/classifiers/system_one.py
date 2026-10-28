"""Any decision model behind a System One API, as a classifier (RFC §6.8).

Many servers answer typed questions with a decision model on the same API,
``POST {base_url}/v1/systemone``: TypeSafe, a LiteLLM proxy (which also
translates OpenAI's Decisions API), Ollama, llama.cpp's ``llama-server``,
vLLM and vLLM Semantic Router. This classifier asks any of them, through the
TypeSafe SDK, the reference client of that API. A score is the expected
level, read from each level's probability whatever the server puts in its
``score`` field: some send the likeliest level instead.

Requires the ``typesafe`` extra::

    pip install roomkit[typesafe]
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from roomkit.classifiers.base import (
    Answer,
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    ClassifierError,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
    YesNoAnswer,
    check_answers,
)

_NO_KEY = "no-key"
"""What a server that asks for no key is sent: the SDK insists on one."""


def _import_sdk() -> Any:
    """The TypeSafe SDK, or a clear error naming the extra."""
    try:
        import typesafe_sdk

        return typesafe_sdk
    except ImportError as exc:
        raise ImportError(
            "typesafe-sdk is required for SystemOneClassifier. "
            "Install it with: pip install roomkit[typesafe]"
        ) from exc


class SystemOneClassifier(Classifier):
    """Answers on a decision model behind any System One API.

    Example::

        from roomkit import SystemOneClassifier

        local = SystemOneClassifier("http://localhost:11434", model="d3:2b")  # Ollama
        proxy = SystemOneClassifier("http://litellm:4000", model="jev", api_key="sk-...")

    Args:
        base_url: The server's root URL; the call goes to
            ``{base_url}/v1/systemone``.
        model: The decision model, as the server names it, sent with every
            call. Required unless a ``client`` is given, whose own model is
            then sent: the SDK's default is a TypeSafe model.
        api_key: The server's key, when it asks for one; a local server
            asks for none. Never read from the environment.
        timeout: Seconds the whole call may take, retries included, before
            ``ClassifierError``. A local server may load its model on the
            first call.
        client: An ``AsyncTypeSafeClient`` to use instead of one made here
            (it is then the caller's to close); ``base_url`` and
            ``api_key`` are then its own.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        client: Any | None = None,
    ) -> None:
        self._sdk = _import_sdk()
        self._model = model
        self._timeout = timeout
        self._owns_client = client is None
        if client is None and not (base_url and model):
            raise ValueError(
                "SystemOneClassifier needs the server's base_url and a model, or a client"
            )
        self._client = client or self._sdk.AsyncTypeSafeClient(
            base_url=base_url, api_key=api_key or _NO_KEY, timeout=timeout
        )

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        asked = {name: self._question(question) for name, question in questions.items()}
        options = {"model": self._model} if self._model else {}
        try:
            call = self._client.system_one(state, asked, **options)
            response = await asyncio.wait_for(call, self._timeout)
        except TimeoutError as exc:
            raise ClassifierError(f"no answer within {self._timeout} s") from exc
        except self._sdk.TypeSafeError as exc:
            raise ClassifierError(str(exc)) from exc
        answers = Answers()
        for name, question in questions.items():
            answer = response.answers.get(name)
            if _fits(question, answer):
                answers[name] = _answer(question, answer)
        return check_answers(questions, answers)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _question(self, question: Question) -> Any:
        sdk = self._sdk
        if isinstance(question, ChoiceQuestion):
            return sdk.Choice(instructions=question.instructions, criteria=dict(question.options))
        if isinstance(question, ScoreQuestion):
            return sdk.Score(instructions=question.instructions, criteria=list(question.levels))
        described = {"true": question.yes, "false": question.no}
        criteria = {k: v for k, v in described.items() if v is not None} or None
        return sdk.Noul(instructions=question.instructions, criteria=criteria)


def _fits(question: Question, answer: Any) -> bool:
    """Whether a server's answer is of its question's type, and a choice one
    of the options asked; ``check_answers`` refuses a question left without."""
    kind = getattr(answer, "type", None)
    if isinstance(question, ChoiceQuestion):
        return kind == "choice" and answer.choice in question.options
    if isinstance(question, ScoreQuestion):
        return kind == "score"
    return kind == "noul"


def _answer(question: Question, answer: Any) -> Answer:
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(answer.choice, dict(answer.probabilities))
    if isinstance(question, ScoreQuestion):
        levels = tuple(answer.probabilities.get(i, 0.0) for i in range(len(question.levels)))
        return ScoreAnswer(_expected_level(levels, answer.score), levels)
    return YesNoAnswer(answer.noul)


def _expected_level(levels: tuple[float, ...], sent: float) -> float:
    """The probability-weighted level, the score RFC §6.8 defines; the server's
    own when it sent no probabilities to weigh."""
    total = sum(levels)
    if total <= 0:
        return sent
    return sum(index * p for index, p in enumerate(levels)) / total
