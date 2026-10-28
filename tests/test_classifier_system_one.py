"""SystemOneClassifier (RFC §6.8): any server that speaks the System One API
(``POST {base_url}/v1/systemone``), through the TypeSafe SDK. A mock transport
stands in for the server; what crosses it is the SDK's own wire."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from roomkit import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    ScoreQuestion,
    SystemOneClassifier,
    YesNoAnswer,
    YesNoQuestion,
)

sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

QUESTIONS = {
    "asks": YesNoQuestion("Does the last message ask for something?", no="a reaction"),
    "language": ChoiceQuestion("Which language?", {"fr": "French", "en": "English"}),
    "urgency": ScoreQuestion("How urgent?", ("can wait", "within the day", "right now")),
}

LEVELS = {"0": 0.28, "1": 0.32, "2": 0.40}
"""Each level's probability as a server sends it: the expected level is 1.12."""


def _answers(score: float) -> dict[str, Any]:
    return {
        "asks": {"type": "noul", "noul": 0.83},
        "language": {
            "type": "choice",
            "choice": "fr",
            "confidence": 0.9,
            "probabilities": {"fr": 0.94, "en": 0.06},
        },
        "urgency": {
            "type": "score",
            "score": score,
            "confidence": 0.2,
            "legend": {"0": "can wait", "1": "within the day", "2": "right now"},
            "probabilities": LEVELS,
        },
    }


class _Server:
    """A System One server behind a mock transport, recording what it is sent."""

    def __init__(self, answers: dict[str, Any] | None = None, status: int = 200) -> None:
        self.answers = answers or _answers(1.12)
        self.status = status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(
                self.status, json={"error": {"message": "not a decision model"}}
            )
        return httpx2.Response(
            200,
            json={"model": "m", "answers": self.answers, "usage": {"input_tokens": 1}},
        )

    def client(self, base_url: str = "http://server.test") -> Any:
        return sdk.AsyncTypeSafeClient(
            api_key="k", base_url=base_url, transport=httpx2.MockTransport(self)
        )


async def test_the_questions_reach_the_server_with_the_model() -> None:
    server = _Server()
    await SystemOneClassifier(client=server.client(), model="d3:2b").classify(
        {"last": "Nova?"}, QUESTIONS
    )

    [request] = server.requests
    assert (request.method, str(request.url)) == ("POST", "http://server.test/v1/systemone")
    body = json.loads(request.content)
    assert body["model"] == "d3:2b"
    assert body["state"] == {"last": "Nova?"}
    assert body["questions"] == {
        "asks": {
            "type": "noul",
            "instructions": "Does the last message ask for something?",
            "criteria": {"false": "a reaction"},
        },
        "language": {
            "type": "choice",
            "instructions": "Which language?",
            "criteria": {"fr": "French", "en": "English"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["can wait", "within the day", "right now"],
        },
    }


@pytest.mark.parametrize("sent", [1, 1.12], ids=["an-integer", "the-expected-level"])
async def test_the_score_is_the_expected_level_whatever_the_server_sends(sent: float) -> None:
    """llama.cpp b11954 sent the likeliest level's index; Jev, Ollama and
    LiteLLM send the expected level. Every one sends each level's
    probability, which the score is read from (RFC §6.8)."""
    answers = await SystemOneClassifier(client=_Server(_answers(sent)).client()).classify(
        "x", QUESTIONS
    )

    urgency = answers["urgency"]
    assert urgency.score == pytest.approx(1.12)  # type: ignore[union-attr]
    assert urgency.probabilities == (0.28, 0.32, 0.40)  # type: ignore[union-attr]
    assert answers["asks"] == YesNoAnswer(0.83)
    assert answers["language"] == ChoiceAnswer("fr", {"fr": 0.94, "en": 0.06})


async def test_a_server_that_refuses_is_a_classifier_error() -> None:
    server = _Server(status=501)
    with pytest.raises(ClassifierError, match="501"):
        await SystemOneClassifier(client=server.client()).classify("x", QUESTIONS)


@pytest.mark.parametrize(
    "language",
    [
        {"type": "noul", "noul": 0.6},
        {"type": "choice", "choice": "de", "confidence": 0.9, "probabilities": {"de": 1.0}},
    ],
    ids=["another-type", "an-option-not-asked"],
)
async def test_an_answer_that_does_not_fit_its_question_is_a_classifier_error(
    language: dict[str, Any],
) -> None:
    server = _Server({**_answers(1.12), "language": language})
    with pytest.raises(ClassifierError, match="language"):
        await SystemOneClassifier(client=server.client()).classify("x", QUESTIONS)


async def test_it_makes_its_client_for_the_server_and_closes_only_that(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []

    class _Made:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    def factory(**kwargs: Any) -> _Made:
        made.append(kwargs)
        return _Made()

    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", factory)
    local = SystemOneClassifier("http://localhost:11434", model="d1:2b", timeout=2.0)
    SystemOneClassifier("https://decisions.example", model="m", api_key="key")
    # Without one, the SDK would send its own default, a TypeSafe model.
    with pytest.raises(ValueError, match="model"):
        SystemOneClassifier("http://localhost:11434")

    # A local server asks for no key; the SDK insists on one, so it gets a
    # stand-in, never TYPESAFE_API_KEY, which is another server's.
    assert made[0]["base_url"] == "http://localhost:11434"
    assert made[0]["api_key"] not in (None, "")
    assert made[0]["timeout"] == 2.0
    assert made[1]["api_key"] == "key"
    await local.close()
    assert local._client.closed  # type: ignore[attr-defined]

    lent = _Made()
    await SystemOneClassifier(client=lent).close()
    assert not lent.closed


def test_without_the_sdk_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ImportError, match=r"roomkit\[typesafe\]"):
        SystemOneClassifier("http://localhost:11434", model="d3:2b")
