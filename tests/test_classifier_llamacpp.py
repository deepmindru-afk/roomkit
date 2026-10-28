"""LlamaCppClassifier (RFC §6.8): a decision model RoomKit runs itself through
llama.cpp, as LlamaCppAIProvider runs a chat model. A stand-in llama-server
answers /health and /v1/systemone."""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from roomkit import ClassifierError, LlamaCppClassifier, ScoreQuestion, YesNoQuestion
from roomkit.providers.llamacpp import LlamaCppConfig
from roomkit.providers.llamacpp import server as server_mod

pytest.importorskip("typesafe_sdk")
pytest.importorskip("httpx")

_FAKE_SERVER = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({{"start": args}}) + "\\n")
    if os.environ.get("FAKE_FAIL"):
        print("error: unknown model architecture: 'lfm2-d9'", flush=True)
        sys.exit(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json({{"status": "ok"}})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(os.environ["FAKE_LOG"], "a") as f:
                f.write(json.dumps({{"path": self.path, "request": request}}) + "\\n")
            # As llama.cpp b11954 answered: the score is the likeliest level.
            self._json({{"model": "d1", "usage": {{"input_tokens": 9, "output_tokens": 0}},
                "answers": {{
                    "asks": {{"type": "noul", "noul": 0.91}},
                    "urgency": {{"type": "score", "score": 2, "confidence": 0.5,
                                "legend": {{"0": "later", "1": "today", "2": "now"}},
                                "probabilities": {{"0": 0.1, "1": 0.3, "2": 0.6}}}},
                }}}})

    print("server is listening", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)

QUESTIONS = {
    "asks": YesNoQuestion("Does the last message ask for something?"),
    "urgency": ScoreQuestion("How urgent?", ("later", "today", "now")),
}


@pytest.fixture(autouse=True)
def _quick_health_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "_HEALTH_POLL_S", 0.01)


@pytest.fixture
def fake_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "llama-server"
    path.write_text(_FAKE_SERVER.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.jsonl"))
    return path


def _log(fake_server: Path) -> list[dict]:
    lines = (fake_server.parent / "log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


async def test_it_starts_the_server_once_and_asks_its_system_one_api(fake_server: Path) -> None:
    classifier = LlamaCppClassifier(
        LlamaCppConfig(model="LiquidAI/d1-2B-GGUF:Q8_0", binary=str(fake_server))
    )
    try:
        first = await classifier.classify({"last": "help now!"}, QUESTIONS)
        await classifier.classify({"last": "thanks"}, QUESTIONS)
    finally:
        await classifier.close()

    log = _log(fake_server)
    starts = [entry["start"] for entry in log if "start" in entry]
    assert len(starts) == 1
    assert starts[0][starts[0].index("-hf") + 1] == "LiquidAI/d1-2B-GGUF:Q8_0"
    asked = [entry for entry in log if "path" in entry]
    assert [entry["path"] for entry in asked] == ["/v1/systemone", "/v1/systemone"]
    assert asked[0]["request"]["state"] == {"last": "help now!"}
    assert first.yes("asks") == pytest.approx(0.91)
    # The expected level, 1.5, not the likeliest level the server sent.
    assert first.score("urgency") == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("failure", "message"),
    [("exits", "unknown model architecture"), ("not-executable", "Permission denied")],
)
async def test_a_server_that_does_not_start_is_a_classifier_error(
    fake_server: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    if failure == "exits":
        monkeypatch.setenv("FAKE_FAIL", "1")
    else:
        fake_server.chmod(0o644)
    classifier = LlamaCppClassifier(LlamaCppConfig(model="x/y-GGUF:Q8_0", binary=str(fake_server)))
    try:
        with pytest.raises(ClassifierError, match=message):
            await classifier.classify("x", QUESTIONS)
    finally:
        await classifier.close()


def test_without_httpx_it_names_both_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's health check needs httpx, which the typesafe extra lacks:
    say so before anything is downloaded or started."""
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"roomkit\[typesafe,llamacpp\]"):
        LlamaCppClassifier(LlamaCppConfig(model="x/y-GGUF:Q8_0"))
