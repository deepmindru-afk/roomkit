"""The OTel provider never exports on the event loop (RMK-408).

``force_flush`` exports in the calling thread, behind the exporter's own
retries: called from the loop, a slow collector froze every task on it.
``flush()`` hands it to a thread and returns at once; ``close()`` waits for it
at most ``shutdown_flush_timeout`` seconds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from roomkit.telemetry.base import SpanKind  # noqa: E402
from roomkit.telemetry.opentelemetry import OpenTelemetryProvider  # noqa: E402


class _SlowCollector(SpanExporter):
    """A collector that takes *delay* seconds to accept a batch."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.exported: list[str] = []
        self.done = threading.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        time.sleep(self.delay)
        self.exported.extend(span.name for span in spans)
        self.done.set()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _provider(collector: _SlowCollector, **kwargs: float) -> OpenTelemetryProvider:
    tracer_provider = TracerProvider()
    # A batch that never leaves on its own schedule: only a flush sends it.
    tracer_provider.add_span_processor(BatchSpanProcessor(collector, schedule_delay_millis=60_000))
    return OpenTelemetryProvider(tracer_provider=tracer_provider, **kwargs)


def test_a_flush_returns_at_once_and_the_spans_still_leave() -> None:
    collector = _SlowCollector(delay=1.0)
    provider = _provider(collector)
    provider.end_span(provider.start_span(SpanKind.VOICE_SESSION, "session"))

    started = time.monotonic()
    provider.flush()
    provider.flush()  # one already running: skipped, not queued behind it
    returned_after = time.monotonic() - started

    assert returned_after < 0.2
    assert collector.done.wait(timeout=5.0)
    assert collector.exported == ["roomkit.session"]


def test_close_waits_for_the_export_within_its_bound(caplog: pytest.LogCaptureFixture) -> None:
    collector = _SlowCollector(delay=2.0)
    provider = _provider(collector, shutdown_flush_timeout=0.2)
    provider.start_span(SpanKind.VOICE_SESSION, "still open")

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="roomkit.telemetry.otel"):
        provider.close()

    assert time.monotonic() - started < 1.0
    assert "did not finish within 0.2s" in caplog.text


def test_close_exports_the_spans_it_ends_when_the_collector_answers() -> None:
    collector = _SlowCollector(delay=0.0)
    provider = _provider(collector)
    provider.start_span(SpanKind.VOICE_SESSION, "still open")

    provider.close()

    assert collector.exported == ["roomkit.still open"]
