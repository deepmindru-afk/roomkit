"""OpenTelemetry telemetry provider — bridges to the OTel SDK."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from roomkit.telemetry.base import SpanKind, TelemetryProvider

# opentelemetry is an optional dependency — all imports are deferred
# to method bodies so the module can be imported for inspection even
# when the SDK is not installed.

logger = logging.getLogger("roomkit.telemetry.otel")


class OpenTelemetryProvider(TelemetryProvider):
    """Bridges RoomKit telemetry to OpenTelemetry SDK.

    Requires ``opentelemetry-api`` and ``opentelemetry-sdk``::

        pip install opentelemetry-api opentelemetry-sdk

    Example::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        from roomkit import RoomKit
        from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

        kit = RoomKit(telemetry=OpenTelemetryProvider(tracer_provider=provider))

    Exporting never runs on the event loop. ``force_flush`` exports in the
    thread that calls it, behind the exporter's own retries, and the SDK does
    not honour its timeout: called from the loop, a slow collector froze every
    task on it. :meth:`flush` (the end of a voice session) hands it to a thread
    and returns at once; :meth:`close` waits for it at most
    ``shutdown_flush_timeout`` seconds, then gives up the spans not yet out.
    """

    def __init__(
        self,
        *,
        tracer_provider: Any = None,
        meter_provider: Any = None,
        service_name: str = "roomkit",
        metadata: dict[str, str] | None = None,
        shutdown_flush_timeout: float = 4.0,
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryProvider requires opentelemetry-api and opentelemetry-sdk. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            ) from exc

        self._StatusCode = StatusCode

        self._tracer_provider = tracer_provider
        if tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(service_name)
        else:
            self._tracer = trace.get_tracer(service_name)

        self._meter_provider = meter_provider
        self._meter: Any = None
        if meter_provider is not None:
            self._meter = meter_provider.get_meter(service_name)
        else:
            try:
                from opentelemetry import metrics

                self._meter = metrics.get_meter(service_name)
            except Exception:  # nosec B110
                pass

        self._active_spans: dict[str, Any] = {}
        # Ended spans kept briefly so late children can still reference them
        # as parents (e.g. AFTER_TTS hook fires after VOICE_SESSION ends).
        self._ended_spans: dict[str, Any] = {}
        self._metadata: dict[str, str] = metadata or {}
        self._shutdown_flush_timeout = shutdown_flush_timeout
        # Held while a flush thread runs: a flush asked meanwhile is skipped,
        # the running one exports what it asked for and more.
        self._flushing = threading.Lock()

    @property
    def name(self) -> str:
        return "opentelemetry"

    def get_span_context(self, span_id: str) -> Any:
        """Return an OTel Context with the given span set as current.

        Overrides :meth:`TelemetryProvider.get_span_context` to return a
        native OTel ``Context`` for robust parent linking across async
        boundaries.
        """
        from opentelemetry import trace

        otel_span = self._active_spans.get(span_id) or self._ended_spans.get(span_id)
        if otel_span is not None:
            return trace.set_span_in_context(otel_span)
        return None

    def inject_context(self, span_id: str) -> dict[str, str]:
        """*span_id*'s trace context, written by the application's
        propagator: W3C ``traceparent`` and ``tracestate``, and the ambient
        ``baggage``, by default (``OTEL_PROPAGATORS=tracecontext`` keeps the
        baggage in its process)."""
        from opentelemetry import propagate

        context = self.get_span_context(span_id)
        if context is None:
            return {}
        carrier: dict[str, str] = {}
        propagate.inject(carrier, context=context)
        return carrier

    def extract_context(self, carrier: Mapping[str, str]) -> Any:
        """The OpenTelemetry context the application's propagator reads from
        *carrier*; ``None`` when it names no span."""
        from opentelemetry import propagate, trace

        context = propagate.extract(dict(carrier))
        if not trace.get_current_span(context).get_span_context().is_valid:
            return None
        return context

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        room_id: str | None = None,
        session_id: str | None = None,
        channel_id: str | None = None,
    ) -> str:
        ctx = self._parent_context(parent_id, name) if parent_id else None
        otel_span = self._tracer.start_span(
            name=f"roomkit.{name}",
            context=ctx,
            attributes=self._build_attributes(kind, attributes, room_id, session_id, channel_id),
        )

        # Use the span's hex ID as our span_id
        span_ctx = otel_span.get_span_context()
        span_id = format(span_ctx.span_id, "016x")
        self._active_spans[span_id] = otel_span

        return span_id

    def _parent_context(self, parent_id: str, name: str) -> Any:
        """The OpenTelemetry context a span named *name* starts under, as a
        child of *parent_id*: the context the caller propagated when its span
        is that parent (one from another process included), else the parent
        this provider started, else the propagated context, the span then
        staying in its trace under an ancestor."""
        from opentelemetry import trace

        from roomkit.telemetry.context import get_current_telemetry_ctx

        propagated = get_current_telemetry_ctx()
        if propagated is not None and _span_id_in(propagated) == parent_id:
            return propagated
        parent_span = self._active_spans.get(parent_id) or self._ended_spans.get(parent_id)
        if parent_span is not None:
            return trace.set_span_in_context(parent_span)
        if propagated is None:
            logger.warning(
                "Parent span %s not found for %s (active=%d, ended=%d)",
                parent_id,
                name,
                len(self._active_spans),
                len(self._ended_spans),
            )
        return propagated

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        otel_span = self._active_spans.pop(span_id, None)
        if otel_span is None:
            return
        # Keep for late parent lookups (e.g. AFTER_TTS hook after session end)
        self._ended_spans[span_id] = otel_span

        if attributes:
            for k, v in attributes.items():
                self._set_otel_attribute(otel_span, k, v)

        if status == "error":
            otel_span.set_status(self._StatusCode.ERROR, error_message or "")
            if error_message:
                otel_span.record_exception(Exception(error_message))
        else:
            otel_span.set_status(self._StatusCode.OK)

        otel_span.end()

    def set_attribute(self, span_id: str, key: str, value: Any) -> None:
        otel_span = self._active_spans.get(span_id)
        if otel_span is not None:
            self._set_otel_attribute(otel_span, key, value)

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self._meter is None:
            return
        try:
            # Use a histogram for timing/duration metrics, gauge for counts
            if "duration" in name or "ttfb" in name:
                histogram = self._meter.create_histogram(
                    name=name,
                    unit=unit or "ms",
                    description=f"RoomKit metric: {name}",
                )
                histogram.record(value, attributes=attributes)
            else:
                counter = self._meter.create_counter(
                    name=name,
                    unit=unit,
                    description=f"RoomKit metric: {name}",
                )
                counter.add(value, attributes=attributes)
        except Exception:
            logger.debug("Failed to record OTel metric %s", name, exc_info=True)

    def flush(self) -> None:
        """Export the ended spans on a thread of its own, and return at once."""
        if self._tracer_provider is None or not self._flushing.acquire(blocking=False):
            return
        try:
            self._flush_thread(self._flush_and_release).start()
        except RuntimeError:
            # No thread to be had: release, or every later flush is skipped.
            self._flushing.release()
            logger.debug("Could not start the OTel flush thread", exc_info=True)

    def close(self) -> None:
        """End the active spans, then export, waiting at most
        ``shutdown_flush_timeout`` seconds."""
        # End any remaining active spans gracefully (shutdown, not error)
        for span_id in list(self._active_spans):
            self.end_span(span_id)
        self._active_spans.clear()
        # Keep _ended_spans — late async tasks (e.g. AFTER_TTS hook fired
        # during shutdown) may still need parent lookup.  The dict will be
        # garbage-collected with the provider instance.
        if self._tracer_provider is None:
            return
        thread = self._flush_thread(self._force_flush)
        try:
            thread.start()
        except RuntimeError:
            # No thread to be had (interpreter shutting down): flush here.
            self._force_flush()
            return
        thread.join(self._shutdown_flush_timeout)
        if thread.is_alive():
            logger.warning(
                "OTel flush did not finish within %.1fs at shutdown; spans may be lost",
                self._shutdown_flush_timeout,
            )

    @staticmethod
    def _flush_thread(target: Any) -> threading.Thread:
        # Daemon: the interpreter never waits on an export at exit, and
        # close() bounds its own wait.
        return threading.Thread(target=target, name="otel-flush", daemon=True)

    def _flush_and_release(self) -> None:
        try:
            self._force_flush()
        finally:
            self._flushing.release()

    def _force_flush(self) -> None:
        try:
            self._tracer_provider.force_flush()
        except Exception:
            logger.debug("Failed to flush OTel tracer provider", exc_info=True)

    def reset(self) -> None:
        self._active_spans.clear()
        self._ended_spans.clear()

    def _build_attributes(
        self,
        kind: SpanKind,
        attributes: dict[str, Any] | None,
        room_id: str | None,
        session_id: str | None,
        channel_id: str | None,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {"roomkit.span_kind": str(kind)}
        # Global metadata tags (searchable in Jaeger)
        for k, v in self._metadata.items():
            attrs[f"roomkit.{k}"] = v
        if room_id:
            attrs["roomkit.room_id"] = room_id
        if session_id:
            attrs["roomkit.session_id"] = session_id
        if channel_id:
            attrs["roomkit.channel_id"] = channel_id
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    attrs[f"roomkit.{k}"] = v
        return attrs

    @staticmethod
    def _set_otel_attribute(otel_span: Any, key: str, value: Any) -> None:
        """Set an attribute on an OTel span, coercing to allowed types."""
        if isinstance(value, (str, int, float, bool)):
            otel_span.set_attribute(f"roomkit.{key}", value)
        elif value is not None:
            otel_span.set_attribute(f"roomkit.{key}", str(value))


def _span_id_in(context: Any) -> str:
    """The id of the span an OpenTelemetry *context* holds, as RoomKit names
    span ids (16 hex digits)."""
    from opentelemetry import trace

    return format(trace.get_current_span(context).get_span_context().span_id, "016x")
