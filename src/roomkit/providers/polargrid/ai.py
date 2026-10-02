"""PolarGrid AI provider — generates responses via PolarGrid chat completions.

PolarGrid serves OpenAI-shaped chat completions from Canadian-hosted
edges (Toronto / Vancouver / Montreal). As of polargrid-sdk 0.8.4 the
chat-completions endpoint supports tool / function calling: ``context.tools``
are forwarded, and tool calls come back both non-streaming
(``message.tool_calls``) and streaming (fragmented ``delta.tool_calls``,
OpenAI-style). Tool arguments cross the wire as a JSON string and are
parsed back into a dict for RoomKit.

``tool_choice`` is left unset so the backend defaults to ``auto`` — note
that forcing a specific tool is *steered*, not hard-guaranteed, on
PolarGrid's backend.

polargrid-sdk 0.8.5+ exposes an ``enable_thinking`` request flag
(``PolarGridConfig.thinking``). When on, the qwen models surface their
reasoning inline as ``<think>...</think>`` tags in the message content
(the same convention vLLM/Ollama reasoning models use). We parse those
tags out and surface them as ``AIResponse.thinking`` (non-streaming) and
``StreamThinkingDelta`` (streaming), leaving the answer text clean —
reusing the OpenAI provider's tag parser. Thinking responses are larger
and slower (the reasoning counts toward latency and ``max_tokens``).

polargrid-sdk 0.9.0 added multimodal chat: ``Message.content`` accepts a
list of OpenAI-shaped parts (``{"type":"image_url","image_url":{"url":...}}``),
so images now cross the wire instead of being flattened to text.
``supports_vision`` is model-driven (from the curated catalog); an
``AIImagePart`` in a user turn renders as an ``image_url`` part, and an image
tool result keeps the tool message text-only and rides on a synthetic ``user``
message (tool/function-response messages reject images, same as OpenAI). The
image ``url`` may be a remote URL or a ``data:`` URI. Whether the model
actually *sees* the image is the deployed model's capability, not the SDK's.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIProvider,
    AIResponse,
    AITool,
    AIToolCall,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCallDelta,
)
from roomkit.providers.ai.chat_request import ChatDialect, chat_messages
from roomkit.providers.ai.openai_dialect import (
    ThinkTagParser,
    ToolCallSlots,
    extract_think_tags,
    json_schema_format,
)
from roomkit.providers.ai.reasoning import thinking_switch
from roomkit.providers.ai.response_schema import (
    check_schema_answer,
    checked_stream,
    schema_for_generate,
)
from roomkit.providers.ai.tool_calls import (
    CallIds,
    call_garbled,
    tool_arguments,
    unreadable_arguments,
)
from roomkit.providers.ai.tool_declaration import chat_tool_declarations
from roomkit.providers.polargrid import sdk_patch
from roomkit.providers.polargrid.config import PolarGridConfig
from roomkit.providers.polargrid.models import (
    MODELS,
    MODELS_BY_ID,
    REGIONS,
    PolarGridRegion,
)
from roomkit.providers.utils import _aclose_stream, http_timeout

logger = logging.getLogger("roomkit.providers.polargrid")


POLARGRID_CHAT = ChatDialect(drops_thinking=True, names_tool_results=True, flattens_text=True)
"""PolarGrid's rendering: text sent flat (blocks only for an image), no empty
message, each tool message naming its tool, and no earlier reasoning: Qwen
regenerates its own each turn and echoes any wrapper it is fed back (Qwen's
multi-turn guidance is to strip ``<think>`` from history)."""


def _content_filtered(done: StreamDone) -> str | None:
    return "content_filter" if done.finish_reason == "content_filter" else None


class PolarGridAIProvider(AIProvider):
    """AI provider using PolarGrid's chat completions API."""

    def __init__(self, config: PolarGridConfig) -> None:
        try:
            import polargrid as _pg
        except ImportError as exc:
            raise ImportError(
                "polargrid-sdk is required for PolarGridAIProvider. "
                "Install it with: pip install roomkit[polargrid]"
            ) from exc
        self._config = config
        self._sdk = _pg
        # Bind exception classes once so we can catch them without
        # re-importing in hot paths and the test suite can swap the
        # module out via sys.modules patching.
        self._auth_error = _pg.AuthenticationError
        self._validation_error = _pg.ValidationError
        self._rate_limit_error = _pg.RateLimitError
        self._network_error = _pg.NetworkError
        self._timeout_error = _pg.TimeoutError
        self._not_found_error = _pg.NotFoundError
        self._server_error = _pg.ServerError
        self._client: Any | None = None

    @property
    def _provider_name(self) -> str:
        return "polargrid"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        # Model-driven: polargrid-sdk 0.9.0 lets the chat endpoint carry
        # images, but seeing them is the deployed model's capability. Read it
        # from the curated catalog; an unknown model is treated as text-only.
        info = MODELS_BY_ID.get(self._config.model)
        return bool(info and info.supports_vision)

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        # Emits StreamEvent objects — text deltas, tool calls, and done.
        return True

    @property
    def supports_response_schema(self) -> bool:
        """Schema-locked output, through a ``json_schema`` response format
        (vLLM guided decoding on PolarGrid's side)."""
        return True

    # -- Model discovery ----------------------------------------------------

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline snapshot of the chat models on PolarGrid's public edges.

        A customer-pilot model (``qwen-3.6-35b-a3b``) is recognised but not
        advertised: see :data:`~roomkit.providers.polargrid.models.PILOT_MODELS`.
        See :meth:`list_models` for the live, edge-specific set (which also
        includes the STT/TTS models).
        """
        return list(MODELS)

    @classmethod
    def _curated_index(cls) -> dict[str, ModelInfo]:
        # Public and pilot models alike: a pilot edge lists qwen-3.6-35b-a3b,
        # and its display name and vision flag should backfill there too.
        return dict(MODELS_BY_ID)

    async def list_models(self) -> list[ModelInfo]:
        """Models loaded on the connected edge, via the SDK's ``list_models``.

        Returns whatever the edge reports (chat + STT/TTS), so the result is
        region-specific — ``dfw-02`` carries no STT and no ``kokoro-82m``, a
        customer-pilot edge lists ``qwen-3.6-35b-a3b``.
        Curated metadata backfills display names / vision where the endpoint
        leaves them blank.
        """
        client = await self._ensure_client()
        try:
            response = await client.list_models()
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        data = getattr(response, "data", None) or []
        live = [self._parse_model(m) for m in data]
        return self._merge_curated(live)

    @staticmethod
    def _parse_model(model: Any) -> ModelInfo:
        """Map one SDK ``ModelInfo`` to a roomkit :class:`ModelInfo`."""
        pg_type = getattr(model, "pg_model_type", None)
        return ModelInfo(
            id=str(getattr(model, "id", "")),
            capabilities=[pg_type] if pg_type else [],
        )

    @classmethod
    def available_regions(cls) -> list[PolarGridRegion]:
        """Curated, offline catalog of PolarGrid edges (id, name, location).

        PolarGrid serves no live full-region list over the edge API, so this
        is the authoritative static set. Canadian edges have ``location``
        starting with ``"Canada"`` (data residency / Law 25).
        """
        return list(REGIONS)

    async def connected_region(self) -> PolarGridRegion:
        """The PolarGrid edge this provider is routed to (id, name, location).

        Reports the *connected* edge only — PolarGrid serves no live list
        of all regions over the edge API (``/v1/status`` 404s on edges).
        Most useful under auto-routing (``region=None``), where the edge is
        picked at connect time, to confirm which edge handles the data
        (residency / Law 25). ``location`` is backfilled from
        :meth:`available_regions`.
        """
        client = await self._ensure_client()
        region_id = client.get_region_id()
        catalog = {r.id: r for r in REGIONS}
        match = catalog.get(region_id)
        return PolarGridRegion(
            id=region_id,
            name=client.get_region_name() or (match.name if match else None),
            location=match.location if match else None,
        )

    # -- Client lifecycle ---------------------------------------------------

    async def _ensure_client(self) -> Any:
        """Lazily create the PolarGrid async client.

        Done on first call instead of in ``__init__`` because the
        auto-routing variant (``region=None``) is itself async — it
        pings edges to pick the fastest. Pinning a region uses the
        synchronous constructor.
        """
        if self._client is not None:
            return self._client

        kwargs: dict[str, Any] = {
            "api_key": self._config.api_key.get_secret_value(),
            # The SDK annotates ``timeout`` as a float but only ever forwards it
            # to ``httpx.AsyncClient(timeout=...)`` (polargrid-sdk 0.10.0), so
            # the connect/read split reaches httpx intact.
            "timeout": http_timeout(self._config),
            "max_retries": self._config.max_retries,
        }
        if self._config.debug:
            kwargs["debug"] = True

        if self._config.region:
            # Region pinned — synchronous constructor is fine.
            self._client = self._sdk.PolarGrid(region=self._config.region, **kwargs)
        else:
            # Auto-routing — the autorouter picks the nearest edge that
            # already serves the configured model (``routing_model``,
            # polargrid-sdk 0.10.0), falling back to the default edge when
            # none does. Edges diverge on which qwen they carry during a
            # fleet rollout, so model-blind routing could land on an edge
            # that answers model-not-found to every request.
            self._client = await self._sdk.PolarGrid.create(
                routing_model=self._config.model, **kwargs
            )
        return self._client

    # -- Message + tool conversion ------------------------------------------

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        """The conversation as PolarGrid reads it (``chat_request``)."""
        return chat_messages(messages, system_prompt, POLARGRID_CHAT, provider=self._provider_name)

    def _build_tools(self, tools: list[AITool]) -> list[dict[str, Any]] | None:
        """Convert RoomKit tools to PolarGrid's OpenAI-shaped tool list."""
        if not tools:
            return None
        return chat_tool_declarations(tools)

    def _build_request(self, context: AIContext, *, stream: bool) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._build_messages(context.messages, context.system_prompt),
            "stream": stream,
        }
        tools = self._build_tools(context.tools)
        if tools:
            # No tool_choice in AIContext — leave it unset so PolarGrid
            # defaults to "auto". Forcing a tool is steered, not hard
            # guaranteed, on their backend anyway.
            req["tools"] = tools
        thinking = thinking_switch(context, self._config.thinking)
        if thinking is not None:
            # polargrid-sdk 0.8.5+ exposes the enable_thinking flag; qwen
            # then emits its reasoning inline as <think>...</think>, which
            # the streaming/non-streaming paths split out as thinking. The
            # turn's switch outranks the configured one (RFC §6.7).
            req["enable_thinking"] = thinking
        max_tokens = context.max_tokens or self._config.max_tokens
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if context.temperature is not None:
            req["temperature"] = context.temperature
        if self._config.top_p is not None:
            req["top_p"] = self._config.top_p
        if context.response_schema is not None:
            req["response_format"] = json_schema_format(context.response_schema)
        if logger.isEnabledFor(logging.DEBUG):
            # Full outgoing payload — no API key (that lives on the client),
            # so the enable_thinking flag, tools, and messages are visible.
            # Handy to share with PolarGrid when debugging behavior.
            logger.debug("PolarGrid request: %s", json.dumps(req, ensure_ascii=False))
        return req

    # -- Error mapping ------------------------------------------------------

    def _retryable_for(self, exc: BaseException) -> bool:
        """Map an SDK exception to its retryable flag via dispatch table.

        Unknown errors default to retryable so RoomKit's RetryPolicy
        decides whether to back off or surface immediately.
        """
        retry_map: tuple[tuple[type[BaseException], bool], ...] = (
            (self._auth_error, False),
            (self._validation_error, False),
            (self._not_found_error, False),
            (self._rate_limit_error, True),
            (self._network_error, True),
            (self._timeout_error, True),
            (self._server_error, True),
        )
        for exc_type, retryable in retry_map:
            if isinstance(exc, exc_type):
                return retryable
        return True

    def _wrap_error(self, exc: BaseException) -> ProviderError:
        return ProviderError(
            str(exc),
            retryable=self._retryable_for(exc),
            provider=self._provider_name,
            status_code=getattr(exc, "status_code", None),
        )

    # -- Non-streaming ------------------------------------------------------

    async def generate(self, context: AIContext) -> AIResponse:
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider=self._provider_name,
        )
        client = await self._ensure_client()
        request = self._build_request(context, stream=False)

        t0 = time.monotonic()
        try:
            response = await client.chat_completion(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        self._record_ttfb(t0)

        choices = getattr(response, "choices", None) or []
        if not choices:
            self._check_schema_answer(context, "", None)
            return AIResponse(content="")
        choice = choices[0]
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", "") or ""
        # qwen surfaces reasoning inline as <think>...</think>; split it out
        # so the answer text is clean and the reasoning rides on .thinking.
        thinking, content = extract_think_tags(raw_content)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = self._extract_usage(response)
        model = getattr(response, "model", self._config.model)
        tool_calls = self._extract_tool_calls(message, finish_reason)
        if not tool_calls:
            self._check_schema_answer(context, content, finish_reason)

        return AIResponse(
            content=content,
            thinking=thinking,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"model": model},
            tool_calls=tool_calls,
        )

    def _check_schema_answer(
        self, context: AIContext, content: str, finish_reason: str | None
    ) -> None:
        """Refuse a constrained answer that did not deliver its JSON document."""
        if context.response_schema is None:
            return
        check_schema_answer(
            content,
            schema=context.response_schema,
            provider=self._provider_name,
            refusal="content_filter" if finish_reason == "content_filter" else None,
            truncated=finish_reason == "length",
        )

    # -- Streaming ----------------------------------------------------------

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield thinking + text deltas, then any tool calls, then done.

        Content deltas pass through a ``<think>`` tag parser so qwen's
        inline reasoning is emitted as :class:`StreamThinkingDelta` and
        the rest as :class:`StreamTextDelta`. Tool calls arrive
        OpenAI-style: fragmented across chunks as ``delta.tool_calls``
        with a stable ``index``, the ``id`` on the first fragment, and
        ``arguments`` concatenated from each fragment's ``function``
        dict. We accumulate by index and emit one :class:`StreamToolCall`
        per call after the text, so the consumer sees
        thinking-then-text-then-tools in natural order. A response schema is
        checked before the done event (RFC §6.7).
        """
        stream = checked_stream(
            self._stream_events(context),
            context,
            provider=self._provider_name,
            refusal=_content_filtered,
            truncated=lambda done: done.finish_reason == "length",
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _stream_events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The streamed call itself."""
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider=self._provider_name,
        )
        client = await self._ensure_client()
        request = self._build_request(context, stream=True)

        t0 = time.monotonic()
        first_token = True
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        tool_call_slots = ToolCallSlots()
        parser = ThinkTagParser()

        stream = sdk_patch.chat_completion_stream(self._sdk, client, request)
        try:
            async for chunk in stream:
                # The usage comes last, as a chunk with no choices.
                chunk_usage = self._extract_usage(chunk)
                if chunk_usage:
                    usage = chunk_usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                tool_deltas = getattr(delta, "tool_calls", None)
                if tool_deltas:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    for composed in self._accumulate_tool_deltas(tool_call_slots, tool_deltas):
                        yield composed

                text = getattr(delta, "content", None)
                if text:
                    for kind, segment in parser.feed(text):
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        if kind == "thinking":
                            yield StreamThinkingDelta(thinking=segment)
                        else:
                            yield StreamTextDelta(text=segment)

            # Flush any buffered text held back for a partial tag.
            for kind, segment in parser.flush():
                if kind == "thinking":
                    yield StreamThinkingDelta(thinking=segment)
                else:
                    yield StreamTextDelta(text=segment)

            for event in tool_call_slots.calls(finish_reason):
                yield event

            yield StreamDone(finish_reason=finish_reason, usage=usage)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        finally:
            # A turn closed early releases the HTTP stream now, not at GC.
            await _aclose_stream(stream)

    # -- Helpers ------------------------------------------------------------

    def _extract_tool_calls(self, message: Any, finish_reason: str | None) -> list[AIToolCall]:
        """Read non-streaming ``message.tool_calls`` into AIToolCalls (RFC §6.4)."""
        raw_calls = getattr(message, "tool_calls", None) or []
        ids = CallIds()
        result: list[AIToolCall] = []
        for tc in raw_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = str(getattr(func, "name", "") or "")
            raw = getattr(func, "arguments", "")
            result.append(
                AIToolCall(
                    id=ids(getattr(tc, "id", None), name),
                    name=name,
                    arguments=tool_arguments(raw),
                    partial=unreadable_arguments(raw),
                    garbled=call_garbled(raw, finish_reason),
                )
            )
        return result

    @staticmethod
    def _accumulate_tool_deltas(
        slots: ToolCallSlots, deltas: list[Any]
    ) -> list[StreamToolCallDelta]:
        """Fold streamed ``ToolCallDelta`` fragments (plain dicts) into *slots*.

        Returns the composition events the caller yields: the call's name on
        the fragment that first carries it, then one per argument fragment, so
        the minutes a large argument takes to compose are visible while they
        happen rather than only in the completed call.
        """
        composed: list[StreamToolCallDelta] = []
        for d in deltas:
            func = getattr(d, "function", None)
            if not isinstance(func, dict):
                func = {}
            event = slots.fold(
                getattr(d, "index", None),
                getattr(d, "id", None),
                func.get("name"),
                func.get("arguments") or "",
            )
            if event is not None:
                composed.append(event)
        return composed

    @staticmethod
    def _extract_usage(obj: Any) -> dict[str, int]:
        usage_obj = getattr(obj, "usage", None)
        if not usage_obj:
            return {}
        prompt = getattr(usage_obj, "prompt_tokens", None)
        completion = getattr(usage_obj, "completion_tokens", None)
        result: dict[str, int] = {}
        if prompt is not None:
            result["input_tokens"] = int(prompt)
        if completion is not None:
            result["output_tokens"] = int(completion)
        return result

    async def close(self) -> None:
        """Release the underlying client if it exposes a close hook."""
        if self._client is None:
            return
        closer = getattr(self._client, "close", None) or getattr(self._client, "aclose", None)
        if closer is None:
            return
        result = closer()
        # SDK may expose sync or async close — handle both.
        if hasattr(result, "__await__"):
            await result
