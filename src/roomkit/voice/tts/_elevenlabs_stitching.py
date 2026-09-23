"""ElevenLabs request stitching from the TTS conversation context (RFC §12.2.2).

ElevenLabs continues a voice across requests when told the ``request_id`` of
the generations that came before (``previous_request_ids``, at most three, no
older than two hours), or failing that the text that came before
(``previous_text``; ignored when ids are sent). An id is only usable when its
audio was read to the end, so it is kept only then; a turn the user cut off is
never used, whatever its generation did.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from roomkit.voice.tts.context import TTSContext

MAX_REQUEST_IDS = 3
REQUEST_ID_TTL_S = 2 * 3600 - 60  # two hours, with a margin for the round trip
_KEPT_IDS = 10  # more than sent: a generation that never became a turn must not evict one


class RequestIdLedger:
    """The ``request_id`` and voice of each fully read generation, per session."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._ids: dict[str, dict[str, tuple[str, str, float]]] = {}

    def record(self, context_id: str, turn_id: str, request_id: str, voice_id: str) -> None:
        """Keep *request_id* as the id of the turn *turn_id* will become."""
        ids = self._ids.setdefault(context_id, {})
        ids[turn_id] = (request_id, voice_id, self._clock())
        for stale in list(ids)[:-_KEPT_IDS]:
            del ids[stale]

    def forget(self, context_id: str) -> None:
        self._ids.pop(context_id, None)

    def stitching_params(self, context: TTSContext, voice_id: str) -> dict[str, Any]:
        """The stitching arguments for the next generation of *context*.

        Continuity follows the last thing the user heard, in the same voice:
        after a turn cut off by a barge-in nothing is sent and the response
        starts afresh, and a turn spoken in another voice (a ``voice_map``
        with several agents) ends the chain.
        """
        turns = [t for t in context.turns if t.role == "assistant"]
        if not turns or turns[-1].interrupted:
            return {}
        known = self._ids.get(context.context_id, {})
        last = known.get(turns[-1].turn_id)
        if last is not None and last[1] != voice_id:
            return {}
        now = self._clock()
        request_ids: list[str] = []
        for turn in reversed(turns):
            entry = known.get(turn.turn_id)
            if (
                turn.interrupted
                or entry is None
                or entry[1] != voice_id
                or now - entry[2] > REQUEST_ID_TTL_S
            ):
                break
            request_ids.insert(0, entry[0])
            if len(request_ids) == MAX_REQUEST_IDS:
                break
        if request_ids:
            return {"previous_request_ids": request_ids}
        return {"previous_text": turns[-1].text}


def request_id_of(headers: Mapping[str, str]) -> str | None:
    """The ``request-id`` response header, whatever its case."""
    for name, value in headers.items():
        if name.lower() == "request-id":
            return value
    return None
