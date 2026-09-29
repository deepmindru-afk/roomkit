"""Shared primitives for the supervisor strategy package.

``WorkerStrategy`` and the per-task / revision defaults are imported by every
submodule and re-exported from the package root. The logger uses the
``roomkit.orchestration.strategies.supervisor`` package-path name so logging
configured against that name matches.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from roomkit.orchestration.status_bus import post_agent_lifecycle

logger = logging.getLogger("roomkit.orchestration.strategies.supervisor")

# Local alias — the Supervisor strategy specifically tracks worker
# lifecycle, but the underlying helper is shared with Pipeline / Swarm /
# Loop so all strategies emit events under the same conventions.
_post_worker_status = post_agent_lifecycle


#: Name of the single tool injected on the supervisor in strategy-tool mode.
#: The supervisor calls it (in the parent room) to dispatch its whole team. It is
#: declared in the rooms the supervisor was installed in, so the ``::task-`` rooms
#: where the supervised flow runs it for dispatch/review never offer it.
_STRATEGY_TOOL_NAME = "delegate_workers"


def _is_subtask_room(room_id: str) -> bool:
    """A delegated child room (``::task-`` segment). The supervisor must run
    normally there — re-dispatching ``delegate_workers`` would recurse the whole
    pipeline (delegate_workers within delegate_workers)."""
    return "::task-" in room_id


class WorkerStrategy(StrEnum):
    """How workers are executed when delegated to."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# Per-worker delegation budget. Each delegated task is bounded individually,
# so the supervisor's chain is governed by a per-task limit rather than one
# global timeout (a single global can never be coherent: it must cover the
# *sum* of every worker, which scales with team size). A worker that exceeds
# this fails on its own budget and the chain keeps going.
_DEFAULT_TASK_TIMEOUT_SECONDS = 120.0

# Max supervisor→worker round-trips per step in supervised mode: the supervisor
# validates each worker's output and, if it's not acceptable, sends it back with
# feedback for a rework. Bounded so a worker that can't satisfy the supervisor
# doesn't loop forever — on exhaustion the step is delivered flagged unvalidated.
_DEFAULT_MAX_REVISIONS = 3
