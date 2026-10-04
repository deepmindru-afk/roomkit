"""What an agent says when the work it was to delegate could not be done."""

from __future__ import annotations

#: Said when delegated work could not be completed: spoken by a realtime
#: session's reasoning delegation (RFC §12.4.1), the supervisor's answer to a
#: message its task-formulation pass was cut short on, and what a supervisor
#: is told when its background workers failed (RFC §19.7.3).
FALLBACK_FAILED = "The delegated work could not be completed."
