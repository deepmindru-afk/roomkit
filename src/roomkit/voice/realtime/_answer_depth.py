"""The chain depth of what a speech-to-speech model says (RFC §12.4, §8.3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnswerDepth:
    """What a speech-to-speech model last heard, and the depth its answer carries.

    The model answers what it heard last: the user, at depth 0, or an event
    injected into its session at that event's depth. Its transcription carries
    that depth plus one, so a chain that runs through a realtime model ends at
    ``max_chain_depth`` like any other instead of restarting from 0 on every
    utterance.
    """

    heard: int = 0

    def injected(self, depth: int) -> None:
        """The model was handed an event of this depth to answer."""
        self.heard = depth

    def user_spoke(self) -> None:
        """The model heard the user, whose words open a chain."""
        self.heard = 0

    @property
    def answer(self) -> int:
        """The depth of the model's next answer."""
        return self.heard + 1
