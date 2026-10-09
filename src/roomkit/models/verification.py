"""Identity verification models (RFC §11.7).

Identification says whose an address is; verification says the person writing
has proven it. RoomKit keeps the state and its lifecycle; how a person proves
who they are (a page behind a link, a code, a login) is the integrator's.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from roomkit.models.enums import VerificationRequestStatus, VerificationStatus


class Verification(BaseModel):
    """An identity's verification, valid on every channel its addresses reach (§11.7.1)."""

    identity_id: str
    organization_id: str | None = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    method: str | None = None
    """The integrator's name for how it was proven ("link", "code", "sso")."""
    verified_via: str | None = None
    """The channel the completing request was opened on."""
    verified_at: datetime | None = None
    expires_at: datetime | None = None
    ended_reason: str | None = None
    """Why it ended: "expired", "revoked", "failed"."""
    request_id: str | None = None
    """The request open for the identity, if any."""
    completed_request_id: str | None = None
    """The request that completed the verification: its end is announced in
    that request's room, on its channel."""


class VerificationRequest(BaseModel):
    """A request asking an identified sender to prove who they are (§11.7.2).

    The secret is never stored: only its hash, salted with the request's id.
    """

    id: str
    identity_id: str
    organization_id: str | None = None
    room_id: str
    channel_id: str
    method: str
    secret_hash: str
    window_start: datetime
    """When the message that opened it was received: the start of the window
    whose held messages are released once verified (§11.7.6)."""
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int
    status: VerificationRequestStatus = VerificationRequestStatus.OPEN


class VerificationPolicy(BaseModel):
    """How long a verification and its requests last, and how many attempts (§11.7)."""

    ttl: float = Field(default=900.0, gt=0)
    """Seconds a completed verification stays in force."""
    request_ttl: float = Field(default=600.0, gt=0)
    """Seconds a request may be completed in."""
    code_request_ttl: float = Field(default=300.0, gt=0)
    """Seconds a code request may be completed in: a code is short (§11.7.7)."""
    max_attempts: int = Field(default=3, ge=1)
    """Wrong secrets a request accepts before it fails."""
    code_length: int = Field(default=6, ge=6, le=12)
    """Digits of a code sent for the person to reply with."""
    release_on_verify: bool = True
    """Whether the messages held during a request are released to the room's
    agents when it completes (RFC §11.7.6)."""
    release_max_messages: int = Field(default=20, ge=1)
    """At most this many held messages are released, the latest ones."""
    release_max_age: float = Field(default=3600.0, gt=0)
    """Held messages older than this many seconds are not released."""
    hold_unverified: bool = True
    """Whether an unverified sender's message is held from the room's agents
    unless an ON_VERIFICATION_REQUIRED hook passes it (RFC §11.7.3); False
    only labels it."""
