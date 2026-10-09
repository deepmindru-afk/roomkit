"""VerificationMixin — identity verification requests and their outcome (RFC §11.7).

RoomKit keeps the state and the lifecycle; the integrator keeps the method (how
a person proves who they are) and calls :meth:`complete_verification` once its
own check passed. A request's secret is returned once and stored only as a
salted hash.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.exceptions import VerificationRefusedError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    EventType,
    HookTrigger,
    IdentificationStatus,
    VerificationRequestStatus,
    VerificationStatus,
    Visibility,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import VERIFICATION_DECISION_KEY
from roomkit.models.verification import Verification, VerificationPolicy, VerificationRequest

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.identity import Identity
    from roomkit.store.base import ConversationStore

METHOD_LINK = "link"
"""A secret carried in a link to the integrator's page (high entropy)."""
METHOD_CODE = "code"
"""A short code the person replies with in the conversation (RFC §11.7.4)."""

_LINK_SECRET_BYTES = 32  # 256 bits; RFC §11.7.7 asks for at least 128
HELD_KEY = "roomkit:verification_held"
"""Event metadata marking a message held for verification (RFC §11.7.3); its
value records the visibility the message had, which its release restores."""
# scrypt, not a fast hash: a 6-digit code read from a leaked store would
# otherwise be found in under a second. At n=2**15 one hash costs ~55 ms, so
# the million codes take hours, against a code's minutes of life.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


@runtime_checkable
class VerificationHost(Protocol):
    """Contract: capabilities a host class must provide for VerificationMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend (stores verifications).
        _lock_manager: Serialises the transitions of one identity.
        _verification_policy: Lifetimes and attempts.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _verification_policy: VerificationPolicy
    _verification_timers: dict[str, asyncio.Task[None]]


class VerificationMixin(HelpersMixin):
    """Identity verification: requests, completion, revocation, state.

    Host contract: :class:`VerificationHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _verification_policy: VerificationPolicy
    _verification_timers: dict[str, asyncio.Task[None]]
    _hook_engine: HookEngine

    async def request_verification(
        self,
        identity_id: str,
        *,
        room_id: str,
        channel_id: str,
        method: str = METHOD_LINK,
        window_start: datetime | None = None,
        organization_id: str | None = None,
    ) -> tuple[VerificationRequest, str]:
        """Open a verification request for an identity; return it and its secret.

        The secret is returned once and never stored: put it in the link to
        your page, or send it as a code (``method="code"``: digits the person
        replies with). Opening a request cancels the identity's earlier open
        one. An identity that is not verified becomes PENDING; one already
        verified stays so until this request completes or its verification
        ends. *window_start* is when the message that called for it was
        received (now by default): the held messages released on completion
        are those from then on (RFC §11.7.6).
        """
        policy = self._verification_policy
        now = datetime.now(UTC)
        code = method == METHOD_CODE
        secret = _code(policy.code_length) if code else secrets.token_urlsafe(_LINK_SECRET_BYTES)
        ttl = policy.code_request_ttl if code else policy.request_ttl
        request_id = uuid4().hex
        request = VerificationRequest(
            id=request_id,
            identity_id=identity_id,
            organization_id=organization_id,
            room_id=room_id,
            channel_id=channel_id,
            method=method,
            secret_hash=await _secret_hash(request_id, secret),
            window_start=window_start or now,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            max_attempts=policy.max_attempts,
        )
        async with self._lock_manager.locked(_lock_key(identity_id, organization_id)):
            current = await self._stored_verification(identity_id, organization_id)
            cancelled = await self._cancel_open_request(current)
            await self._store.create_verification_request(request)
            in_force = _in_force(current, now)
            await self._store.put_verification(
                current.model_copy(
                    update={
                        "status": VerificationStatus.VERIFIED
                        if in_force
                        else VerificationStatus.PENDING,
                        "request_id": request.id,
                    }
                )
            )
        # Announced once the lock is released: a handler may call back in.
        if cancelled is not None:
            await self._emit_verification_event(
                "verification_ended", cancelled, reason="cancelled"
            )
        await self._emit_verification_event("verification_requested", request)
        return request, secret

    async def complete_verification(
        self,
        request_id: str,
        secret: str,
        *,
        method: str | None = None,
        organization_id: str | None = None,
    ) -> Verification:
        """Complete a request once your own check passed; return the verification.

        Raises :class:`VerificationRefusedError` when the request is unknown in
        *organization_id*, no longer open, expired, or the secret does not
        match (a mismatch counts as an attempt; the last one fails it).
        *method* records how the person proved it, the request's by default.
        """
        request = await self._store.get_verification_request(request_id)
        if request is None or (request.organization_id or None) != (organization_id or None):
            raise VerificationRefusedError("not_found")
        key = _lock_key(request.identity_id, request.organization_id)
        async with self._lock_manager.locked(key):
            outcome, request, verification = await self._attempt(request_id, secret, method)
        if outcome == "completed":
            self._arm_expiry(verification)
            await self._emit_verification_event("verification_completed", request)
            await self._verification_hook(
                HookTrigger.ON_VERIFICATION_COMPLETED,
                request,
                {"method": verification.method, "expires_at": _iso(verification.expires_at)},
            )
            return verification
        if outcome in ("expired", "failed"):
            if outcome == "failed":
                # It may have ended a verification in force: no timer outlives it.
                self._disarm_expiry(request.identity_id, request.organization_id)
            await self._emit_verification_event("verification_ended", request, reason=outcome)
            await self._verification_hook(
                HookTrigger.ON_VERIFICATION_ENDED, request, {"reason": outcome}
            )
        raise VerificationRefusedError(outcome)

    async def revoke_verification(
        self,
        identity_id: str,
        *,
        reason: str = "revoked",
        organization_id: str | None = None,
    ) -> Verification | None:
        """End an identity's verification before its time; None if none was in force.

        Its open request is cancelled in every case, verified or still pending:
        a link sent before the revocation can never verify the person after it
        (RFC §11.7.7).
        """
        now = datetime.now(UTC)
        self._disarm_expiry(identity_id, organization_id)
        async with self._lock_manager.locked(_lock_key(identity_id, organization_id)):
            current = await self._stored_verification(identity_id, organization_id)
            completing = await self._completing_request(current)
            cancelled = await self._cancel_open_request(current)
            ended: Verification | None = None
            if _in_force(current, now):
                ended = current.model_copy(
                    update={
                        "status": VerificationStatus.REVOKED,
                        "ended_reason": reason,
                        "request_id": None,
                    }
                )
                await self._store.put_verification(ended)
            elif cancelled is not None:
                # Pending, now with nothing open: back to unverified.
                await self._store.put_verification(
                    current.model_copy(
                        update={"status": VerificationStatus.UNVERIFIED, "request_id": None}
                    )
                )
        if cancelled is not None:
            await self._emit_verification_event(
                "verification_ended", cancelled, reason="cancelled"
            )
        if ended is not None:
            await self._emit_framework_event(
                "verification_ended",
                data={"identity_id": identity_id, "reason": reason},
            )
            if completing is not None:
                await self._verification_hook(
                    HookTrigger.ON_VERIFICATION_ENDED, completing, {"reason": reason}
                )
        return ended

    async def check_verifications(self) -> list[Verification]:
        """End the verifications whose time passed; arm the timers of the others.

        A verification ends when its time runs out (RFC §11.7.5), by a timer
        this process arms when it completes. Call this when the process
        starts (``async with RoomKit(...)`` does): it ends what expired while
        nothing was running and re-arms the rest. Returns the ones it ended.
        """
        ended: list[Verification] = []
        now = datetime.now(UTC)
        for verification in await self._store.list_verifications_in_force():
            if _in_force(verification, now):
                self._arm_expiry(verification)
                continue
            expired = await self._expire(verification)
            if expired is not None:
                ended.append(expired)
        return ended

    async def verification_of(
        self, identity_id: str, *, organization_id: str | None = None
    ) -> Verification:
        """The identity's verification as it stands now: UNVERIFIED when it has none.

        A verification whose time has passed reads EXPIRED, whether or not its
        end has been recorded yet.
        """
        current = await self._stored_verification(identity_id, organization_id)
        if current.status == VerificationStatus.VERIFIED and not _in_force(
            current, datetime.now(UTC)
        ):
            return current.model_copy(
                update={"status": VerificationStatus.EXPIRED, "ended_reason": "expired"}
            )
        return current

    # -- inbound pipeline (RFC §10.1 step 5b) -----------------------------------

    async def _apply_verification(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        resolved_identity: Identity | None,
    ) -> RoomEvent | InboundResult:
        """Label an identified sender's message, consume a code, or hold it.

        Returns the event to go on with, or the result of a message that stops
        here before anything of it is stored (a code reply, a block).
        """
        identity = _sender_identity(event, context, resolved_identity)
        if identity is None:
            return event
        identity_id, organization_id = identity
        state = await self.verification_of(identity_id, organization_id=organization_id)
        if await self._consume_code(event, room_id, state):
            # Its text is never stored nor broadcast, and no agent reads it.
            return InboundResult(blocked=True, reason="verification_code")
        event = event.model_copy(
            update={"source": event.source.model_copy(update={"verification": state.status})}
        )
        if state.status == VerificationStatus.VERIFIED:
            return event
        decided = await self._hook_engine.run_sync_hooks(
            room_id, HookTrigger.ON_VERIFICATION_REQUIRED, event, context
        )
        if not decided.allowed:
            return InboundResult(blocked=True, reason=decided.reason or "verification_required")
        event = decided.event if isinstance(decided.event, RoomEvent) else event
        passed = decided.metadata.get(VERIFICATION_DECISION_KEY) == "pass"
        if passed or not self._verification_policy.hold_unverified:
            return event
        return _held(event)

    async def _consume_code(self, event: RoomEvent, room_id: str, state: Verification) -> bool:
        """Whether *event* is a reply to the sender's open code request here, attempted."""
        if state.request_id is None or not isinstance(event.content, TextContent):
            return False
        text = event.content.body.strip()
        if not text.isdigit() or len(text) != self._verification_policy.code_length:
            return False
        request = await self._store.get_verification_request(state.request_id)
        if (
            request is None
            or request.method != METHOD_CODE
            or request.status != VerificationRequestStatus.OPEN
            or request.room_id != room_id
        ):
            return False
        # A wrong code counts as an attempt and is consumed all the same.
        with contextlib.suppress(VerificationRefusedError):
            await self.complete_verification(
                request.id, text, organization_id=request.organization_id
            )
        return True

    # -- internal ------------------------------------------------------------

    async def _resume_verifications(self) -> None:
        """Resume the verifications in force as the kit starts.

        A store that does not keep verifications (RFC §11.7 is Level 2) has
        none to resume: the kit starts all the same.
        """
        try:
            await self.check_verifications()
        except NotImplementedError:
            return

    def _arm_expiry(self, verification: Verification) -> None:
        """Arm (or re-arm) the timer that ends *verification* when it expires."""
        if verification.expires_at is None:
            return
        key = _lock_key(verification.identity_id, verification.organization_id)
        self._disarm_expiry(verification.identity_id, verification.organization_id)
        task = asyncio.get_running_loop().create_task(self._expire_when_due(verification))
        self._verification_timers[key] = task
        task.add_done_callback(lambda done: _forget_timer(self._verification_timers, key, done))

    def _disarm_expiry(self, identity_id: str, organization_id: str | None) -> None:
        task = self._verification_timers.pop(_lock_key(identity_id, organization_id), None)
        if task is not None:
            task.cancel()

    def _cancel_verification_timers(self) -> None:
        """Stop every expiry timer (the kit is closing)."""
        for task in self._verification_timers.values():
            task.cancel()
        self._verification_timers.clear()

    async def _expire_when_due(self, verification: Verification) -> None:
        if verification.expires_at is None:
            return
        delay = (verification.expires_at - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(0.0, delay))
        await self._expire(verification)

    async def _expire(self, verification: Verification) -> Verification | None:
        """End *verification* if it is still the one in force and its time passed."""
        key = _lock_key(verification.identity_id, verification.organization_id)
        async with self._lock_manager.locked(key):
            current = await self._stored_verification(
                verification.identity_id, verification.organization_id
            )
            still_it = (
                current.status == VerificationStatus.VERIFIED
                and current.expires_at == verification.expires_at
            )
            if not still_it or _in_force(current, datetime.now(UTC)):
                return None
            completing = await self._completing_request(current)
            ended = current.model_copy(
                update={"status": VerificationStatus.EXPIRED, "ended_reason": "expired"}
            )
            await self._store.put_verification(ended)
        await self._emit_framework_event(
            "verification_ended",
            data={"identity_id": ended.identity_id, "reason": "expired"},
        )
        if completing is not None:
            await self._verification_hook(
                HookTrigger.ON_VERIFICATION_ENDED, completing, {"reason": "expired"}
            )
        return ended

    async def _completing_request(self, current: Verification) -> VerificationRequest | None:
        """The request that completed the verification in force, if any."""
        if current.status != VerificationStatus.VERIFIED or current.request_id is None:
            return None
        return await self._store.get_verification_request(current.request_id)

    async def _verification_hook(
        self, trigger: HookTrigger, request: VerificationRequest, data: dict[str, Any]
    ) -> None:
        """Fire a verification hook in the room the request was opened in (ASYNC)."""
        await self._fire_lifecycle_hook(
            request.room_id,
            trigger,
            EventType.SYSTEM,
            code=str(trigger),
            message=f"Verification of {request.identity_id}",
            data={
                "identity_id": request.identity_id,
                "organization_id": request.organization_id,
                "request_id": request.id,
                "channel_id": request.channel_id,
                **data,
            },
        )

    async def _attempt(
        self, request_id: str, secret: str, method: str | None
    ) -> tuple[str, VerificationRequest, Verification]:
        """One attempt at a request, under the identity's lock: (outcome, request, state)."""
        request = await self._store.get_verification_request(request_id)
        if request is None:
            raise VerificationRefusedError("not_found")
        current = await self._stored_verification(request.identity_id, request.organization_id)
        if request.status != VerificationRequestStatus.OPEN:
            return "not_open", request, current
        now = datetime.now(UTC)
        if request.expires_at <= now:
            request = request.model_copy(update={"status": VerificationRequestStatus.EXPIRED})
            await self._store.update_verification_request(request)
            await self._settle_pending(current, request)
            return "expired", request, current
        if not hmac.compare_digest(request.secret_hash, await _secret_hash(request.id, secret)):
            attempts = request.attempts + 1
            failed = attempts >= request.max_attempts
            status = VerificationRequestStatus.FAILED if failed else VerificationRequestStatus.OPEN
            request = request.model_copy(update={"attempts": attempts, "status": status})
            await self._store.update_verification_request(request)
            if failed:
                await self._settle_pending(current, request)
                return "failed", request, current
            return "mismatch", request, current
        request = request.model_copy(update={"status": VerificationRequestStatus.COMPLETED})
        await self._store.update_verification_request(request)
        verification = Verification(
            identity_id=request.identity_id,
            organization_id=request.organization_id,
            status=VerificationStatus.VERIFIED,
            method=method or request.method,
            verified_via=request.channel_id,
            verified_at=now,
            expires_at=now + timedelta(seconds=self._verification_policy.ttl),
            request_id=request.id,
        )
        await self._store.put_verification(verification)
        return "completed", request, verification

    async def _settle_pending(self, current: Verification, request: VerificationRequest) -> None:
        """Settle the identity once its request ended unfulfilled.

        An expired request proves nothing either way: the identity is left as
        it was before it. A failed one is someone who could not prove it, so a
        verification in force ends with it (fail closed): asking a verified
        person again, on a suspicion, and getting wrong secrets must not leave
        the old verification standing.
        """
        if current.request_id != request.id:
            return
        update: dict[str, Any] = {"request_id": None}
        if request.status == VerificationRequestStatus.FAILED:
            update.update(status=VerificationStatus.UNVERIFIED, ended_reason="failed")
        elif current.status == VerificationStatus.PENDING:
            update["status"] = VerificationStatus.UNVERIFIED
        await self._store.put_verification(current.model_copy(update=update))

    async def _cancel_open_request(self, current: Verification) -> VerificationRequest | None:
        """Cancel the identity's open request, if it has one; return it to announce."""
        if current.request_id is None:
            return None
        request = await self._store.get_verification_request(current.request_id)
        if request is None or request.status != VerificationRequestStatus.OPEN:
            return None
        cancelled = request.model_copy(update={"status": VerificationRequestStatus.CANCELLED})
        await self._store.update_verification_request(cancelled)
        return cancelled

    async def _stored_verification(
        self, identity_id: str, organization_id: str | None
    ) -> Verification:
        stored = await self._store.get_verification(identity_id, organization_id)
        if stored is not None:
            return stored
        return Verification(identity_id=identity_id, organization_id=organization_id)

    async def _emit_verification_event(
        self, event_type: str, request: VerificationRequest, *, reason: str | None = None
    ) -> None:
        data: dict[str, Any] = {
            "identity_id": request.identity_id,
            "request_id": request.id,
            "method": request.method,
            "expires_at": request.expires_at.isoformat(),
        }
        if reason is not None:
            data["reason"] = reason
        await self._emit_framework_event(
            event_type, room_id=request.room_id, channel_id=request.channel_id, data=data
        )


def _code(length: int) -> str:
    """A code of *length* digits, drawn uniformly."""
    return f"{secrets.randbelow(10**length):0{length}d}"


async def _secret_hash(request_id: str, secret: str) -> str:
    """The stored form of a secret: scrypt, salted with its request's id.

    Computed off the event loop: tens of milliseconds of CPU on purpose.
    """
    digest = await asyncio.to_thread(
        hashlib.scrypt,
        secret.encode(),
        salt=request_id.encode(),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=64 * 1024 * 1024,
        dklen=32,
    )
    return f"scrypt${digest.hex()}"


def _in_force(verification: Verification, now: datetime) -> bool:
    return (
        verification.status == VerificationStatus.VERIFIED
        and verification.expires_at is not None
        and verification.expires_at > now
    )


def _lock_key(identity_id: str, organization_id: str | None) -> str:
    """One identity's transitions are serialised under this key (any lock manager)."""
    return f"verification:{organization_id or ''}:{identity_id}"


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _forget_timer(
    timers: dict[str, asyncio.Task[None]], key: str, done: asyncio.Task[None]
) -> None:
    """Drop a finished timer, unless a newer one took its key."""
    if timers.get(key) is done:
        del timers[key]


def _sender_identity(
    event: RoomEvent, context: RoomContext, resolved_identity: Identity | None
) -> tuple[str, str | None] | None:
    """(identity_id, organization_id) of an identified sender; None otherwise.

    The resolver's answer, or the room's IDENTIFIED participant when the
    sender was already identified and resolution was skipped (RFC §11.6).
    """
    organization_id = context.room.organization_id
    if resolved_identity is not None:
        return resolved_identity.id, resolved_identity.organization_id or organization_id
    participant_id = event.source.participant_id
    for participant in context.participants:
        if (
            participant.id == participant_id
            and participant.identification == IdentificationStatus.IDENTIFIED
            and participant.identity_id
        ):
            return participant.identity_id, organization_id
    return None


def _held(event: RoomEvent) -> RoomEvent:
    """*event* held from the room's agents; its people and transports still see it."""
    return event.model_copy(
        update={
            "visibility": Visibility.TRANSPORT,
            "metadata": {**event.metadata, HELD_KEY: {"visibility": event.visibility}},
        }
    )
