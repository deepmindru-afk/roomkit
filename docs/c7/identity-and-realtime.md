# Identity and Realtime Events

## Identity Resolution

The identity pipeline maps external sender IDs to known participants. It runs as part of the inbound pipeline, after `handle_inbound()` and before hooks.

### How It Works

```
Inbound message arrives with sender_id
  -> IdentityResolver.resolve(sender_id, channel_type)
  -> Returns IdentityResult with status:
     IDENTIFIED      -> participant_id stamped on event, processing continues
     AMBIGUOUS       -> ON_IDENTITY_AMBIGUOUS hook fires
     PENDING         -> ON_IDENTITY_AMBIGUOUS hook fires
     UNKNOWN         -> ON_IDENTITY_UNKNOWN hook fires
     REJECTED        -> ON_IDENTITY_UNKNOWN hook fires
```

### Identity Hooks

```python
from roomkit import RoomKit, HookTrigger
from roomkit.models import Identity, IdentityHookResult

kit = RoomKit()

@kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
async def handle_unknown(event, ctx):
    # Option 1: Resolve to a known identity
    return IdentityHookResult.resolved(Identity(
        id="user-123",
        display_name="Alice",
    ))

    # Option 2: Challenge the sender to identify
    return IdentityHookResult.challenge(inject=InjectedEvent(
        content=TextContent(body="Please provide your account number."),
    ))

    # Option 3: Reject the message
    return IdentityHookResult.reject("Unknown sender")

    # Option 4: Keep as pending
    return IdentityHookResult.pending(candidates=[...])
```

### Custom Identity Resolver

```python
from roomkit.identity import IdentityResolver
from roomkit.models import IdentificationStatus, Identity, IdentityResult

class DatabaseIdentityResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        user = await db.find_by_phone(message.sender_id)
        if user:
            return IdentityResult(
                status=IdentificationStatus.IDENTIFIED,
                identity=Identity(id=user.id, display_name=user.name),
            )
        return IdentityResult(status=IdentificationStatus.UNKNOWN)

kit = RoomKit(identity_resolver=DatabaseIdentityResolver())
```

### Manual Resolution

```python
# Resolve a pending participant to a known identity
await kit.resolve_participant(
    room_id="room-1",
    participant_id="pending-123",
    identity_id="user-456",
)
```

### Identity Verification

Identification says whose an address is; verification says the person writing
has proven it (RFC §11.7). With `verification_policy=VerificationPolicy()`,
every message of an identified sender carries `event.source.verification`, and
an unverified one fires `ON_VERIFICATION_REQUIRED` (SYNC): `HookResult.allow()`
holds it from the agents (the room's transports still see it),
`HookResult.allow_unverified()` passes it, `block()` refuses it. The method
(a page behind a link, a code replied in the conversation) is the integrator's.

```python
kit = RoomKit(identity_resolver=resolver, verification_policy=VerificationPolicy())

@kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
async def ask_to_verify(event, ctx) -> HookResult:
    state = await kit.sender_verification(event, ctx)  # identity_id, request_id
    if state is not None and state.request_id is None:
        request, secret = await kit.request_verification(
            state.identity_id, room_id=event.room_id,
            channel_id=event.source.channel_id, window_start=event.created_at,
        )
        ...  # send the link after the hook returns (the room is locked)
    return HookResult.allow()

# The integrator's page, after its own check:
await kit.complete_verification(request_id, secret)  # or record_failed_attempt(request_id)
await kit.revoke_verification(identity_id, reason="advisor")
```

On completion the held messages of the request's window are released to the
agents as the author's own turns. A held message whose delivery was cancelled
is released with no answer: when it is the last released, no agent answers,
and the earlier ones are not answered in its place. Its `delivery_key` names
it until then (`kit.cancel_delivery(key)`); its handle
(`defer_delivery=True`) cancels it only while its delivery set runs. The cancellation is written on the held event, so a release in
another process honours it. `ON_VERIFICATION_COMPLETED` and
`ON_VERIFICATION_ENDED` (ASYNC; `ended` is `"verification"` or `"request"`)
fire when it happens; a timer ends a verification at its expiry, and a
request left unanswered at its own (the identity then has none open, so the
next message asks again); `async with RoomKit(...)` re-arms the verifications'
timers from the store. A code reply
(`method="code"`) is consumed, never stored.

## Realtime Ephemeral Events

Ephemeral events (typing, presence, reactions) are not stored in conversation history. They're delivered in real-time to subscribers.

### Publishing Events

```python
from roomkit import RoomKit

kit = RoomKit()

# Typing indicator
await kit.publish_typing("room-1", "alice", is_typing=True)
await kit.publish_typing("room-1", "alice", is_typing=False)

# Presence
await kit.publish_presence("room-1", "alice", "online")   # online/away/offline

# Reaction
await kit.publish_reaction("room-1", "alice", target_event_id="evt-123", emoji="thumbsup")

# Read receipt
await kit.publish_read_receipt("room-1", "alice", event_id="evt-123")

# Tool call events (AIChannel publishes these automatically)
from roomkit.realtime import EphemeralEventType

await kit.publish_tool_call("room-1", "ai-agent", [
    {"id": "tc1", "name": "search", "arguments": {"q": "test"}}
], EphemeralEventType.TOOL_CALL_START)
```

### Subscribing to Events

```python
async def on_ephemeral(event):
    print(f"Ephemeral: {event.type} from {event.user_id}")

sub_id = await kit.subscribe_room("room-1", on_ephemeral)

# Unsubscribe later
await kit.unsubscribe_room(sub_id)
```

Each `publish_*` call and `subscribe_room` take `organization_id=` (RFC
§17.2): another organization's room raises `RoomNotFoundError`, nothing
published or subscribed. Unset, no room is read: a room id is published to
and subscribed to as given, as before.

### Event Types

| Type | Description |
|------|-------------|
| `TYPING_START` | User started typing |
| `TYPING_STOP` | User stopped typing |
| `PRESENCE_ONLINE` | User came online |
| `PRESENCE_AWAY` | User went away |
| `PRESENCE_OFFLINE` | User went offline |
| `READ_RECEIPT` | User read a message |
| `REACTION` | User reacted to a message |
| `TOOL_CALL_DELTA` | AI is composing a tool call (streaming only) |
| `TOOL_CALL_START` | AI started executing a tool |
| `TOOL_CALL_END` | AI finished executing a tool |
| `THINKING_START` / `THINKING_DELTA` / `THINKING_END` | AI reasoning, coalesced |
| `TEXT_DELTA` | AI answer text as it is written, coalesced (`text`, `round`, `channel_id`) |
| `CUSTOM` | Custom ephemeral event |

AI channels publish `TOOL_CALL_DELTA` with the tool name and cumulative
`arguments_chars`, never the argument content. A terminal frame with an empty
`tool_calls` closes every composition attempt, including cancellation, provider
failure, retry, and fallback; counts restart for a retry. The complete call
still arrives in `TOOL_CALL_START`.

AI channels publish `TEXT_DELTA` with the answer's text as it is written, for
an observer that is not a transport of the room: a debugger, a dashboard, or
the code that delegated a task, which subscribes to the child room (its id is
in `ON_TASK_DELEGATED`'s event metadata, `child_room_id`, before the agent
starts). The text is coalesced on `AIChannel(text_coalesce_ms=80,
text_coalesce_chars=256)` (`text_coalesce_ms=0` publishes every token), and
flushed before a round's tool-call events, so text and calls arrive in order.
Joined, the texts are every round's answer text (`round` tells them apart),
where a delegated task's result holds the last segment only. A client that
already renders the answer from a streaming transport (a WebSocket) reads one
or the other, not both. Never persisted: the stored message is the answer.

A turn's ephemerals (its reasoning, its tool calls, a custom one its channel
publishes, and what work it began publishes after it) reach a room's
subscribers only when the turn's answer reaches the whole room: the agent can
write, its binding's visibility is `all` (read at each publish, so narrowing it
mid-turn stops them at once), and the triggering message's
`response_visibility` restricts nothing (RFC §8.4). Subscribers are not bound
channels, so an agent whose answers are scoped elsewhere (a supervisor's
whisper, an internal agent, a read-only agent) publishes none of them.

### Read Tracking

```python
# Mark a specific event as read
await kit.mark_read("room-1", "ws-user", "evt-123")

# Mark all events as read
await kit.mark_all_read("room-1", "ws-user")
```

### Custom Realtime Backend

The default `InMemoryRealtime` works for single-process deployments. For distributed systems, implement `RealtimeBackend`:

```python
from roomkit.realtime import RealtimeBackend

class RedisRealtimeBackend(RealtimeBackend):
    # Implement publish, subscribe, unsubscribe using Redis Pub/Sub
    ...

kit = RoomKit(realtime=RedisRealtimeBackend())
```
