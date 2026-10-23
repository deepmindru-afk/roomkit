# Rooms and Channels

## Creating Rooms

```python
from roomkit import RoomKit

kit = RoomKit()

# Auto-generated ID
room = await kit.create_room()

# Explicit ID with metadata
room = await kit.create_room(
    room_id="support-123",
    metadata={"topic": "billing", "priority": "high"},
)
```

A room is created once (RFC §14.3): `create_room()` with an id the store
already holds raises `RoomAlreadyExistsError` and leaves the room as it was.
Read the room first (`get_room()`) when the id may exist.

## Room Lifecycle

```python
from roomkit import RoomKit

kit = RoomKit()

# Create room
room = await kit.create_room(room_id="session-1")

# Update metadata
await kit.update_room_metadata("session-1", {"status": "escalated"})

# Close a room
await kit.close_room("session-1")

# Check timers (call periodically, e.g. every 60s)
transitioned = await kit.check_all_timers()
```

## Channel Types

The `ChannelType` enum defines 23 values: `SMS`, `MMS`, `RCS`, `EMAIL`, `WHATSAPP`, `WHATSAPP_PERSONAL`, `WEBSOCKET`, `AI`, `VOICE`, `REALTIME_VOICE`, `REALTIME_AUDIO_VIDEO`, `PUSH`, `MESSENGER`, `TELEGRAM`, `TEAMS`, `DISCORD`, `BUZZ`, `WEBHOOK`, `VIDEO`, `AUDIO_VIDEO`, `CONFERENCE`, `CLI`, `SYSTEM`. RoomKit ships with these factories and channel classes:

| Channel | Factory / Class | Category | Use Case |
|---------|----------------|----------|----------|
| SMS | `SMSChannel(id, provider=...)` | TRANSPORT | Text messages via Twilio, Telnyx, Sinch, VoiceMeUp |
| RCS | `RCSChannel(id, provider=..., fallback=True)` | TRANSPORT | Rich messaging via Twilio, Telnyx |
| Email | `EmailChannel(id, provider=...)` | TRANSPORT | Email via ElasticEmail, SendGrid |
| WhatsApp | `WhatsAppChannel(id, provider=...)` | TRANSPORT | WhatsApp Business API |
| WhatsApp Personal | `WhatsAppPersonalChannel(id, provider=...)` | TRANSPORT | WhatsApp via neonize |
| Messenger | `MessengerChannel(id, provider=...)` | TRANSPORT | Facebook Messenger |
| Telegram | `TelegramChannel(id, provider=...)` | TRANSPORT | Telegram Bot API |
| Teams | `TeamsChannel(id, provider=...)` | TRANSPORT | Microsoft Teams Bot Framework |
| Discord | `DiscordChannel(id, provider=...)` | TRANSPORT | Discord bot (recipient key `discord_channel_id`) |
| Buzz | `BuzzChannel(id, provider=...)` | TRANSPORT | Nostr relay channels (recipient key `buzz_channel_id`) |
| Webhook | `HTTPChannel(id, provider=...)` | TRANSPORT | Generic HTTP webhook (`ChannelType.WEBHOOK`) |
| WebSocket | `WebSocketChannel(id)` | TRANSPORT | Real-time bidirectional |
| CLI | `CLIChannel(id)` | TRANSPORT | Interactive terminal (stdin/stdout REPL) |
| AI | `AIChannel(id, provider=...)` | INTELLIGENCE | LLM responses |
| Agent | `Agent(id, provider=...)` | INTELLIGENCE | Agent with tools + greeting |
| ACP | `ACPChannel(id, command, cwd=...)` | INTELLIGENCE | External coding agent via Agent Client Protocol (`channel_type` is `AI`; one session per room) |
| Voice | `VoiceChannel(id, stt=..., tts=..., backend=...)` | TRANSPORT | Real-time audio |
| Realtime Voice | `RealtimeVoiceChannel(id, provider=...)` | TRANSPORT | Speech-to-speech AI |
| Video | `VideoChannel(id, backend=..., vision=...)` | TRANSPORT | Video track with vision analysis |
| Audio+Video | `AudioVideoChannel(id, backend=..., stt=..., tts=...)` | TRANSPORT | Combined voice + video |
| Realtime Audio+Video | `RealtimeAudioVideoChannel(id, provider=..., transport=...)` | TRANSPORT | Speech-to-speech AI with video |
| Conference | `ConferenceChannel(id, backend=...)` | TRANSPORT | Multi-party SFU conference with AI bot (LiveKit or mock backend) |

`MMS`, `PUSH`, and `SYSTEM` have no factory: `MMS` is set automatically when an SMS channel carries media, `SYSTEM` marks framework-generated events, and `PUSH` is reserved.

## Registering and Attaching Channels

Channels are registered globally, then attached to specific rooms:

```python
from roomkit import RoomKit, SMSChannel, ChannelCategory
from roomkit.channels.ai import AIChannel
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

kit = RoomKit()

# Create and register channels
sms = SMSChannel("sms-main", provider=TwilioSMSProvider(TwilioConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+1234567890",
)))
ai = AIChannel("ai-agent", provider=AnthropicAIProvider(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-opus-5",
)))

kit.register_channel(sms)
kit.register_channel(ai)

# Create room and attach
await kit.create_room(room_id="support-room")
await kit.attach_channel("support-room", "sms-main")
await kit.attach_channel("support-room", "ai-agent", category=ChannelCategory.INTELLIGENCE)
```

## Channel Access Levels

Control what a channel can do in a room:

```python
from roomkit import Access

# Default: read and write
await kit.attach_channel("room", "channel", access=Access.READ_WRITE)

# Read only: receives messages but cannot send
await kit.attach_channel("room", "channel", access=Access.READ_ONLY)

# Write only: sends messages but doesn't receive broadcasts
await kit.attach_channel("room", "channel", access=Access.WRITE_ONLY)

# None: temporarily disabled
await kit.set_access("room", "channel", Access.NONE)
```

## Muting Channels

Muting suppresses a channel's output (AI responses) without detaching it. Side effects (tasks, observations) still fire.

```python
# Mute AI output
await kit.mute("room", "ai-agent")

# Unmute
await kit.unmute("room", "ai-agent")

# Mute only the output direction (AI won't respond, but still sees messages)
await kit.mute_output("room", "ai-agent")
await kit.unmute_output("room", "ai-agent")
```

## Per-Room Channel Configuration

Pass metadata when attaching to customize per-room behavior:

```python
await kit.attach_channel(
    "weather-room",
    "ai-agent",
    category=ChannelCategory.INTELLIGENCE,
    metadata={
        "system_prompt": "You are a weather assistant.",
        "temperature": 0.3,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        ],
    },
)
```

## Content Types

Events carry typed content. RoomKit supports 11 content types, all discriminated unions on the `type` field:

| Type | `type` Literal | Key Fields | Use Case |
|------|---------------|-----------|----------|
| `TextContent` | `"text"` | `body`, `language` | Plain text messages |
| `RichContent` | `"rich"` | `body`, `format` (html/markdown), `plain_text`, `buttons`, `cards`, `quick_replies` | Formatted messages with UI elements |
| `MediaContent` | `"media"` | `url`, `mime_type`, `filename`, `size_bytes`, `caption` | Images, documents, files |
| `AudioContent` | `"audio"` | `url`, `mime_type`, `duration_seconds`, `transcript` | Voice messages |
| `VideoContent` | `"video"` | `url`, `mime_type`, `duration_seconds`, `thumbnail_url` | Video messages |
| `LocationContent` | `"location"` | `latitude`, `longitude`, `label`, `address` | Geographic coordinates |
| `CompositeContent` | `"composite"` | `parts` (list of EventContent, max depth 5) | Multi-part messages |
| `TemplateContent` | `"template"` | `template_id`, `language`, `parameters`, `body` | WhatsApp Business / RCS templates |
| `SystemContent` | `"system"` | `body`, `code`, `data` (dict) | System-generated events |
| `EditContent` | `"edit"` | `target_event_id`, `new_content`, `edit_source` | Message edits |
| `DeleteContent` | `"delete"` | `target_event_id`, `delete_type`, `reason` | Message deletion |

```python
from roomkit.models.event import (
    TextContent, RichContent, MediaContent, AudioContent, VideoContent,
    LocationContent, CompositeContent, TemplateContent, SystemContent,
    EditContent, DeleteContent,
)

# Plain text
TextContent(body="Hello!")

# Rich content with buttons and Markdown
RichContent(body="Choose:", format="markdown", plain_text="Choose:", buttons=[{"text": "A", "payload": "a"}])

# Media, Audio, Location, Template
MediaContent(url="https://example.com/image.png", mime_type="image/png", caption="Photo")
AudioContent(url="https://example.com/voice.ogg", mime_type="audio/ogg", transcript="Hello there")
LocationContent(latitude=40.7128, longitude=-74.0060, label="NYC Office")
TemplateContent(template_id="order_confirm", language="en", parameters={"1": "ORD-123"})

# Multi-part (max depth 5)
CompositeContent(parts=[
    TextContent(body="Here's the photo:"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg"),
])

# Edit and Delete
EditContent(target_event_id="evt-abc", new_content=TextContent(body="Corrected text"))
DeleteContent(target_event_id="evt-abc", delete_type="sender", reason="User retracted")
```

## Content Transcoding

When broadcasting events, the EventRouter automatically **transcodes** content to match each target channel's capabilities. A channel that only supports TEXT will receive a text fallback of a RichContent message.

### How Transcoding Works

1. Event broadcast starts → EventRouter iterates over all target channel bindings
2. For each target, call `ContentTranscoder.transcode(content, source_binding, target_binding)`
3. Transcoder checks `target_binding.capabilities.media_types` to decide if content passes through or needs conversion
4. If transcode returns `None` → delivery is skipped for that channel

### Channel Capabilities

Each channel binding declares what content types it supports via `ChannelCapabilities`:

```python
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType

# SMS channel: text only, 160 char limit
sms_caps = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT],
    max_length=160,
)

# WebSocket channel: rich content, media, audio, video
ws_caps = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT, ChannelMediaType.RICH, ChannelMediaType.MEDIA,
        ChannelMediaType.AUDIO, ChannelMediaType.VIDEO, ChannelMediaType.LOCATION,
    ],
    supports_edit=True,
    supports_delete=True,
)
```

`ChannelMediaType` enum: `TEXT`, `RICH`, `MEDIA`, `AUDIO`, `VIDEO`, `LOCATION`, `TEMPLATE`.

### Default Fallback Chain

| Content Type | If Target Supports It | Fallback |
|-------------|----------------------|----------|
| `TextContent` | Always passes through | — |
| `RichContent` | RICH in media_types → pass | `TextContent(plain_text or body)` |
| `MediaContent` | MEDIA in media_types → pass | `TextContent("[Media: {caption or filename or url}]")`, the URL only when a link, else the media type |
| `AudioContent` | AUDIO in media_types → pass | `TextContent(transcript)` or `"[Voice message: {url}]"`, the URL only when a link, else the media type |
| `VideoContent` | VIDEO in media_types → pass | `TextContent("[Video: {url}]")`, the URL only when a link, else the media type |
| `LocationContent` | LOCATION in media_types → pass | `TextContent("[Location: {label} ({lat}, {lon})]")` |
| `CompositeContent` | Recursive transcode of all parts | If all parts become text → flatten to single TextContent |
| `TemplateContent` | TEMPLATE in media_types → pass | `TextContent(body or "[Template: {id}]")` |
| `EditContent` | `supports_edit` → pass | Transcode `new_content` + prefix "Correction:" |
| `DeleteContent` | `supports_delete` → pass | `TextContent("[Message deleted]")` |

### Multichannel Transcoding Example

A single event gets adapted differently for each channel:

```python
# User sends a rich message with an image from WebSocket
content = CompositeContent(parts=[
    RichContent(body="**Check this out!**", format="markdown", plain_text="Check this out!"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg", caption="Sunset"),
])

# WebSocket channel: receives CompositeContent as-is (supports RICH + MEDIA)
# SMS channel: receives TextContent("Check this out!\n[Media: Sunset]")
# Voice channel (TTS): receives TextContent("Check this out!\n[Media: Sunset]")
```

### Custom Content Transcoder

Implement the `ContentTranscoder` ABC to customize how content is adapted:

```python
from roomkit.core.router import ContentTranscoder
from roomkit.models.channel import ChannelBinding
from roomkit.models.event import EventContent, TextContent, MediaContent
from roomkit.models.enums import ChannelMediaType

class MyTranscoder(ContentTranscoder):
    async def transcode(
        self,
        content: EventContent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> EventContent | None:
        target_types = target_binding.capabilities.media_types

        # Custom: convert images to descriptive text for voice channels
        if isinstance(content, MediaContent) and content.mime_type.startswith("image/"):
            if ChannelMediaType.MEDIA not in target_types:
                caption = content.caption or "an image"
                return TextContent(body=f"[Image received: {caption}]")

        # Custom: truncate long text for SMS
        if isinstance(content, TextContent):
            max_len = target_binding.capabilities.max_length
            if max_len and len(content.body) > max_len:
                return TextContent(body=content.body[: max_len - 3] + "...")

        return content

# Override the default transcoder on the RoomKit instance
kit = RoomKit()
kit._transcoder = MyTranscoder()
```

## Edits and Deletes

An event is an edit when its type is `EDIT` and its content an `EditContent`,
a delete when `DELETE` and `DeleteContent` (RFC §10.3). The pipeline checks it
under the room lock on every pass that commits an event: an inbound message, an
agent's response, a hook's injected event, a `send_event()`. A refusal stores
nothing: the inbound path returns `InboundResult(blocked=True, reason=...)`,
`send_event()` raises `ChangeRefusedError` (its `reason`), a refused response or
injected change is logged at warning level.

| Rule | Refused with |
|------|--------------|
| A type and a content that disagree, or a change held elsewhere (a composite's part, an edit's new content) | `target_event_not_found` |
| The target is not an event of the room, is `BLOCKED`, deleted, or itself an edit or a delete (an edited event may be edited again) | `target_event_not_found` |
| A sender's change (`edit_source` `"sender"` or none, `DeleteType.SENDER`) of an event they did not author | `not_original_author`; `identity_required_for_edit` when either has no author |
| `DeleteType.ADMIN` from anyone but an OWNER or AGENT, neither LEFT nor BANNED, on the channel they write on | `not_authorized`; `identity_required_for_edit` with no sender id |
| A SYSTEM change (`DeleteType.SYSTEM`, `edit_source="system"`) from anything but a SYSTEM channel or a hook | `not_authorized` |
| An edit or a delete a hook injects that is not a SYSTEM change | `not_authorized` |

An event that travelled outbound (a channel's own output, an agent's response)
is its channel's, whatever participant it names: an agent edits its own
responses and no one else's. Any `edit_source` other than `"system"` is checked
as the sender's. A hook's `modify` keeps the target, the `edit_source` or
`delete_type` and the kind of content; once a hook ran under the lock, the
target is read and checked again, so a delete committed during an edit's hooks
refuses the edit.

An accepted change commits with its target in one store transaction: an edit
writes `new_content` and `metadata.edited=True` on the target, a delete
`metadata.deleted=True`. The change reaches no one who could not see its target
(its visibility is narrowed to the target's), it is delivered with the target as
the commit left it, and `ON_EVENT_UPDATED` / `ON_EVENT_DELETED` fire once the
lock is released.

```python
await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="alice",
        event_type=EventType.EDIT,
        content=EditContent(target_event_id=message.id, new_content=TextContent(body="50000$")),
    )
)
```

A soft-deleted event stays in every history a channel is handed, its content
withheld: an empty text in place of what was said and no `source.raw_payload`,
its id, index, thread, metadata and `channel_data` kept, so the delete that
follows names an event the reader holds. An edit of a deleted event has its new
content withheld alike. `visible_events()`, the context a channel parses a
message with and the one a broadcast delivers with all hold it so; the stored
record and the contexts hooks receive keep the content.

`kit.delete_event(room_id, event_id, cascade_replies=True)` is the host's hard
delete: it removes the event, its thread replies unless told not to, and the
edits and deletes of every event it removes (an edit holds the corrected text),
returns their ids, and fires `ON_EVENT_DELETED` once per removed event.
`kit.update_event()` rewrites an event's content, source or metadata. Neither
runs the checks above: the host owns their authorization.

## Detaching Channels

```python
await kit.detach_channel("room", "channel-id")
```

## Binding Metadata Updates

```python
await kit.update_binding_metadata("room", "ai-agent", {"temperature": 0.9})
```

## Querying Rooms

```python
# Get a room
room = await kit.get_room("room-id")

# List bindings
bindings = await kit.store.list_bindings("room-id")

# List participants
participants = await kit.store.list_participants("room-id")

# Query event timeline
events = await kit.get_timeline("room-id", offset=0, limit=50)
```

A participant is one shared record per `(room_id, participant_id)`, even when
`connected_via` names several channels. Its `status` is therefore global to the
record, not per-channel presence. When two channel memberships have independent
connect/disconnect lifecycles, give them distinct participant ids and correlate
them through `identity_id`.

An inbound sender id names its participant only on a channel in their
`connected_via` (RFC §5.5): the same id arriving through another channel is
another sender, with none of the participant's name, role, administrative
authority, authorship, identification or verification: routed without a room
id it gets a room of its own, not the participant's, and
`current_tool_actor_id()` reads `None` for it. Link one person's
channels with `add_member` (or `join`) on each, or let the identity pipeline
resolve them, which records the channel it resolved them on:

```python
await kit.add_member("room-id", "console", "u-1", role=ParticipantRole.OWNER)
# A message on "web" from sender id "u-1": labelled "@web", no role, no ADMIN delete.
await kit.add_member("room-id", "web", "u-1")  # records "web"; the role stays OWNER
# Now the same message on "web" is the owner's.
```

## Cancelling a turn

Cancelling a message's delivery stops its agents' turns and only those: a turn
still waiting in the room's lane (or behind the room's ACP prompt) never
starts, a running one is cut with its tools, and nothing answers after. The
message itself is still delivered to the room's other channels and fires its
`AFTER_BROADCAST` hooks. The room's other turns go on.

Cancelling a task awaiting `kit.process_inbound()` cancels and drains that
call's tools, generation and stream consumers before propagating
`asyncio.CancelledError`. Other rooms and shared providers remain usable.
Committed events and already streamed text remain in the timeline.

With `defer_delivery=True`, explicitly call
`await result.delivery.cancel(reason="hangup", timeout=5.0)` to stop that turn.
The returned result's `cancellation_reason` names its terminal cancellation;
`await result.delivery.wait()` joins the same cleanup. Cancelling a `wait()`
alone leaves deferred work running. A completed handle is unchanged by cancel.

The default cleanup budget is five seconds on both paths. `TimeoutError`
means cleanup is incomplete: keep resources still in use, and for a deferred
call await its handle later. Repeated cancellation does not interrupt
finalizers. A handle cannot drain its own room lock or lane (`RuntimeError`).

Code without the handle names the delivery by a key the caller set on the
message, `InboundMessage(..., delivery_key="task-42")`, and cancels it from
anywhere in the process, a tool handler included:
`await kit.cancel_delivery("task-42", reason="lead_cancelled")`. It answers how
many deliveries the key named and cancelled (0 once they are over),
returns at once, and each call's caller reads `cancellation_reason` on its
result. The key names the delivery from the start of `process_inbound`: a
message cancelled before its commit (while it is routed, parsed, or its sender
resolved) is committed and delivered all the same, and its turns never start.
A message held for verification stays named until its release is answered: a
cancel while it waits counts 1, and its release gets no answer.
Several messages may share a key. The key is never stored and grants
nothing: check who may cancel before calling, and pass `organization_id=` so a
key two tenants both chose only reaches the caller's rooms. Pass it to
`process_inbound` as well: a call without it belongs to no organization until
it reads its room (right after routing), and a scoped cancel reaching it
before is applied then, uncounted. A call refused before its commit reports
its refusal, not the cancellation. A `kit.steer(Cancel(), ...)` reaches only
turns that have started; use the delivery to stop one message's turns.
External effects already committed and tasks detached by application handlers
are outside this cancellation contract. See `examples/cancel_delivery.py`.
