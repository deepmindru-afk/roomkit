"""Default content transcoder with fallback conversions."""

from __future__ import annotations

from roomkit._media import audio_label, location_label, media_label, video_label
from roomkit.core.router import ContentTranscoder
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    LocationContent,
    MediaContent,
    RichContent,
    TemplateContent,
    TextContent,
    VideoContent,
)


def takes_file(capabilities: ChannelCapabilities, media: MediaContent) -> bool:
    """Whether a channel of *capabilities* takes the file *media* (RFC §5.8):
    its type is listed by a non-empty ``supported_media_types``, compared
    without parameters and whatever the case, and its known size is within
    ``max_media_size_bytes``. An empty list takes every type; no limit, or a
    file of unknown size, any size. A file it does not take is delivered as
    its text form (§6.6): a provider's refusal would lose the whole message.
    """
    listed = {_bare_type(mime) for mime in capabilities.supported_media_types}
    if listed and _bare_type(media.mime_type) not in listed:
        return False
    limit = capabilities.max_media_size_bytes
    return limit is None or media.size_bytes is None or media.size_bytes <= limit


def takes_files_of(capabilities: ChannelCapabilities, content: EventContent) -> bool:
    """Whether a channel of *capabilities* takes every file *content* holds
    (RFC §5.8): a file itself, a composite's parts, an edit's new content."""
    if isinstance(content, MediaContent):
        return takes_file(capabilities, content)
    if isinstance(content, CompositeContent):
        return all(takes_files_of(capabilities, part) for part in content.parts)
    if isinstance(content, EditContent):
        return takes_files_of(capabilities, content.new_content)
    return True


def _bare_type(mime_type: str) -> str:
    """*mime_type* without its parameters, lower-case (``image/jpeg``)."""
    return mime_type.split(";", 1)[0].strip().lower()


class DefaultContentTranscoder(ContentTranscoder):
    """Transcodes content using simple fallback rules."""

    async def transcode(
        self,
        content: EventContent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> EventContent | None:
        """Transcode content for the target channel's capabilities.

        Returns ``None`` if the content cannot be represented at all
        on the target channel (signalling a transcoding failure).
        """
        target_types = target_binding.capabilities.media_types

        if isinstance(content, TextContent):
            return content

        if isinstance(content, RichContent):
            if ChannelMediaType.RICH in target_types:
                return content
            return TextContent(body=content.plain_text or content.body)

        if isinstance(content, MediaContent):
            if ChannelMediaType.MEDIA in target_types and takes_file(
                target_binding.capabilities, content
            ):
                return content
            return TextContent(body=media_label(content))

        if isinstance(content, AudioContent):
            if ChannelMediaType.AUDIO in target_types:
                return content
            return TextContent(body=audio_label(content))

        if isinstance(content, VideoContent):
            if ChannelMediaType.VIDEO in target_types:
                return content
            return TextContent(body=video_label(content))

        if isinstance(content, LocationContent):
            if ChannelMediaType.LOCATION in target_types:
                return content
            return TextContent(body=location_label(content))

        if isinstance(content, CompositeContent):
            parts: list[EventContent] = []
            for part in content.parts:
                transcoded = await self.transcode(part, source_binding, target_binding)
                if transcoded is None:
                    continue
                parts.append(transcoded)
            if not parts:
                return None
            # If all parts are text, flatten to single TextContent
            text_parts = [p for p in parts if isinstance(p, TextContent)]
            if len(text_parts) == len(parts):
                return TextContent(body="\n".join(p.body for p in text_parts))
            return CompositeContent(parts=parts)

        if isinstance(content, EditContent):
            if target_binding.capabilities.supports_edit:
                if takes_files_of(target_binding.capabilities, content.new_content):
                    return content
                # Delivered natively all the same (RFC §6.6), its new content
                # carrying a file the target does not take as its text form
                # (§5.8): never sent as is for the provider to refuse.
                new = await self.transcode(content.new_content, source_binding, target_binding)
                if new is None:
                    return None
                return content.model_copy(update={"new_content": new})
            # Fallback: transcode the new_content and prefix with "Correction:"
            new = await self.transcode(content.new_content, source_binding, target_binding)
            if new is None:
                return TextContent(body="[Correction]")
            if isinstance(new, TextContent):
                return TextContent(body=f"Correction: {new.body}")
            return new

        if isinstance(content, DeleteContent):
            if target_binding.capabilities.supports_delete:
                return content
            return TextContent(body="[Message deleted]")

        if isinstance(content, TemplateContent):
            if ChannelMediaType.TEMPLATE in target_types:
                return content
            return TextContent(body=content.body or f"[Template: {content.template_id}]")

        return content
