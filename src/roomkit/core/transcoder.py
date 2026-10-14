"""Default content transcoder with fallback conversions."""

from __future__ import annotations

from roomkit._media import audio_label, media_label, video_label
from roomkit.core.router import ContentTranscoder
from roomkit.models.channel import ChannelBinding
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
            if ChannelMediaType.MEDIA in target_types:
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
            label = content.label or content.address or ""
            return TextContent(
                body=f"[Location: {label} ({content.latitude}, {content.longitude})]"
            )

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
                return content
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
