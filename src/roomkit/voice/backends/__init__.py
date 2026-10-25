"""Voice transport backends.

Each backend is imported from here. A backend's module loads on first use,
with the transport library it needs; ``import roomkit`` loads none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.voice.backends.mock import MockVoiceBackend, MockVoiceCall

if TYPE_CHECKING:
    from roomkit.voice.backends._sip_types import PT_G722, PT_PCMA, PT_PCMU
    from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend, BuzzHuddleWatcher
    from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice
    from roomkit.voice.backends.local import LocalAudioBackend
    from roomkit.voice.backends.rtp import RTPVoiceBackend
    from roomkit.voice.backends.sip import SIPVoiceBackend
    from roomkit.voice.backends.twilio_ws import TwilioWebSocketBackend
    from roomkit.voice.backends.webtransport import WebTransportBackend
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "BuzzHuddleBackend": "roomkit.voice.backends.buzz_huddle",
            "BuzzHuddleWatcher": "roomkit.voice.backends.buzz_huddle",
            "FastRTCVoiceBackend": "roomkit.voice.backends.fastrtc",
            "LocalAudioBackend": "roomkit.voice.backends.local",
            "PT_G722": "roomkit.voice.backends._sip_types",
            "PT_PCMA": "roomkit.voice.backends._sip_types",
            "PT_PCMU": "roomkit.voice.backends._sip_types",
            "RTPVoiceBackend": "roomkit.voice.backends.rtp",
            "SIPVoiceBackend": "roomkit.voice.backends.sip",
            "TwilioWebSocketBackend": "roomkit.voice.backends.twilio_ws",
            "WebTransportBackend": "roomkit.voice.backends.webtransport",
            "mount_fastrtc_voice": "roomkit.voice.backends.fastrtc",
        },
    )

__all__ = [
    "BuzzHuddleBackend",
    "BuzzHuddleWatcher",
    "FastRTCVoiceBackend",
    "LocalAudioBackend",
    "MockVoiceBackend",
    "MockVoiceCall",
    "PT_G722",
    "PT_PCMA",
    "PT_PCMU",
    "RTPVoiceBackend",
    "SIPVoiceBackend",
    "TwilioWebSocketBackend",
    "WebTransportBackend",
    "mount_fastrtc_voice",
]
